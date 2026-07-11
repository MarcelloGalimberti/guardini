# =============================================================================
# run_backtest.py — Rolling-origin cross-validation.
# Uso:
#   python run_backtest.py --data-dir <cartella dati> [--models base,v8] \
#       [--cutoffs 2025-09,2025-10,...] [--horizon 3]
# Output: results/backtest_detail.csv, results/backtest_summary.xlsx
# =============================================================================
import argparse
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_pivot, classify  # noqa: E402
from metrics import mase_scale, evaluate, summarize  # noqa: E402
import models as M  # noqa: E402

DEFAULT_CUTOFFS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venduto", default="VENDUTO_LUG 23_GIU 26.xlsx")
    ap.add_argument("--era-diventa", default="db_era_diventa_v2.xlsx")
    ap.add_argument("--promo", default="db_clienti_promo.xlsx")
    ap.add_argument("--escludere", default="ARTICOLI DA ESCLUDERE.xlsx")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--cutoffs", default=",".join(DEFAULT_CUTOFFS))
    ap.add_argument("--models", default="base,v8",
                    help="base = SeasonalNaive+MediaMobile3+SBA_v8_full | v8 = NP+SBA composito")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sf-models", default=None, help="lista modelli statsforecast, es. AutoETS,AutoTheta")
    args = ap.parse_args()

    dd = args.data_dir
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("Caricamento pipeline dati (replica v8)...", flush=True)
    pivot, anagrafica = load_pivot(
        os.path.join(dd, args.venduto),
        os.path.join(dd, args.era_diventa),
        os.path.join(dd, args.promo),
        os.path.join(dd, args.escludere),
    )
    print(f"Pivot: {pivot.shape[0]} serie × {pivot.shape[1]} mesi "
          f"({pivot.columns[0].date()} → {pivot.columns[-1].date()})", flush=True)

    cutoffs = [pd.Timestamp(c + "-01") for c in args.cutoffs.split(",")]
    model_groups = args.models.split(",")
    h = args.horizon

    dettagli = []
    for cutoff in cutoffs:
        train_cols = [c for c in pivot.columns if c <= cutoff]
        test_cols = [cutoff + pd.DateOffset(months=i) for i in range(1, h + 1)]
        missing = [c for c in test_cols if c not in pivot.columns]
        if missing:
            print(f"SKIP cutoff {cutoff.date()}: mesi test mancanti {missing}")
            continue

        train = pivot[train_cols]
        # serie esistenti al cutoff (almeno una vendita nello storico)
        train = train[(train != 0).any(axis=1)]
        actual = pivot.loc[train.index, test_cols]

        classi = classify(train)
        scale = mase_scale(train)
        print(f"\n=== Cutoff {cutoff.date()} | train {len(train_cols)} mesi | "
              f"{len(train)} serie | classi: "
              f"{classi['Classe'].value_counts().to_dict()}", flush=True)

        runs = {}
        if "base" in model_groups:
            for nome, fn in M.MODELLI_BASE.items():
                runs[nome] = fn(train, classi, h)
            # SBA v8 applicato a TUTTE le serie (per capire il valore del solo rate)
            flat = M.sba_v8_flat(classi, train.index)
            sba_all = pd.DataFrame({c: flat for c in test_cols})
            runs["SBA_v8_tutte"] = sba_all

        if "v8" in model_groups:
            print("  Fit NeuralProphet (config Iron v8)...", flush=True)
            runs["v8_NP+SBA"] = M.v8_composite(train, classi, h)

        if "sf" in model_groups:
            import models_sf
            runs.update(models_sf.run_statsforecast(train, h, which=args.sf_models.split(",") if args.sf_models else None))

        if "lgbm" in model_groups:
            import models_sf
            runs["LightGBM_globale"] = models_sf.run_lgbm(train, h)

        if "xgb" in model_groups:
            import models_sf
            runs["XGBoost_globale"] = models_sf.run_xgb(train, h)

        if "chronos" in model_groups:
            import models_chronos
            runs["Chronos_Bolt"] = models_chronos.run_chronos(train, h)

        # Un file per (modello, cutoff): run concorrenti o interrotti non possono
        # corrompere i risultati già salvati.
        parts_dir = os.path.join(out_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        for nome, fc in runs.items():
            fc = fc.reindex(columns=test_cols).fillna(0)
            res = evaluate(fc, actual, classi, scale, cutoff, nome)
            dettagli.append(res)
            part = os.path.join(parts_dir, f"{nome}__{cutoff.date()}.csv")
            res.to_csv(part + ".tmp", index=False)
            os.replace(part + ".tmp", part)
            print(f"  {nome}: ok", flush=True)

    # Il dettaglio complessivo viene sempre ricostruito dai parts
    import glob
    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "backtest_detail.csv"), index=False)

    riepilogo = summarize(df, by=("modello",))
    per_classe = summarize(df, by=("modello", "Classe"))
    per_cutoff = summarize(df, by=("modello", "cutoff"))

    with pd.ExcelWriter(os.path.join(out_dir, "backtest_summary.xlsx"), engine="xlsxwriter") as w:
        riepilogo.to_excel(w, sheet_name="Riepilogo", index=False)
        per_classe.to_excel(w, sheet_name="Per classe", index=False)
        per_cutoff.to_excel(w, sheet_name="Per cutoff", index=False)

    print("\n================ RIEPILOGO (pooled su tutti i cutoff) ================")
    print(riepilogo.to_string(index=False))
    print("\nDettaglio per classe e cutoff in:", os.path.join(out_dir, "backtest_summary.xlsx"))


if __name__ == "__main__":
    main()
