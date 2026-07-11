import { useState } from 'react'
import { api, fmt } from '../api'
import ClassBadge from '../components/ClassBadge'
import Plot from '../components/Plot'
import { stileClasse } from '../theme'
import type { ClassificaResp, RunMeta, SintesiDati } from '../types'

interface Props {
  meta: RunMeta
  sintesi?: SintesiDati
  classifica: ClassificaResp | null
  onClassificato: (c: ClassificaResp) => void
  onAvanti: () => void
}

const PARAMS_DEFAULT = { recent_months: 6, min_nonzero: 12, adi_limit: 1.4, cv_limit: 0.8 }

export default function StepClassi({ meta, sintesi, classifica, onClassificato, onAvanti }: Props) {
  const [avanzate, setAvanzate] = useState(false)
  const [params, setParams] = useState(PARAMS_DEFAULT)
  const [inCorso, setInCorso] = useState(false)
  const [errore, setErrore] = useState<string | null>(null)

  const esegui = async () => {
    setInCorso(true)
    setErrore(null)
    try {
      onClassificato(await api.classifica(meta.run_id, params))
    } catch (e) {
      setErrore((e as Error).message)
    } finally {
      setInCorso(false)
    }
  }

  const num = (k: keyof typeof params, label: string, step = 1) => (
    <label className="flex items-center justify-between gap-3 text-sm">
      <span className="text-neutral-300">{label}</span>
      <input
        type="number" className="input" step={step} value={params[k]}
        onChange={(e) => setParams({ ...params, [k]: Number(e.target.value) })}
      />
    </label>
  )

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-medium">Classificazione della domanda</h2>
          <p className="text-[15px] text-neutral-400 mt-1">
            Ogni codice viene classificato per regolarità della domanda: a ogni classe il
            backtest ha assegnato il modello di forecasting più accurato.
            {sintesi && <> Dataset: {fmt.format(sintesi.n_codici)} codici · {sintesi.periodo}.</>}
          </p>
        </div>
        <button className="btn text-xs" onClick={() => setAvanzate(!avanzate)}>
          Impostazioni avanzate {avanzate ? '▴' : '▾'}
        </button>
      </div>

      {avanzate && (
        <div className="card p-4 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
          {num('adi_limit', 'Soglia ADI (Syntetos-Boylan, standard 1,32)', 0.01)}
          {num('cv_limit', 'Soglia CV', 0.01)}
          {num('min_nonzero', 'Mesi minimi con vendite')}
          {num('recent_months', 'Mesi per "attivo di recente"')}
          <p className="text-xs text-neutral-400 md:col-span-2">
            I default sono quelli validati dal backtest. Modificarli cambia l'instradamento
            dei codici verso i modelli.
          </p>
        </div>
      )}

      {!classifica && (
        <button className="btn-primary" onClick={esegui} disabled={inCorso}>
          {inCorso ? 'Classificazione…' : 'Classifica i codici'}
        </button>
      )}
      {errore && <p className="text-sm text-red-300 bg-red-950/50 rounded-lg px-3 py-2">{errore}</p>}

      {classifica && (
        <>
          <div className="card overflow-hidden">
            <table className="w-full text-[15px]">
              <thead>
                <tr className="text-left text-neutral-400 border-b border-[#34353D]">
                  <th className="px-4 py-2.5 font-medium">Tipo di domanda</th>
                  <th className="px-4 py-2.5 font-medium text-right">Codici</th>
                  <th className="px-4 py-2.5 font-medium text-right">Volume storico</th>
                  <th className="px-4 py-2.5 font-medium text-right">% volume</th>
                  <th className="px-4 py-2.5 font-medium">Modello</th>
                  <th className="px-4 py-2.5 font-medium text-right" title="WAPE cumulato a 3 mesi dal backtest out-of-sample">
                    Errore atteso
                  </th>
                </tr>
              </thead>
              <tbody>
                {classifica.riepilogo.map((r) => (
                  <tr key={r.classe} className="border-b border-[#2C2D34] last:border-0">
                    <td className="px-4 py-2">
                      <ClassBadge classe={r.classe} warning={r.warning} />
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">{fmt.format(r.n_codici)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{fmt.format(r.volume)} pz</td>
                    <td className="px-4 py-2 text-right tabular-nums">{r.pct_volume}%</td>
                    <td className="px-4 py-2 text-neutral-400 text-xs">{r.modello}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-neutral-400">
                      {r.wape_backtest != null ? `${r.wape_backtest}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card p-4">
            <p className="text-[15px] font-medium mb-1">Mappa della domanda (ADI / CV)</p>
            <p className="text-sm text-neutral-400 mb-2">
              Orizzontale: ogni quanto compra il mercato. Verticale: quanto variano le quantità.
              I codici in basso a sinistra sono i più prevedibili.
            </p>
            <Plot
              data={Object.values(
                classifica.matrice.reduce<Record<string, { x: number[]; y: number[]; text: string[]; label: string; colore: string }>>(
                  (acc, p) => {
                    const s = stileClasse(p.classe)
                    acc[s.label] = acc[s.label] || { x: [], y: [], text: [], label: s.label, colore: s.mid }
                    acc[s.label].x.push(p.adi)
                    acc[s.label].y.push(p.cv)
                    acc[s.label].text.push(p.codice)
                    return acc
                  }, {}),
              ).map((g) => ({
                x: g.x, y: g.y, text: g.text, name: g.label, type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { size: 6, opacity: 0.75, color: g.colore },
              }))}
              layout={{
                height: 420,
                xaxis: { title: { text: 'Intervallo medio tra acquisti (ADI)' } },
                yaxis: { title: { text: 'Variabilità delle quantità (CV %)' } },
                shapes: [
                  { type: 'line', x0: classifica.soglie.adi, x1: classifica.soglie.adi, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dot', color: '#999', width: 1 } },
                  { type: 'line', y0: classifica.soglie.cv, y1: classifica.soglie.cv, x0: 0, x1: 1, xref: 'paper', line: { dash: 'dot', color: '#999', width: 1 } },
                ],
                legend: { orientation: 'h', y: 1.12 },
              }}
            />
          </div>

          <div className="flex justify-end gap-2">
            <button className="btn" onClick={esegui} disabled={inCorso}>
              {inCorso ? 'Ricalcolo…' : 'Ricalcola'}
            </button>
            <button className="btn-primary" onClick={onAvanti}>Continua</button>
          </div>
        </>
      )}
    </div>
  )
}
