export type TipoFile = 'venduto' | 'era_diventa' | 'promo' | 'esclusioni'

export interface Validazione {
  tipo: TipoFile
  nome_file: string
  ok: boolean
  righe: number
  messaggi: string[]
  dettagli: {
    codici_univoci?: number
    periodo?: string
    classi_valutazione?: string[]
  }
}

export interface RunMeta {
  run_id: string
  creato: string
  stato: 'creato' | 'dati_ok' | 'classificato' | 'forecast_in_corso' | 'completato' | 'errore'
  progress: string | null
  validazioni: Partial<Record<TipoFile, Validazione>>
  parametri: Record<string, unknown>
  kpi: Kpi
  sintesi_dati?: SintesiDati
  errore?: string
}

export interface SintesiDati {
  n_codici: number
  n_mesi: number
  periodo: string
  volume_totale: number
}

export interface Kpi {
  volume_mese?: number
  n_codici_forecast?: number
  pct_volume_regolare?: number
  n_new_da_verificare?: number
  mesi_output?: string[]
}

export interface RiepilogoClasse {
  classe: string
  label: string
  modello: string
  n_codici: number
  volume: number
  pct_volume: number
  wape_backtest: number | null
  warning: boolean
}

export interface PuntoMatrice {
  codice: string
  adi: number
  cv: number
  classe: string
  label: string
}

export interface ClassificaResp {
  riepilogo: RiepilogoClasse[]
  matrice: PuntoMatrice[]
  soglie: { adi: number; cv: number }
}

export interface RigaForecast {
  codice: string
  descrizione: string
  classe: string
  label: string
  metodo: string
  warning: boolean
  storico_totale: number
  totale: number
  [mese: string]: string | number | boolean
}

export interface Results {
  kpi: Kpi
  mesi: string[]
  righe: RigaForecast[]
  treemap: { codice: string; descrizione: string; totale: number; label: string; warning: boolean }[]
}

export interface SerieDettaglio {
  codice: string
  descrizione: string
  classe: string
  label: string
  metodo: string
  warning: boolean
  adi: number | null
  cv: number | null
  storico: { mese: string; qty: number }[]
  forecast: { mese: string; qty: number }[]
}

export interface Confronto {
  volume: { attuale: number; precedente: number; delta_pct: number }
  codici: { comuni: number; nuovi: number; usciti: number }
  classi: { classe: string; label: string; attuale: number; precedente: number; delta_pct: number }[]
  top_variazioni: { codice: string; attuale: number; precedente: number; delta: number }[]
}
