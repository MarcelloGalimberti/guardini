import type {
  ClassificaResp, Confronto, Results, RunMeta, SerieDettaglio, SintesiDati,
  TipoFile, Validazione,
} from './types'

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `Errore ${res.status}`
    try {
      const body = await res.json()
      if (body.detail) msg = body.detail
    } catch { /* corpo non json */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

export const api = {
  creaRun: () =>
    fetch('/api/runs', { method: 'POST' }).then(j<{ run_id: string }>),

  elencoRun: () => fetch('/api/runs').then(j<RunMeta[]>),

  statoRun: (id: string) => fetch(`/api/runs/${id}`).then(j<RunMeta>),

  caricaFile: (id: string, tipo: TipoFile, file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(`/api/runs/${id}/files/${tipo}`, { method: 'POST', body: fd })
      .then(j<Validazione>)
  },

  prepara: (id: string, classi: string[], usaEraDiventa: boolean) =>
    fetch(`/api/runs/${id}/prepare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classi_valutazione: classi, usa_era_diventa: usaEraDiventa }),
    }).then(j<SintesiDati>),

  classifica: (id: string, params: Record<string, number>) =>
    fetch(`/api/runs/${id}/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }).then(j<ClassificaResp>),

  avviaForecast: (id: string, orizzonte: number, capAbilitato: boolean, capFactor: number) =>
    fetch(`/api/runs/${id}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ orizzonte, cap_abilitato: capAbilitato, cap_factor: capFactor }),
    }).then(j<{ stato: string }>),

  results: (id: string) => fetch(`/api/runs/${id}/results`).then(j<Results>),

  serie: (id: string, codice: string) =>
    fetch(`/api/runs/${id}/series/${encodeURIComponent(codice)}`).then(j<SerieDettaglio>),

  confronto: (id: string, altro: string) =>
    fetch(`/api/runs/${id}/confronto/${altro}`).then(j<Confronto>),

  exportUrl: (id: string, cosa: 'galileo' | 'dettaglio' | 'new') =>
    `/api/runs/${id}/export/${cosa}`,
}

export const fmt = new Intl.NumberFormat('it-IT')
