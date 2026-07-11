import { useEffect, useState } from 'react'
import { api } from '../api'
import type { RunMeta } from '../types'

interface Props {
  meta: RunMeta
  onAggiorna: (m: RunMeta) => void
  onCompletato: () => void
}

export default function StepForecast({ meta, onAggiorna, onCompletato }: Props) {
  const [orizzonte, setOrizzonte] = useState(4)
  const [errore, setErrore] = useState<string | null>(null)

  const inCorso = meta.stato === 'forecast_in_corso'

  useEffect(() => {
    if (!inCorso) return
    const timer = setInterval(async () => {
      const m = await api.statoRun(meta.run_id)
      onAggiorna(m)
      if (m.stato === 'completato') {
        clearInterval(timer)
        onCompletato()
      }
      if (m.stato === 'errore') clearInterval(timer)
    }, 1500)
    return () => clearInterval(timer)
  }, [inCorso, meta.run_id])

  const avvia = async () => {
    setErrore(null)
    try {
      // Guardrail anti-picco: gestito internamente (tetto 1,5× max storico).
      // Con il motore segmentato non scatta mai sui dati correnti (verificato:
      // 0 valori limitati) — resta come assicurazione silenziosa, senza UI.
      await api.avviaForecast(meta.run_id, orizzonte, true, 1.5)
      onAggiorna({ ...meta, stato: 'forecast_in_corso', progress: 'Avvio…' })
    } catch (e) {
      setErrore((e as Error).message)
    }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h2 className="text-xl font-medium">Calcolo del forecast</h2>
        <p className="text-[15px] text-neutral-400 mt-1">
          Il primo mese dell'orizzonte viene scartato nel file per J-Galileo (mese corrente
          già avviato): con orizzonte 4 il gestionale riceve 3 mesi di previsione.
        </p>
      </div>

      <div className="card p-5">
        <label className="flex items-center justify-between gap-3 text-[15px]">
          <span className="text-neutral-300">Orizzonte previsionale (mesi)</span>
          <input
            type="number" className="input" min={2} max={12} value={orizzonte}
            onChange={(e) => setOrizzonte(Number(e.target.value))} disabled={inCorso}
          />
        </label>
      </div>

      {inCorso && (
        <div className="card p-5 flex items-center gap-3">
          <span className="w-4 h-4 rounded-full border-2 border-brand border-t-transparent animate-spin" />
          <div>
            <p className="text-[15px] font-medium">Fit dei modelli in corso…</p>
            <p className="text-sm text-neutral-400">{meta.progress ?? 'Avvio…'}</p>
          </div>
        </div>
      )}

      {meta.stato === 'errore' && (
        <div className="text-[15px] text-red-300 bg-red-950/50 border border-red-900 rounded-lg px-4 py-3">
          <p className="font-medium">Il calcolo si è interrotto.</p>
          <pre className="text-sm whitespace-pre-wrap mt-1">{meta.errore}</pre>
        </div>
      )}
      {errore && (
        <p className="text-[15px] text-red-300 bg-red-950/50 border border-red-900 rounded-lg px-4 py-3">
          {errore}
        </p>
      )}

      <div className="flex justify-end">
        <button className="btn-primary" onClick={avvia} disabled={inCorso}>
          {inCorso ? 'Calcolo in corso…' : meta.stato === 'completato' ? 'Ricalcola forecast' : 'Avvia il forecast'}
        </button>
      </div>
    </div>
  )
}
