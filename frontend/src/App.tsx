import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from './api'
import Stepper from './components/Stepper'
import StepClassi from './steps/StepClassi'
import StepDati from './steps/StepDati'
import StepForecast from './steps/StepForecast'
import StepRisultati from './steps/StepRisultati'
import type { ClassificaResp, RunMeta } from './types'

const STEP_DA_STATO: Record<RunMeta['stato'], number> = {
  creato: 0, dati_ok: 1, classificato: 2, forecast_in_corso: 2, completato: 3, errore: 2,
}

export default function App() {
  const qc = useQueryClient()

  // Impedisce al browser di aprire/scaricare i file trascinati fuori dalle
  // dropzone (drop mancato di pochi pixel = nessun feedback, sembra un bug)
  useEffect(() => {
    const blocca = (e: DragEvent) => e.preventDefault()
    window.addEventListener('dragover', blocca)
    window.addEventListener('drop', blocca)
    return () => {
      window.removeEventListener('dragover', blocca)
      window.removeEventListener('drop', blocca)
    }
  }, [])
  const [meta, setMeta] = useState<RunMeta | null>(null)
  const [step, setStep] = useState(0)
  const [classifica, setClassifica] = useState<ClassificaResp | null>(null)
  const [storicoAperto, setStoricoAperto] = useState(false)

  const { data: elenco } = useQuery({
    queryKey: ['runs'],
    queryFn: api.elencoRun,
    refetchInterval: 30000,
  })

  const massimo = meta ? STEP_DA_STATO[meta.stato] : 0

  const nuovoRun = async () => {
    const { run_id } = await api.creaRun()
    setMeta(await api.statoRun(run_id))
    setClassifica(null)
    setStep(0)
    qc.invalidateQueries({ queryKey: ['runs'] })
  }

  const apriRun = async (id: string) => {
    const m = await api.statoRun(id)
    setMeta(m)
    setClassifica(null)
    setStep(STEP_DA_STATO[m.stato])
    setStoricoAperto(false)
  }

  return (
    <div className="min-h-screen">
      <header className="bg-surface border-b border-surface-line sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 py-2.5 flex items-center gap-4">
          <div className="flex items-center gap-3 mr-auto">
            <img src="/guardini.png" alt="Guardini" className="h-9 rounded" />
            <div>
              <p className="text-[15px] font-medium leading-tight">Forecasting domanda</p>
              {meta && (
                <p className="text-xs text-neutral-400 leading-tight">
                  Run del {new Date(meta.creato).toLocaleString('it-IT')}
                </p>
              )}
            </div>
          </div>
          {meta && <Stepper corrente={step} massimoRaggiunto={massimo} onVai={setStep} />}
          <button className="btn text-sm" onClick={() => setStoricoAperto(!storicoAperto)}>
            Run precedenti
          </button>
          <img src="/logo_adi.png" alt="ADI Business Consulting" className="h-9 rounded" />
        </div>
        {storicoAperto && (
          <div className="max-w-6xl mx-auto px-4 pb-3">
            <div className="card p-2 max-h-64 overflow-y-auto">
              {(elenco ?? []).length === 0 && (
                <p className="text-sm text-neutral-400 px-2 py-1">Nessun run salvato.</p>
              )}
              {(elenco ?? []).map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => apriRun(r.run_id)}
                  className="w-full flex items-center gap-3 px-2 py-1.5 rounded-lg hover:bg-[#2A2B33] text-left text-[15px]"
                >
                  <span className="text-neutral-300">{new Date(r.creato).toLocaleString('it-IT')}</span>
                  <span className={
                    'text-xs px-2 py-0.5 rounded-full ' +
                    (r.stato === 'completato'
                      ? 'bg-[#1E3A2E] text-[#7ADFB2]'
                      : 'bg-[#2A2B33] text-neutral-400')
                  }>
                    {r.stato.replace('_', ' ')}
                  </span>
                  {r.kpi?.volume_mese != null && (
                    <span className="text-xs text-neutral-400 ml-auto tabular-nums">
                      {r.kpi.volume_mese.toLocaleString('it-IT')} pz/mese
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        {!meta ? (
          <div className="card p-10 max-w-xl mx-auto mt-16 text-center space-y-4">
            <img src="/guardini.png" alt="Guardini" className="h-12 mx-auto rounded" />
            <h1 className="text-2xl font-medium">Forecasting domanda materiali</h1>
            <p className="text-[15px] text-neutral-400">
              Carica le estrazioni J-Galileo, classifica la domanda e genera il file di
              forecast da importare nel gestionale. Ogni run resta salvato e confrontabile.
            </p>
            <div className="flex justify-center gap-2">
              <button className="btn-primary" onClick={nuovoRun}>Nuovo run</button>
              {(elenco ?? []).length > 0 && (
                <button className="btn" onClick={() => setStoricoAperto(true)}>
                  Riapri un run
                </button>
              )}
            </div>
          </div>
        ) : (
          <>
            {step === 0 && (
              <StepDati
                meta={meta}
                onValidato={(tipo, v) =>
                  setMeta({ ...meta, validazioni: { ...meta.validazioni, [tipo]: v } })}
                onPreparato={(s) => {
                  setMeta({ ...meta, stato: 'dati_ok', sintesi_dati: s })
                  setStep(1)
                }}
              />
            )}
            {step === 1 && (
              <StepClassi
                meta={meta}
                sintesi={meta.sintesi_dati}
                classifica={classifica}
                onClassificato={(c) => {
                  setClassifica(c)
                  setMeta({ ...meta, stato: meta.stato === 'completato' ? 'completato' : 'classificato' })
                }}
                onAvanti={() => setStep(2)}
              />
            )}
            {step === 2 && (
              <StepForecast
                meta={meta}
                onAggiorna={setMeta}
                onCompletato={() => {
                  setStep(3)
                  qc.invalidateQueries({ queryKey: ['results', meta.run_id] })
                  qc.invalidateQueries({ queryKey: ['runs'] })
                }}
              />
            )}
            {step === 3 && <StepRisultati meta={meta} runPrecedenti={elenco ?? []} />}
          </>
        )}
      </main>
    </div>
  )
}
