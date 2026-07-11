import { useState } from 'react'
import { api } from '../api'
import FileCard from '../components/FileCard'
import type { RunMeta, SintesiDati, TipoFile, Validazione } from '../types'

const FILE_DEFS: { tipo: TipoFile; titolo: string; descrizione: string; opzionale?: boolean }[] = [
  { tipo: 'venduto', titolo: 'Venduto J-Galileo', descrizione: 'Estrazione del fatturato per cliente, articolo e data (es. VENDUTO_LUG 23_GIU 26.xlsx)' },
  { tipo: 'era_diventa', titolo: 'Era-Diventa', descrizione: 'Mappatura sostituzione codici articolo (db_era_diventa.xlsx)' },
  { tipo: 'promo', titolo: 'Clienti promozioni', descrizione: 'Clienti promo esclusi dal forecast (db_clienti_promo.xlsx)' },
  { tipo: 'esclusioni', titolo: 'Articoli da escludere', descrizione: 'Codici articolo da non prevedere', opzionale: true },
]

interface Props {
  meta: RunMeta
  onValidato: (tipo: TipoFile, v: Validazione) => void
  onPreparato: (s: SintesiDati) => void
}

export default function StepDati({ meta, onValidato, onPreparato }: Props) {
  const validazioni = meta.validazioni
  const classiDisponibili = validazioni.venduto?.dettagli.classi_valutazione ?? []
  const [classiSel, setClassiSel] = useState<string[]>(['PF interno+confez.'])
  const [usaEraDiventa, setUsaEraDiventa] = useState(true)
  const [inCorso, setInCorso] = useState(false)
  const [errore, setErrore] = useState<string | null>(null)

  const pronti = ['venduto', 'era_diventa', 'promo'].every(
    (t) => validazioni[t as TipoFile]?.ok)

  const avanti = async () => {
    setInCorso(true)
    setErrore(null)
    try {
      onPreparato(await api.prepara(meta.run_id, classiSel, usaEraDiventa))
    } catch (e) {
      setErrore((e as Error).message)
    } finally {
      setInCorso(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-medium">Carica i file esportati da J-Galileo</h2>
        <p className="text-[15px] text-neutral-400 mt-1">
          Ogni file viene controllato subito: se il tracciato non è quello atteso lo vedi qui,
          non a metà elaborazione.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {FILE_DEFS.map((f) => (
          <FileCard
            key={f.tipo} runId={meta.run_id} tipo={f.tipo} titolo={f.titolo}
            descrizione={f.descrizione} opzionale={f.opzionale}
            validazione={validazioni[f.tipo]}
            onValidato={(v) => onValidato(f.tipo, v)}
          />
        ))}
      </div>

      {pronti && (
        <div className="card p-4 space-y-4">
          <div>
            <p className="text-[15px] font-medium mb-2">Classi di valutazione da includere</p>
            <div className="flex flex-wrap gap-2">
              {classiDisponibili.map((c) => {
                const sel = classiSel.includes(c)
                return (
                  <button
                    key={c}
                    onClick={() => setClassiSel(sel ? classiSel.filter((x) => x !== c) : [...classiSel, c])}
                    className={
                      'px-3 py-1 rounded-full text-xs border transition-colors ' +
                      (sel
                        ? 'bg-brand text-[#201503] border-brand'
                        : 'bg-[#26272E] text-neutral-300 border-[#4A4C57] hover:border-brand')
                    }
                  >
                    {c}
                  </button>
                )
              })}
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox" checked={usaEraDiventa}
              onChange={(e) => setUsaEraDiventa(e.target.checked)}
              className="accent-[#F0A422]"
            />
            Applica le sostituzioni Era-Diventa (consigliato) — in alternativa i codici
            vengono raggruppati per i primi 5 caratteri
          </label>
        </div>
      )}

      {errore && <p className="text-sm text-red-300 bg-red-950/50 rounded-lg px-3 py-2">{errore}</p>}

      <div className="flex justify-end">
        <button
          className="btn-primary"
          disabled={!pronti || classiSel.length === 0 || inCorso}
          onClick={avanti}
        >
          {inCorso ? 'Preparazione dati…' : 'Continua'}
        </button>
      </div>
    </div>
  )
}
