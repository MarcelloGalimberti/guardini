import { useRef, useState } from 'react'
import { api } from '../api'
import type { TipoFile, Validazione } from '../types'

interface Props {
  runId: string
  tipo: TipoFile
  titolo: string
  descrizione: string
  opzionale?: boolean
  validazione?: Validazione
  onValidato: (v: Validazione) => void
}

export default function FileCard({ runId, tipo, titolo, descrizione, opzionale, validazione, onValidato }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [caricamento, setCaricamento] = useState(false)
  const [trascina, setTrascina] = useState(false)
  const [errore, setErrore] = useState<string | null>(null)

  const carica = async (file: File) => {
    setCaricamento(true)
    setErrore(null)
    try {
      const v = await api.caricaFile(runId, tipo, file)
      onValidato(v)
    } catch (e) {
      setErrore((e as Error).message)
    } finally {
      setCaricamento(false)
    }
  }

  const ok = validazione?.ok
  const ko = validazione && !validazione.ok

  return (
    <div
      className={
        'card p-4 transition-colors cursor-pointer hover:border-[#5A5C68] ' +
        (trascina ? 'border-brand bg-brand-light ' : '') +
        (ok ? 'border-[#3C7A5E] ' : ko ? 'border-red-700 ' : '')
      }
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setTrascina(true) }}
      onDragLeave={() => setTrascina(false)}
      onDrop={(e) => {
        e.preventDefault()
        setTrascina(false)
        if (e.dataTransfer.files[0]) carica(e.dataTransfer.files[0])
      }}
    >
      <input
        ref={inputRef} type="file" accept=".xlsx,.xls" className="hidden"
        onClick={(e) => { (e.target as HTMLInputElement).value = '' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) carica(f)
          e.target.value = ''
        }}
      />
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[15px] font-medium">
            {titolo}
            {opzionale && <span className="ml-2 text-sm text-neutral-500 font-normal">opzionale</span>}
          </p>
          <p className="text-sm text-neutral-400 mt-0.5">{descrizione}</p>
        </div>
        <span className="text-lg leading-none">
          {caricamento ? '…' : ok ? '✅' : ko ? '❌' : '📄'}
        </span>
      </div>

      {ok && validazione && (
        <p className="text-sm mt-2 rounded-lg px-2.5 py-1.5 bg-[#1E3A2E] text-[#7ADFB2]">
          {validazione.nome_file} — {validazione.righe.toLocaleString('it-IT')} righe
          {validazione.dettagli.periodo && <> · {validazione.dettagli.periodo}</>}
          {validazione.dettagli.codici_univoci && (
            <> · {validazione.dettagli.codici_univoci.toLocaleString('it-IT')} codici</>
          )}
        </p>
      )}
      {ko && validazione && (
        <p className="text-sm text-red-300 mt-2 bg-red-950/50 rounded-lg px-2.5 py-1.5">
          {validazione.messaggi[0]}
        </p>
      )}
      {errore && <p className="text-sm text-red-300 mt-2">{errore}</p>}
      {!validazione && !caricamento && (
        <p className="text-sm text-neutral-500 mt-2">Trascina qui il file o clicca per selezionarlo</p>
      )}
    </div>
  )
}
