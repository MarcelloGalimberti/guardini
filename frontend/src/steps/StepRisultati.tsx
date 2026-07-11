import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { api, fmt } from '../api'
import ClassBadge from '../components/ClassBadge'
import Plot from '../components/Plot'
import SeriesModal from '../components/SeriesModal'
import type { RunMeta } from '../types'

interface Props {
  meta: RunMeta
  runPrecedenti: RunMeta[]
}

export default function StepRisultati({ meta, runPrecedenti }: Props) {
  const runId = meta.run_id
  const [ricerca, setRicerca] = useState('')
  const [soloWarning, setSoloWarning] = useState(false)
  const [codiceAperto, setCodiceAperto] = useState<string | null>(null)
  const [confrontoCon, setConfrontoCon] = useState<string>('')

  const { data: res } = useQuery({
    queryKey: ['results', runId],
    queryFn: () => api.results(runId),
  })
  const { data: conf } = useQuery({
    queryKey: ['confronto', runId, confrontoCon],
    queryFn: () => api.confronto(runId, confrontoCon),
    enabled: !!confrontoCon,
  })

  const righeFiltrate = useMemo(() => {
    if (!res) return []
    const q = ricerca.trim().toLowerCase()
    return res.righe
      .filter((r) => !soloWarning || r.warning)
      .filter((r) => !q || r.codice.toLowerCase().includes(q) || r.descrizione.toLowerCase().includes(q))
      .sort((a, b) => b.totale - a.totale)
  }, [res, ricerca, soloWarning])

  if (!res) return <p className="text-sm text-neutral-400">Caricamento risultati…</p>

  const kpi = res.kpi
  const altriRun = runPrecedenti.filter((r) => r.run_id !== runId && r.stato === 'completato')

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="card p-4">
          <p className="text-sm text-neutral-400">Volume previsto / mese</p>
          <p className="text-2xl font-medium mt-1 tabular-nums">
            {fmt.format(kpi.volume_mese ?? 0)} <span className="text-xs text-neutral-400">pz</span>
          </p>
        </div>
        <div className="card p-4">
          <p className="text-sm text-neutral-400">Codici con forecast</p>
          <p className="text-2xl font-medium mt-1 tabular-nums">{fmt.format(kpi.n_codici_forecast ?? 0)}</p>
        </div>
        <div className="card p-4">
          <p className="text-sm text-neutral-400">Volume su domanda regolare</p>
          <p className="text-2xl font-medium mt-1 tabular-nums">{kpi.pct_volume_regolare ?? 0}%</p>
        </div>
        <div className="p-4 rounded-xl" style={{ backgroundColor: '#3A2E14' }}>
          <p className="text-xs" style={{ color: '#F5C95B' }}>Nuovi articoli da verificare</p>
          <p className="text-2xl font-medium mt-1 tabular-nums" style={{ color: '#F5C95B' }}>
            ⚠ {kpi.n_new_da_verificare ?? 0}
          </p>
        </div>
      </div>

      {(kpi.n_new_da_verificare ?? 0) > 0 && (
        <div className="text-sm rounded-xl px-4 py-3"
             style={{ backgroundColor: '#3A2E14', color: '#F5C95B' }}>
          Su {kpi.n_new_da_verificare} articoli nuovi (prima vendita da meno di 6 mesi) nessun
          modello statistico è affidabile: il forecast emesso è prudenziale. Per i lanci
          importanti, verificare le quantità con il cliente —{' '}
          <button className="underline" onClick={() => setSoloWarning(true)}>mostra i codici</button>
          {' '}o{' '}
          <a className="underline" href={api.exportUrl(runId, 'new')}>scarica l'elenco</a>.
        </div>
      )}

      <div className="card p-4">
        <p className="text-sm font-medium">Volumi previsti per codice</p>
        <p className="text-sm text-neutral-400 mb-2">
          Top 60 codici per volume sull'orizzonte ({res.mesi.join(', ')}) — clicca un
          riquadro per il dettaglio.
        </p>
        <Plot
          data={[{
            type: 'treemap',
            labels: res.treemap.map((t) => t.codice),
            parents: res.treemap.map(() => ''),
            values: res.treemap.map((t) => t.totale),
            text: res.treemap.map((t) => `${t.label}${t.warning ? ' ⚠' : ''}`),
            textinfo: 'label+text+value',
            hovertext: res.treemap.map((t) => t.descrizione),
            hoverinfo: 'text+value',
            marker: {
              colors: res.treemap.map((t) =>
                t.warning ? '#FCF3D7'
                  : t.label === 'Regolare' ? '#E1F5EE'
                  : t.label === 'Irregolare' ? '#FAEEDA'
                  : t.label === 'Sporadica' ? '#EEEDFE'
                  : '#F1EFE8'),
              line: { width: 1, color: '#ffffff' },
            },
            tiling: { packing: 'squarify' },
          } as never]}
          layout={{ height: 380, margin: { t: 8, r: 8, b: 8, l: 8 } }}
          onClick={(e) => {
            const p = e.points?.[0] as { label?: string } | undefined
            if (p?.label) setCodiceAperto(p.label)
          }}
        />
      </div>

      <div className="card overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[#34353D] flex-wrap">
          <p className="text-[15px] font-medium mr-auto">Forecast per codice</p>
          <label className="flex items-center gap-1.5 text-xs text-neutral-300">
            <input
              type="checkbox" checked={soloWarning} className="accent-[#F0A422]"
              onChange={(e) => setSoloWarning(e.target.checked)}
            />
            Solo da verificare
          </label>
          <input
            className="input w-56" placeholder="Cerca codice o descrizione…"
            value={ricerca} onChange={(e) => setRicerca(e.target.value)}
          />
        </div>
        <div className="max-h-[480px] overflow-y-auto">
          <table className="w-full text-[15px]">
            <thead className="sticky top-0 bg-[#202127]">
              <tr className="text-left text-neutral-400 border-b border-[#34353D]">
                <th className="px-4 py-2 font-medium">Codice</th>
                <th className="px-4 py-2 font-medium">Domanda</th>
                {res.mesi.map((m) => (
                  <th key={m} className="px-4 py-2 font-medium text-right">{m.slice(5)}/{m.slice(2, 4)}</th>
                ))}
                <th className="px-4 py-2 font-medium text-right">Totale</th>
              </tr>
            </thead>
            <tbody>
              {righeFiltrate.slice(0, 300).map((r) => (
                <tr
                  key={r.codice}
                  className="border-b border-[#2C2D34] last:border-0 hover:bg-[#2A2B33] cursor-pointer"
                  onClick={() => setCodiceAperto(r.codice)}
                >
                  <td className="px-4 py-1.5">
                    <p className="font-mono text-xs">{r.codice}</p>
                    <p className="text-xs text-neutral-500 truncate max-w-[220px]">{r.descrizione}</p>
                  </td>
                  <td className="px-4 py-1.5">
                    <ClassBadge classe={r.classe} metodo={r.metodo} warning={r.warning} />
                  </td>
                  {res.mesi.map((m) => (
                    <td key={m} className="px-4 py-1.5 text-right tabular-nums">
                      {fmt.format(r[m] as number)}
                    </td>
                  ))}
                  <td className="px-4 py-1.5 text-right tabular-nums font-medium">{fmt.format(r.totale)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {righeFiltrate.length > 300 && (
            <p className="text-xs text-neutral-400 px-4 py-2">
              Mostrate le prime 300 righe di {fmt.format(righeFiltrate.length)} — usa la ricerca per restringere.
            </p>
          )}
        </div>
      </div>

      <div className="card p-4 flex items-center gap-2 flex-wrap">
        <a className="btn-primary" href={api.exportUrl(runId, 'galileo')}>
          ⬇ Esporta file J-Galileo
        </a>
        <a className="btn" href={api.exportUrl(runId, 'dettaglio')}>Dettaglio Excel</a>
        <a className="btn" href={api.exportUrl(runId, 'new')}>Elenco nuovi articoli</a>
        {altriRun.length > 0 && (
          <select
            className="input w-auto ml-auto"
            value={confrontoCon}
            onChange={(e) => setConfrontoCon(e.target.value)}
          >
            <option value="">Confronta con run precedente…</option>
            {altriRun.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {new Date(r.creato).toLocaleString('it-IT')}
              </option>
            ))}
          </select>
        )}
      </div>

      {conf && (
        <div className="card p-4 space-y-3">
          <p className="text-sm font-medium">
            Confronto con il run del {new Date(
              altriRun.find((r) => r.run_id === confrontoCon)?.creato ?? '',
            ).toLocaleString('it-IT')}
          </p>
          <p className="text-sm text-neutral-300">
            Volume sull'orizzonte: {fmt.format(conf.volume.attuale)} pz vs{' '}
            {fmt.format(conf.volume.precedente)} pz ({conf.volume.delta_pct > 0 ? '+' : ''}
            {conf.volume.delta_pct}%) · codici in comune {fmt.format(conf.codici.comuni)},
            nuovi {conf.codici.nuovi}, usciti {conf.codici.usciti}.
          </p>
          <table className="text-sm w-full max-w-md">
            <tbody>
              {conf.classi.map((c) => (
                <tr key={c.classe} className="border-b border-[#2C2D34] last:border-0">
                  <td className="py-1"><ClassBadge classe={c.classe} /></td>
                  <td className="py-1 text-right tabular-nums">{fmt.format(c.attuale)} pz</td>
                  <td className={'py-1 text-right tabular-nums w-20 ' +
                    (c.delta_pct > 10 ? 'text-red-700' : c.delta_pct < -10 ? 'text-blue-700' : 'text-neutral-400')}>
                    {c.delta_pct > 0 ? '+' : ''}{c.delta_pct}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {codiceAperto && (
        <SeriesModal runId={runId} codice={codiceAperto} onChiudi={() => setCodiceAperto(null)} />
      )}
    </div>
  )
}
