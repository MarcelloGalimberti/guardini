import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import ClassBadge from './ClassBadge'
import Plot from './Plot'

interface Props {
  runId: string
  codice: string
  onChiudi: () => void
}

export default function SeriesModal({ runId, codice, onChiudi }: Props) {
  const { data } = useQuery({
    queryKey: ['serie', runId, codice],
    queryFn: () => api.serie(runId, codice),
  })

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
      onClick={onChiudi}
    >
      <div
        className="card max-w-3xl w-full p-5 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 mb-1">
          <div>
            <p className="font-medium font-mono text-[15px]">{codice}</p>
            <p className="text-[15px] text-neutral-400">{data?.descrizione}</p>
          </div>
          <button className="btn text-sm" onClick={onChiudi}>Chiudi</button>
        </div>

        {data && (
          <>
            <div className="flex items-center gap-2 my-3 flex-wrap">
              <ClassBadge classe={data.classe} metodo={data.metodo} warning={data.warning} tecnico />
              {data.adi != null && (
                <span className="text-sm text-neutral-500">ADI {data.adi} · CV {data.cv}%</span>
              )}
            </div>

            {data.warning && (
              <p className="text-sm rounded-lg px-3 py-2 mb-3 bg-[#3A2E14] text-[#F5C95B]">
                Articolo nuovo (meno di 6 mesi di vendite): nessun modello è affidabile su
                questo codice. Verificare le quantità con il cliente prima dell'ordine.
              </p>
            )}

            <Plot
              data={[
                {
                  x: data.storico.map((p) => p.mese), y: data.storico.map((p) => p.qty),
                  name: 'Venduto', type: 'scatter', mode: 'lines+markers',
                  line: { color: '#54C79B', width: 2 }, marker: { size: 5 },
                },
                {
                  x: [
                    ...(data.storico.length ? [data.storico[data.storico.length - 1].mese] : []),
                    ...data.forecast.map((p) => p.mese),
                  ],
                  y: [
                    ...(data.storico.length ? [data.storico[data.storico.length - 1].qty] : []),
                    ...data.forecast.map((p) => p.qty),
                  ],
                  name: 'Forecast', type: 'scatter', mode: 'lines+markers',
                  line: { color: '#F0A422', width: 2, dash: 'dot' }, marker: { size: 5 },
                },
              ]}
              layout={{ height: 320, legend: { orientation: 'h', y: 1.15 },
                        yaxis: { title: { text: 'Pezzi' }, rangemode: 'tozero' } }}
            />
          </>
        )}
      </div>
    </div>
  )
}
