import Plotly from 'plotly.js-dist-min'
import type { Data, Layout, PlotMouseEvent, PlotlyHTMLElement } from 'plotly.js-dist-min'
import { useEffect, useRef } from 'react'

interface Props {
  data: Data[]
  layout?: Partial<Layout>
  onClick?: (punto: PlotMouseEvent) => void
  style?: React.CSSProperties
}

const GRIGLIA = 'rgba(255,255,255,0.08)'
const ASSE = { gridcolor: GRIGLIA, zerolinecolor: GRIGLIA, linecolor: GRIGLIA }

const BASE_LAYOUT: Partial<Layout> = {
  font: { family: 'Inter, system-ui, sans-serif', size: 13.5, color: '#C9CAD1' },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { t: 24, r: 16, b: 40, l: 56 },
  separators: ',.',
}

export default function Plot({ data, layout, onClick, style }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const el = ref.current
    const finale: Partial<Layout> = {
      ...BASE_LAYOUT,
      ...layout,
      xaxis: { ...ASSE, ...(layout?.xaxis as object | undefined) },
      yaxis: { ...ASSE, ...(layout?.yaxis as object | undefined) },
    }
    Plotly.react(el, data, finale, {
      displayModeBar: false, responsive: true,
    }).then(() => {
      if (onClick) (el as unknown as PlotlyHTMLElement).on('plotly_click', onClick)
    })
    return () => { Plotly.purge(el) }
  }, [data, layout])

  return <div ref={ref} style={style} />
}
