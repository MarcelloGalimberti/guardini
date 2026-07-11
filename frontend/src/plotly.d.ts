declare module 'plotly.js-dist-min' {
  export interface PlotMouseEvent {
    points: Array<{ label?: string; x?: unknown; y?: unknown; text?: string }>
  }
  export interface PlotlyHTMLElement extends HTMLElement {
    on(evento: 'plotly_click', cb: (e: PlotMouseEvent) => void): void
  }
  export type Data = Record<string, unknown>
  export type Layout = Record<string, unknown>
  export function react(
    el: HTMLElement, data: Data[], layout?: Partial<Layout>,
    config?: Record<string, unknown>,
  ): Promise<void>
  export function purge(el: HTMLElement): void
  const Plotly: {
    react: typeof react
    purge: typeof purge
  }
  export default Plotly
}
