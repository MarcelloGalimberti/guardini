import { stileClasse } from '../theme'

interface Props {
  classe: string
  metodo?: string
  warning?: boolean
  tecnico?: boolean
}

export default function ClassBadge({ classe, metodo, warning, tecnico }: Props) {
  const s = stileClasse(classe)
  return (
    <span
      className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full whitespace-nowrap"
      style={{ backgroundColor: s.bg, color: s.text }}
      title={tecnico ? undefined : `Classe tecnica: ${classe}${metodo ? ` · Modello: ${metodo}` : ''}`}
    >
      {warning && <span aria-hidden>⚠</span>}
      {s.label}
      {tecnico && metodo && <span className="opacity-70">· {metodo}</span>}
    </span>
  )
}
