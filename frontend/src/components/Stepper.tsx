const STEPS = ['Dati', 'Classificazione', 'Forecast', 'Risultati']

interface Props {
  corrente: number          // 0-based
  massimoRaggiunto: number
  onVai: (step: number) => void
}

export default function Stepper({ corrente, massimoRaggiunto, onVai }: Props) {
  return (
    <nav className="flex items-center gap-1 text-[15px]" aria-label="Avanzamento">
      {STEPS.map((nome, i) => {
        const attivo = i === corrente
        const fatto = i < massimoRaggiunto
        const raggiungibile = i <= massimoRaggiunto
        return (
          <button
            key={nome}
            onClick={() => raggiungibile && onVai(i)}
            disabled={!raggiungibile}
            className={
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors ' +
              (attivo
                ? 'text-brand font-medium bg-brand-light'
                : fatto
                  ? 'text-neutral-400 hover:text-brand hover:bg-[#2A2B33]'
                  : 'text-neutral-600 cursor-default')
            }
          >
            <span
              className={
                'w-5 h-5 rounded-full flex items-center justify-center text-xs font-medium ' +
                (attivo
                  ? 'bg-brand text-[#201503]'
                  : fatto
                    ? 'bg-[#4A4C57] text-neutral-200'
                    : 'bg-[#2A2B33] text-neutral-500')
              }
            >
              {fatto ? '✓' : i + 1}
            </span>
            {nome}
          </button>
        )
      })}
    </nav>
  )
}
