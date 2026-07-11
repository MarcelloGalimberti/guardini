// Scala semantica unica per le classi di domanda (linguaggio a due livelli):
// usata OVUNQUE — badge, matrice, treemap, KPI — per coerenza visiva.
export interface StileClasse {
  label: string
  bg: string
  text: string
  mid: string
}

export const CLASSI: Record<string, StileClasse> = {
  Smooth: { label: 'Regolare', bg: '#E1F5EE', text: '#085041', mid: '#1D9E75' },
  Erratic: { label: 'Irregolare', bg: '#FAEEDA', text: '#633806', mid: '#EF9F27' },
  Intermittent: { label: 'Sporadica', bg: '#EEEDFE', text: '#3C3489', mid: '#7F77DD' },
  Lumpy: { label: 'Sporadica', bg: '#EEEDFE', text: '#3C3489', mid: '#534AB7' },
  'Insufficient Data': { label: 'Storico limitato', bg: '#F1EFE8', text: '#444441', mid: '#B4B2A9' },
  New: { label: 'Nuovo articolo', bg: '#FCF3D7', text: '#7A5A00', mid: '#D9A400' },
}

export const stileClasse = (classe: string): StileClasse =>
  CLASSI[classe] ?? { label: classe, bg: '#F1EFE8', text: '#444441', mid: '#B4B2A9' }
