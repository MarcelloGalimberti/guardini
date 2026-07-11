/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Ambra dal logo Guardini; verde dal punto ADI come accento secondario
        brand: { DEFAULT: '#F0A422', dark: '#D18C0E', light: '#3A2E14' },
        adi: { green: '#7AB648', yellow: '#FFD520' },
        surface: { DEFAULT: '#202127', deep: '#17171B', line: '#34353D' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
