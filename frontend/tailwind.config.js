/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'banana-unripe': '#4ade80', // Green
        'banana-ripe': '#facc15',   // Yellow
        'banana-overripe': '#a16207', // Brown
        'banana-dispose': '#ef4444', // Red
      }
    },
  },
  plugins: [],
}
