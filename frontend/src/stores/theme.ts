import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useThemeStore = defineStore('theme', {
  state: () => ({
    darkMode: ref(false)
  }),
  actions: {
    toggleDarkMode() {
      this.darkMode = !this.darkMode
      localStorage.setItem('darkMode', JSON.stringify(this.darkMode))
      if (this.darkMode) {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
    },
    loadTheme() {
      const savedTheme = localStorage.getItem('darkMode')
      if (savedTheme) {
        this.darkMode = JSON.parse(savedTheme)
        if (this.darkMode) {
          document.documentElement.classList.add('dark')
        }
      } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
        this.darkMode = prefersDark
        if (prefersDark) {
          document.documentElement.classList.add('dark')
        }
      }
    }
  }
}) 