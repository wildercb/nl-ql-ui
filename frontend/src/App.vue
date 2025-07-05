<template>
  <div id="app" class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <Navbar @showAuth="showAuthModal = true" />
    <RouterView />
    <AuthModal :show="showAuthModal" @close="showAuthModal = false" @authenticated="onAuthenticated" @guest="onGuestSession" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useThemeStore } from '@/stores/theme'
import { useAuthStore } from '@/stores/auth'
import Navbar from '@/components/Navbar.vue'
import AuthModal from '@/components/AuthModal.vue'

const themeStore = useThemeStore()
const authStore = useAuthStore()

const showAuthModal = ref(false)

const onAuthenticated = (authData: any) => { 
  authStore.setAuthenticated(authData); 
  showAuthModal.value = false; 
}

const onGuestSession = (sessionData: any) => { 
  authStore.setGuestSession(sessionData); 
  showAuthModal.value = false; 
}

onMounted(() => {
  // Initialize theme
  themeStore.loadTheme()
  // Load persisted auth session
  authStore.loadPersistedSession()
})
</script>

<style>
/* Global styles are imported in main.ts */
</style> 