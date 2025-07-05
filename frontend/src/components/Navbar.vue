<template>
  <nav class="bg-gray-800 border-b border-gray-700">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo and Navigation Links -->
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <h1 class="text-xl font-bold text-white">MPPW-MCP</h1>
          </div>
          <div class="hidden md:block ml-10">
            <div class="flex items-baseline space-x-4">
              <router-link 
                to="/" 
                class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                :class="{ 'bg-gray-700 text-white': $route.path === '/' }"
              >
                <i class="fas fa-home mr-2"></i>Home
              </router-link>
              <router-link 
                to="/history" 
                class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                :class="{ 'bg-gray-700 text-white': $route.path === '/history' }"
              >
                <i class="fas fa-history mr-2"></i>History
              </router-link>
              <router-link 
                to="/docs" 
                class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                :class="{ 'bg-gray-700 text-white': $route.path === '/docs' }"
              >
                <i class="fas fa-book mr-2"></i>Docs
              </router-link>
            </div>
          </div>
        </div>

        <!-- Mobile menu button -->
        <div class="md:hidden">
          <button 
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium"
          >
            <i class="fas fa-bars"></i>
          </button>
        </div>

        <!-- Authentication Section -->
        <div class="hidden md:block">
          <div v-if="authStore.isAuthenticated" class="flex items-center space-x-4">
            <span class="text-sm text-gray-300">
              Welcome, {{ authStore.isGuest ? 'Guest' : authStore.currentUser?.email }}
            </span>
            <button 
              @click="handleLogout" 
              class="text-red-400 hover:text-red-300 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
            >
              <i class="fas fa-sign-out-alt mr-2"></i>Logout
            </button>
          </div>
          <button 
            v-else 
            @click="handleSignIn" 
            class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200"
          >
            Sign In
          </button>
        </div>
      </div>

      <!-- Mobile menu -->
      <div v-if="mobileMenuOpen" class="md:hidden">
        <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3 border-t border-gray-700">
          <router-link 
            to="/" 
            class="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium"
            :class="{ 'bg-gray-700 text-white': $route.path === '/' }"
            @click="mobileMenuOpen = false"
          >
            <i class="fas fa-home mr-2"></i>Home
          </router-link>
          <router-link 
            to="/history" 
            class="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium"
            :class="{ 'bg-gray-700 text-white': $route.path === '/history' }"
            @click="mobileMenuOpen = false"
          >
            <i class="fas fa-history mr-2"></i>History
          </router-link>
          <router-link 
            to="/docs" 
            class="text-gray-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium"
            :class="{ 'bg-gray-700 text-white': $route.path === '/docs' }"
            @click="mobileMenuOpen = false"
          >
            <i class="fas fa-book mr-2"></i>Docs
          </router-link>
          
          <!-- Mobile auth section -->
          <div class="pt-4 border-t border-gray-700">
            <div v-if="authStore.isAuthenticated" class="flex items-center justify-between px-3 py-2">
              <span class="text-sm text-gray-300">
                Welcome, {{ authStore.isGuest ? 'Guest' : authStore.currentUser?.email }}
              </span>
              <button 
                @click="handleLogout" 
                class="text-red-400 hover:text-red-300 px-3 py-2 rounded-md text-sm font-medium"
              >
                <i class="fas fa-sign-out-alt mr-2"></i>Logout
              </button>
            </div>
            <button 
              v-else 
              @click="handleSignIn" 
              class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200"
            >
              Sign In
            </button>
          </div>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';

const authStore = useAuthStore();
const router = useRouter();
const mobileMenuOpen = ref(false);

const handleSignIn = () => {
  // Emit event to parent to show auth modal
  emit('showAuth');
};

const handleLogout = () => {
  authStore.logout();
  router.push('/');
  mobileMenuOpen.value = false;
};

const emit = defineEmits<{
  showAuth: [];
}>();
</script> 