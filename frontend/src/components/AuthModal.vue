<template>
  <div 
    v-if="show" 
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm"
    @click="closeModal"
  >
    <div 
      class="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 transform transition-all duration-300 scale-100"
      @click.stop
    >
      <!-- Header -->
      <div class="bg-gradient-to-r from-green-600 to-green-700 text-white p-6 rounded-t-2xl">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <i class="fas fa-user text-white text-sm"></i>
            </div>
            <h2 class="text-xl font-bold">{{ isLogin ? 'Welcome Back' : 'Join Us' }}</h2>
          </div>
          <button 
            @click="closeModal"
            class="text-white hover:text-green-200 transition-colors duration-200"
          >
            <i class="fas fa-times text-xl"></i>
          </button>
        </div>
        <p class="text-green-100 mt-2 text-sm">
          {{ isLogin ? 'Sign in to access your query history' : 'Create an account to save your queries' }}
        </p>
      </div>

      <!-- Tabs -->
      <div class="flex bg-gray-50 border-b">
        <button
          @click="switchToLogin"
          :class="[
            'flex-1 py-3 px-4 text-sm font-medium transition-all duration-200',
            isLogin 
              ? 'bg-white text-green-600 border-b-2 border-green-600' 
              : 'text-gray-600 hover:text-green-600'
          ]"
        >
          <i class="fas fa-sign-in-alt mr-2"></i>
          Sign In
        </button>
        <button
          @click="switchToRegister"
          :class="[
            'flex-1 py-3 px-4 text-sm font-medium transition-all duration-200',
            !isLogin 
              ? 'bg-white text-green-600 border-b-2 border-green-600' 
              : 'text-gray-600 hover:text-green-600'
          ]"
        >
          <i class="fas fa-user-plus mr-2"></i>
          Sign Up
        </button>
      </div>

      <!-- Form Content -->
      <div class="p-6">
        <!-- Login Form -->
        <form v-if="isLogin" key="login-form" @submit.prevent="submitForm" class="space-y-4" autocomplete="off">
          <!-- Username for login -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              v-model="loginData.username"
              type="text"
              name="login-username"
              autocomplete="username"
              :class="inputClasses"
              placeholder="Enter your username"
              required
            />
          </div>

          <!-- Password for login -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <div class="relative">
              <input
                v-model="loginData.password"
                :type="showPassword ? 'text' : 'password'"
                name="login-password"
                autocomplete="current-password"
                :class="inputClasses"
                placeholder="Enter your password"
                required
              />
              <button
                type="button"
                @click="showPassword = !showPassword"
                class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-green-600"
              >
                <i :class="showPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
              </button>
            </div>
          </div>

          <!-- Remember me for login -->
          <div class="flex items-center">
            <input
              id="remember-login"
              v-model="loginData.remember_me"
              type="checkbox"
              class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
            />
            <label for="remember-login" class="ml-2 block text-sm text-gray-700">
              Remember me
            </label>
          </div>

          <!-- Error message for login -->
          <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-3">
            <div class="flex items-center">
              <i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>
              <span class="text-red-700 text-sm">{{ error }}</span>
            </div>
          </div>

          <!-- Debug info for login -->
          <div v-if="debugMode" class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs">
            <div>Mode: Login</div>
            <div>Payload will be: {{ JSON.stringify(loginData, null, 2) }}</div>
          </div>

          <!-- Submit button for login -->
          <button
            type="submit"
            :disabled="isLoading"
            :class="[
              'w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center',
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
            ]"
          >
            <i v-if="isLoading" class="fas fa-spinner fa-spin mr-2"></i>
            <i v-else class="fas fa-sign-in-alt mr-2"></i>
            {{ isLoading ? 'Please wait...' : 'Sign In' }}
          </button>
        </form>

        <!-- Registration Form -->
        <form v-else key="register-form" @submit.prevent="submitForm" class="space-y-4" autocomplete="off">
          <!-- Full Name for registration -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Full Name
            </label>
            <input
              v-model="registerData.full_name"
              type="text"
              name="register-fullname"
              autocomplete="name"
              :class="inputClasses"
              placeholder="Enter your full name"
            />
          </div>

          <!-- Username for registration -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              v-model="registerData.username"
              type="text"
              name="register-username"
              autocomplete="new-username"
              :class="inputClasses"
              placeholder="Enter your username"
              required
            />
          </div>

          <!-- Email for registration -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Email
            </label>
            <input
              v-model="registerData.email"
              type="email"
              name="register-email"
              autocomplete="email"
              :class="inputClasses"
              placeholder="Enter your email"
              required
            />
          </div>

          <!-- Password for registration -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <div class="relative">
              <input
                v-model="registerData.password"
                :type="showPassword ? 'text' : 'password'"
                name="register-password"
                autocomplete="new-password"
                :class="inputClasses"
                placeholder="Enter your password"
                required
              />
              <button
                type="button"
                @click="showPassword = !showPassword"
                class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-green-600"
              >
                <i :class="showPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
              </button>
            </div>
          </div>

          <!-- Error message for registration -->
          <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-3">
            <div class="flex items-center">
              <i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>
              <span class="text-red-700 text-sm">{{ error }}</span>
            </div>
          </div>

          <!-- Debug info for registration -->
          <div v-if="debugMode" class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs">
            <div>Mode: Registration</div>
            <div>Payload will be: {{ JSON.stringify(registerData, null, 2) }}</div>
          </div>

          <!-- Submit button for registration -->
          <button
            type="submit"
            :disabled="isLoading"
            :class="[
              'w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center',
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
            ]"
          >
            <i v-if="isLoading" class="fas fa-spinner fa-spin mr-2"></i>
            <i v-else class="fas fa-user-plus mr-2"></i>
            {{ isLoading ? 'Please wait...' : 'Create Account' }}
          </button>
        </form>

        <!-- Guest option -->
        <div class="mt-6 pt-6 border-t border-gray-200">
          <button
            @click="continueAsGuest"
            class="w-full py-2 px-4 text-sm text-gray-600 hover:text-green-600 transition-colors duration-200 flex items-center justify-center"
          >
            <i class="fas fa-user-secret mr-2"></i>
            Continue as Guest (session-only history)
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, defineProps, defineEmits } from 'vue';
import { useAuthStore } from '../stores/auth';

const props = defineProps<{
  show: boolean;
}>();

const emit = defineEmits(['close', 'authenticated', 'guest']);

const authStore = useAuthStore();

const isLogin = ref(true);
const isLoading = ref(false);
const error = ref<string | null>(null);
const showPassword = ref(false);
const debugMode = ref(false);

const loginData = reactive({
  username: '',
  password: '',
  remember_me: false
});

const registerData = reactive({
  username: '',
  email: '',
  password: '',
  full_name: ''
});

const closeModal = () => {
  if (!isLoading.value) {
    emit('close');
  }
};

const switchToLogin = () => {
  isLogin.value = true;
  error.value = null;
};

const switchToRegister = () => {
  isLogin.value = false;
  error.value = null;
};

const submitForm = async () => {
  error.value = null;
  isLoading.value = true;
  const endpoint = isLogin.value ? '/api/auth/login' : '/api/auth/register';
  const payload = isLogin.value ? loginData : registerData;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'An error occurred.');
    }

    authStore.setAuthenticated(data);
    emit('authenticated', data);
    closeModal();
  } catch (err: any) {
    error.value = err.message;
  } finally {
    isLoading.value = false;
  }
};

const continueAsGuest = async () => {
  isLoading.value = true;
  try {
    const response = await fetch('/api/auth/guest', { method: 'POST' });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to create guest session');
    }
    emit('guest', data);
    closeModal();
  } catch (err: any) {
    error.value = err.message;
  } finally {
    isLoading.value = false;
  }
};

const inputClasses = computed(() => [
  'w-full px-4 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-green-500 focus:border-green-500 transition-all duration-200',
  'placeholder-gray-400 text-gray-900'
]);

// Reset form when modal is opened/closed
watch(() => props.show, (newVal) => {
  if (newVal) {
    isLogin.value = true;
    error.value = null;
    Object.assign(loginData, { username: '', password: '', remember_me: false });
    Object.assign(registerData, { full_name: '', username: '', email: '', password: '' });
  }
});
</script>

<style scoped>
/* Add Font Awesome if not already included */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

/* Custom animations */
@keyframes modalEnter {
  from {
    opacity: 0;
    transform: scale(0.9) translateY(-20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.modal-enter {
  animation: modalEnter 0.3s ease-out;
}
</style> 