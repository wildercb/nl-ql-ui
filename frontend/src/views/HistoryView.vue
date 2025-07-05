<template>
  <div class="flex-1 bg-gray-900 text-white">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold">Query History</h1>
        <p class="text-lg text-gray-400">
          Review your past queries and their results.
        </p>
      </div>
      
      <div v-if="isLoading" class="text-center">
        <i class="fas fa-spinner fa-spin text-3xl"></i>
        <p class="mt-2">Loading history...</p>
      </div>

      <div v-else-if="error" class="bg-red-900 border border-red-700 text-red-300 px-4 py-3 rounded-lg">
        <p><strong>Error:</strong> {{ error }}</p>
      </div>

      <div v-else-if="history.length === 0" class="text-center text-gray-500 py-8">
        <p>No translation history available yet.</p>
        <p class="text-sm mt-2">Start translating queries to see your history here.</p>
      </div>

      <div v-else class="bg-gray-800 shadow-xl rounded-lg overflow-hidden">
        <ul class="divide-y divide-gray-700">
          <li v-for="item in history" :key="item.id" class="p-4 hover:bg-gray-700 transition-colors duration-200">
            <div class="flex items-center justify-between">
              <div class="flex-grow">
                <p class="text-sm text-gray-400 truncate">{{ new Date(item.timestamp).toLocaleString() }}</p>
                <p class="font-semibold text-white mt-1">{{ item.natural_query }}</p>
              </div>
              <div class="flex items-center space-x-4">
                <span :class="item.confidence > 0.8 ? 'text-green-400' : 'text-yellow-400'">
                  Confidence: {{ (item.confidence * 100).toFixed(0) }}%
                </span>
                <button @click="viewHistoryItem(item)" class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded-md text-sm font-semibold">
                  View
                </button>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';
import { useHistoryStore } from '../stores/history';

const authStore = useAuthStore();
const historyStore = useHistoryStore();
const router = useRouter();

const history = ref<any[]>([]);
const isLoading = ref(true);
const error = ref<string | null>(null);

onMounted(async () => {
  if (authStore.isAuthenticated) {
    try {
      const response = await fetch('/api/history', {
        headers: {
          'Authorization': `Bearer ${authStore.accessToken}`
        }
      });
      if (!response.ok) {
        throw new Error('Failed to fetch history.');
      }
      const data = await response.json();
      history.value = data.history || [];
    } catch (err: any) {
      error.value = err.message;
    } finally {
      isLoading.value = false;
    }
  } else if (authStore.isGuestSession) {
    history.value = authStore.guestHistory;
    isLoading.value = false;
  } else {
    error.value = "Please sign in or start a guest session to view history.";
    isLoading.value = false;
  }
});

const viewHistoryItem = (item: any) => {
  historyStore.setSelectedHistoryItem(item);
  router.push('/');
};
</script> 