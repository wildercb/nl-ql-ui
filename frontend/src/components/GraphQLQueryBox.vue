<template>
  <div class="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
    <div class="px-6 py-3 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
      <h3 class="text-md font-semibold text-gray-800 dark:text-gray-100 flex items-center">
        <i class="fas fa-code mr-2 text-blue-600"></i> Generated GraphQL
      </h3>
      <button
        class="text-sm px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
        :disabled="!query.trim()"
        @click="$emit('send')"
      >
        Send
      </button>
    </div>

    <pre class="p-4 text-xs whitespace-pre-wrap break-words max-h-60 overflow-y-auto custom-scrollbar text-gray-800 dark:text-gray-100">
{{ query || 'Waiting for translation…' }}
    </pre>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue'

defineProps<{ query: string }>()
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: rgba(156, 163, 175, 0.6);
  border-radius: 3px;
}
</style> 