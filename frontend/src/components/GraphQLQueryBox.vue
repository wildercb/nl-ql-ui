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

    <pre class="p-4 text-xs whitespace-pre break-words max-h-60 overflow-y-auto custom-scrollbar text-gray-800 dark:text-gray-100">
{{ displayQuery }}
    </pre>
    <div v-if="formatError" class="text-xs text-red-400 p-2 bg-red-500/10 border-t border-red-400/20">
      <strong>Formatting Error:</strong> {{ formatError }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps, ref, watch, computed } from 'vue'
import prettier from 'prettier/standalone'
import parserGraphql from 'prettier/plugins/graphql'

const props = defineProps<{ query: string }>()
const formattedQuery = ref('')
const isFormatting = ref(false)
const formatError = ref<string | boolean>(false)

/**
 * Attempts to normalize a GraphQL query string to fix common syntax issues.
 * Currently, it replaces single-quoted strings with double-quoted strings.
 * @param query The raw GraphQL query string.
 * @returns A normalized query string.
 */
const normalizeGql = (query: string): string => {
  // Replace single quotes used for strings with double quotes.
  // e.g., { user(id: '1') } => { user(id: "1") }
  // This is a common variation that the strict GraphQL parser dislikes.
  return query.replace(/:\s*'([^']*)'/g, ': "$1"')
}

const formatQuery = async (query: string) => {
  if (!query?.trim()) {
    formattedQuery.value = 'Waiting for translation…'
    formatError.value = false
    return
  }
  isFormatting.value = true
  formatError.value = false
  try {
    const normalizedQuery = normalizeGql(query)
    formattedQuery.value = await prettier.format(normalizedQuery, {
      parser: 'graphql',
      plugins: [parserGraphql],
      printWidth: 80
    })
  } catch (e: any) {
    console.error('GraphQL Formatting Error:', e)
    formattedQuery.value = query // Fallback to raw query
    formatError.value = e.message || 'An unknown error occurred during formatting.'
  } finally {
    isFormatting.value = false
  }
}

watch(() => props.query, (newQuery) => {
  formatQuery(newQuery)
}, { immediate: true })

const displayQuery = computed(() => isFormatting.value ? 'Formatting…' : formattedQuery.value)
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