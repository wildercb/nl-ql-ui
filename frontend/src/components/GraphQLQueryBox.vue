<template>
  <div class="rounded-none border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
    <div class="px-6 py-3 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
      <h3 class="text-md font-semibold text-gray-800 dark:text-gray-100 flex items-center">
        <i class="fas fa-code mr-2 text-primary-600"></i> Generated GraphQL
        <span v-if="isUpdating" class="ml-2 text-xs text-green-400 animate-pulse">
          <i class="fas fa-sync-alt fa-spin mr-1"></i>Updating...
        </span>
      </h3>
      <button
        class="text-sm px-3 py-1 rounded bg-primary-600 hover:bg-primary-700 text-white disabled:opacity-50"
        :disabled="!query.trim()"
        @click="$emit('send')"
      >
        Send
      </button>
    </div>

    <!-- Editable GraphQL query box -->
    <textarea
      v-model="editableQuery"
      @input="updateParent"
      rows="10"
      class="w-full p-4 text-xs font-mono bg-transparent whitespace-pre-wrap break-words custom-scrollbar outline-none resize-y text-gray-800 dark:text-gray-100"
      :class="{ 'border-l-4 border-green-500': isUpdating }"
    ></textarea>
    <div v-if="formatError" class="text-xs text-red-400 p-2 bg-red-500/10 border-t border-red-400/20">
      <strong>Formatting Error:</strong> {{ formatError }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps, ref, watch } from 'vue'
import prettier from 'prettier/standalone'
import parserGraphql from 'prettier/plugins/graphql'

const props = defineProps<{ query: string }>()
const editableQuery = ref(props.query)
const isFormatting = ref(false)
const formatError = ref<string | boolean>(false)
const isUpdating = ref(false)

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
    editableQuery.value = 'Waiting for translation…'
    formatError.value = false
    return
  }
  isFormatting.value = true
  formatError.value = false
  try {
    const normalizedQuery = normalizeGql(query)
    editableQuery.value = await prettier.format(normalizedQuery, {
      parser: 'graphql',
      plugins: [parserGraphql],
      printWidth: 80
    })
  } catch (e: any) {
    console.error('GraphQL Formatting Error:', e)
    editableQuery.value = query // Fallback to raw query
    formatError.value = e.message || 'An unknown error occurred during formatting.'
  } finally {
    isFormatting.value = false
  }
}

// When parent provides a new query (from translator), prettify it and update the editor
watch(
  () => props.query,
  async (newQuery) => {
    // avoid overwriting if the user is currently editing (simple heuristic)
    if (newQuery !== editableQuery.value) {
      isUpdating.value = true
      await formatQuery(newQuery)
      // Keep the updating indicator for a moment to show the change
      setTimeout(() => {
        isUpdating.value = false
      }, 1000)
    }
  },
  { immediate: true }
)

// helper to emit change to parent
const updateParent = () => {
  // debounce optional, but fine for now
  emit('update:query', editableQuery.value)
}

const emit = defineEmits<{ 'update:query': [string]; send: [] }>()
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