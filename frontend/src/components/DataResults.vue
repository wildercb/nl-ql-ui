<template>
  <div class="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
      <h3 class="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center">
        <i class="fas fa-database mr-2 text-green-600"></i> Results
      </h3>
      <span v-if="results.length" class="text-xs text-gray-500">{{ results.length }} docs</span>
    </div>

    <div v-if="loading" class="text-center py-10">
      <i class="fas fa-spinner fa-spin text-green-500 text-2xl"></i>
    </div>

    <div v-else-if="results.length === 0" class="text-center py-10 text-gray-400">
      No data returned.
    </div>

    <!-- Results list -->
    <div v-else class="p-6 space-y-6 max-h-[50vh] overflow-y-auto custom-scrollbar">
      <div
        v-for="(doc, idx) in results"
        :key="idx"
        class="border border-gray-200 dark:border-gray-700 rounded-xl p-4 bg-gray-50 dark:bg-gray-800"
      >
        <pre class="text-xs whitespace-pre-wrap break-words text-gray-800 dark:text-gray-100">
{{ formatted(doc) }}
        </pre>
        <!-- If doc contains media urls show thumbnails -->
        <div v-if="mediaUrls(doc).length" class="flex flex-wrap gap-3 mt-2">
          <component
            v-for="(m, i) in mediaUrls(doc)"
            :key="i"
            :is="m.type === 'image' ? 'img' : 'video'"
            :src="m.url"
            class="max-h-40 rounded shadow"
            v-bind="m.type === 'video' ? { controls: true } : {}"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue'

interface Doc {
  [key: string]: any
}

const props = defineProps<{ results: Doc[]; loading: boolean }>()

const formatted = (doc: Doc) => JSON.stringify(doc, null, 2)

// naive media detection
const mediaUrls = (doc: Doc) => {
  const urls: { url: string; type: 'image' | 'video' }[] = []
  Object.values(doc).forEach((val) => {
    if (typeof val === 'string') {
      if (val.match(/\.(png|jpe?g|gif|webp)$/i)) {
        urls.push({ url: val, type: 'image' })
      } else if (val.match(/\.(mp4|webm|ogg)$/i)) {
        urls.push({ url: val, type: 'video' })
      }
    }
  })
  return urls
}
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