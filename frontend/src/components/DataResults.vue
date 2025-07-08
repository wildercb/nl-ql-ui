<template>
  <div class="rounded-none border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
      <h3 class="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center">
        <i class="fas fa-database mr-2 text-green-600"></i> Results
      </h3>
      <div class="flex items-center gap-2">
        <span v-if="results.length" class="text-xs text-gray-500">{{ results.length }} docs</span>
        <!-- Download button -->
        <button
          @click="downloadJSON"
          :disabled="!results.length"
          class="bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold px-3 py-1 rounded-md flex items-center"
        >
          <i class="fas fa-download mr-1"></i> Download
        </button>
      </div>
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
          <a
            v-for="(m, i) in mediaUrls(doc)"
            :key="i"
            :href="m.url"
            target="_blank"
            rel="noopener noreferrer"
            class="group relative"
          >
            <component
              :is="m.type === 'image' ? 'img' : 'video'"
              :src="m.url"
              class="max-h-40 rounded shadow object-cover"
              v-bind="m.type === 'video' ? { controls: true } : {}"
            />
            <span class="absolute bottom-1 right-1 text-[10px] bg-black/60 text-white px-1 rounded opacity-0 group-hover:opacity-100">
              {{ m.type }}
            </span>
          </a>
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

// 🔍 Enhanced (recursive) media detection – walks nested objects/arrays.
const mediaUrls = (doc: Doc) => {
  const urls: { url: string; type: 'image' | 'video' }[] = []

  const visit = (value: any) => {
    if (!value) return
    if (typeof value === 'string') {
      if (value.match(/\.(png|jpe?g|gif|webp)$/i)) {
        urls.push({ url: value, type: 'image' })
      } else if (value.match(/\.(mp4|webm|ogg)$/i)) {
        urls.push({ url: value, type: 'video' })
      }
    } else if (Array.isArray(value)) {
      value.forEach(visit)
    } else if (typeof value === 'object') {
      Object.values(value).forEach(visit)
    }
  }

  visit(doc)
  return urls
}

// 📥 Download the entire results array as a JSON file
const downloadJSON = () => {
  if (!props.results.length) return
  const blob = new Blob([JSON.stringify(props.results, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'results.json'
  link.click()
  URL.revokeObjectURL(url)
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