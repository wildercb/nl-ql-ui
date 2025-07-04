<template>
  <div ref="container" class="rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg overflow-hidden bg-white dark:bg-gray-900">
    <!-- Header -->
    <div class="border-b border-gray-200 dark:border-gray-700 px-6 py-4 bg-white dark:bg-gray-900">
      <h3 class="text-lg font-bold flex items-center text-gray-800 dark:text-gray-100">
        <i class="fas fa-project-diagram mr-2 text-purple-600"></i>
        <span>{{ title }}</span>
      </h3>
    </div>

    <!-- Message list -->
    <div ref="scrollArea" class="max-h-[60vh] overflow-y-auto px-6 py-4 space-y-6 custom-scrollbar bg-white dark:bg-gray-900">
      <!-- Empty state -->
      <div v-if="messages.length === 0 && !loading" class="text-center text-gray-400">
        <i class="fas fa-robot text-4xl text-purple-500 mb-4"></i>
        <p>No messages yet</p>
      </div>

      <!-- Messages -->
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="flex items-start space-x-3"
      >
        <!-- Avatar -->
        <div
          :class="[
            'w-9 h-9 rounded-full flex items-center justify-center text-white text-sm flex-shrink-0',
            msg.role === 'user' ? 'bg-green-600' : 'bg-purple-600'
          ]"
        >
          {{ msg.role === 'user' ? 'U' : (msg.agent?.charAt(0).toUpperCase() || 'A') }}
        </div>

        <!-- Bubble -->
        <div class="flex-1">
          <div
            :class="[
              'inline-block rounded-xl px-4 py-3 whitespace-pre-wrap break-words shadow',
              msg.role === 'user'
                ? 'bg-green-50 text-gray-800 border border-green-200 dark:bg-green-900/30 dark:border-green-700 dark:text-green-100'
                : 'bg-purple-50 text-gray-800 border border-purple-200 dark:bg-purple-900/40 dark:border-purple-700 dark:text-purple-100'
            ]"
          >
            {{ msg.content }}
          </div>
          <div class="text-xs text-gray-500 mt-1">
            {{ msg.agent || 'You' }} • {{ msg.timestamp }}
          </div>
        </div>
      </div>

      <!-- Loading spinner -->
      <div v-if="loading" class="text-center py-4">
        <i class="fas fa-spinner fa-spin text-purple-500"></i>
      </div>
    </div>

    <!-- Optional footer slot for future input -->
    <slot name="footer" />
  </div>
</template>

<script setup lang="ts">
import { defineProps, ref, watch, nextTick } from 'vue'

interface Message {
  role: 'user' | 'agent'
  agent?: string
  content: string
  timestamp: string
}

const props = defineProps<{ title?: string; messages: Message[]; loading: boolean }>()

const scrollArea = ref<HTMLElement | null>(null)

const scrollToBottom = () => {
  nextTick(() => {
    if (scrollArea.value) {
      scrollArea.value.scrollTop = scrollArea.value.scrollHeight
    }
  })
}

watch(
  () => props.messages.length,
  () => scrollToBottom()
)
</script>

<style scoped>
/* Custom scrollbar styling */
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: rgba(156, 163, 175, 0.6); /* gray-500 */
  border-radius: 3px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background-color: transparent;
}
</style> 