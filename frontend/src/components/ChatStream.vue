<template>
  <div class="h-full flex flex-col bg-gray-800 rounded-none border border-gray-700">
    <!-- Header -->
    <div class="px-4 py-2 border-b border-gray-700">
      <h3 class="text-md font-semibold text-gray-200">{{ title || 'Agent Stream' }}</h3>
    </div>
    
    <!-- Messages -->
    <div ref="scrollArea" class="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 custom-scrollbar">
      <div v-if="messages.length === 0" class="text-center text-gray-400 pt-8">
        <i class="fas fa-comments text-4xl mb-2"></i>
        <p class="font-semibold">{{ title }}</p>
        <p class="text-sm">Enter a query below to start the conversation.</p>
      </div>

      <div v-for="(msg, index) in messages" :key="index" class="w-full">
        <!-- User Message -->
        <div v-if="msg.role === 'user'" class="flex items-end justify-end w-full">
          <div class="bg-blue-600 text-white p-3 rounded-lg w-full shadow-md break-words">
            <p class="whitespace-pre-wrap" v-html="renderMarkdown(msg.content)"></p>
            <div class="text-right text-xs text-blue-200 mt-1">{{ msg.timestamp }}</div>
          </div>
          <div class="w-8 h-8 ml-2 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">U</div>
        </div>

        <!-- Agent / Assistant Message -->
        <div v-else class="flex items-start w-full mt-2">
          <div class="w-8 h-8 mr-2 rounded-full bg-gray-700 text-white flex items-center justify-center">
            <i :class="getAgentIcon(msg.agent)" />
          </div>
          <div class="bg-gray-700 p-3 rounded-lg w-full shadow-md break-words">
            <p class="font-bold text-purple-400 text-sm capitalize mb-1">{{ msg.agent }}</p>
            <div class="text-white prose prose-sm max-w-none" v-html="renderMarkdown(msg.content)" />
            <div v-if="msg.isStreaming" class="typing-indicator">
              <span></span><span></span><span></span>
            </div>
            <div class="text-left text-xs text-gray-400 mt-1">{{ msg.timestamp }}</div>
          </div>
        </div>
      </div>
       <div v-if="loading" class="flex items-center justify-center p-4">
          <i class="fas fa-spinner fa-spin text-2xl text-purple-500"></i>
          <span class="ml-2">Processing...</span>
      </div>
    </div>

    <!-- Chat Input -->
    <div class="p-4 border-t border-gray-700">
      <div class="flex space-x-2">
        <input
          v-model="chatInput"
          @keyup.enter="sendMessage"
          :disabled="isChatProcessing"
          type="text"
          placeholder="Continue the conversation..."
          class="flex-1 bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
        />
        <button
          @click="sendMessage"
          :disabled="!chatInput.trim() || isChatProcessing"
          class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <i class="fas fa-paper-plane"></i>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';
import { marked } from 'marked';

interface Message {
  role: 'user' | 'agent';
  agent?: string;
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

const props = defineProps<{
  title?: string;
  messages: Message[];
  loading: boolean;
  selectedModel?: string;
}>();

const emit = defineEmits<{
  sendMessage: [message: string];
}>();

const scrollArea = ref<HTMLElement | null>(null);
const chatInput = ref('');
const isChatProcessing = ref(false);

const scrollToBottom = () => {
  nextTick(() => {
    if (scrollArea.value) {
      scrollArea.value.scrollTop = scrollArea.value.scrollHeight;
    }
  });
};

const sendMessage = async () => {
  if (!chatInput.value.trim() || isChatProcessing.value) return;
  
  const message = chatInput.value.trim();
  chatInput.value = '';
  isChatProcessing.value = true;
  
  try {
    emit('sendMessage', message);
  } catch (error) {
    console.error('Failed to send message:', error);
  } finally {
    isChatProcessing.value = false;
  }
};

const renderMarkdown = (content: string) => {
  if (!content) return '';
  return marked(content, { gfm: true, breaks: true });
};

const messageClass = (msg: Message) => {
  return msg.role === 'user' ? 'justify-end' : 'justify-start';
};

const getAgentIcon = (agentName?: string) => {
  const icons: { [key: string]: string } = {
    rewriter: 'fas fa-pencil-alt',
    translator: 'fas fa-language',
    reviewer: 'fas fa-check-double',
    optimizer: 'fas fa-bolt',
    system: 'fas fa-cogs'
  };
  return icons[agentName || 'system'] || 'fas fa-robot';
};

watch(() => props.messages, scrollToBottom, { deep: true });
watch(() => props.loading, scrollToBottom);

</script>

<style scoped>
.prose {
  color: inherit;
}
.prose :where(code):not(:where([class~="not-prose"] *))::before,
.prose :where(code):not(:where([class~="not-prose"] *))::after {
  content: none;
}
.prose :where(pre) {
    background-color: #111827; /* bg-gray-900 */
    color: #e5e7eb; /* text-gray-200 */
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    overflow-x: auto;
}
.prose :where(code) {
    background-color: #374151; /* bg-gray-700 */
    color: #f3f4f6; /* text-gray-100 */
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
    font-size: 0.85em;
    font-weight: 600;
}
.prose pre code {
  background-color: transparent;
  padding: 0;
  font-weight: inherit;
}

.custom-scrollbar::-webkit-scrollbar {
  width: 8px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: #4B5563; /* bg-gray-600 */
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background-color: transparent;
}

.typing-indicator {
  display: inline-block;
  margin-left: 6px;
}
.typing-indicator span {
  height: 6px;
  width: 6px;
  background-color: #a78bfa; /* purple-400 */
  border-radius: 50%;
  display: inline-block;
  animation: wave 1.3s infinite;
}
.typing-indicator span:nth-of-type(2) {
  animation-delay: -1.1s;
}
.typing-indicator span:nth-of-type(3) {
  animation-delay: -0.9s;
}
@keyframes wave {
  0%, 60%, 100% { transform: translate(0, 0); }
  30% { transform: translate(0, -6px); }
}
</style> 