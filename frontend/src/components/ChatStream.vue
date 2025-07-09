<template>
  <div class="h-full flex flex-col bg-gray-800 rounded-none border border-gray-700">
    <!-- Header -->
    <div class="px-4 py-2 border-b border-gray-700 flex items-center">
      <h3 class="text-md font-semibold text-gray-200">{{ title || 'Agent Stream' }}</h3>
      <button
        class="ml-auto text-xs px-3 py-1 rounded-md transition-colors duration-200 focus:outline-none"
        :class="showPrompts ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600 text-gray-300'"
        @click="togglePrompts"
      >
        <i class="fas fa-code mr-1" />{{ showPrompts ? 'Hide Prompts' : 'Show Prompts' }}
      </button>
    </div>
    
    <!-- Messages -->
    <div ref="scrollArea" class="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 custom-scrollbar">
      <div v-if="displayMessages.length === 0" class="text-center text-gray-400 pt-8">
        <i class="fas fa-comments text-4xl mb-2"></i>
        <p class="font-semibold">{{ title }}</p>
        <p class="text-sm">Enter a query below to start the conversation.</p>
      </div>

      <div v-for="(msg, index) in displayMessages" :key="index" class="w-full">
        <!-- User Message -->
        <div v-if="msg.role === 'user'" class="flex items-end justify-end w-full">
          <div class="bg-primary-600 text-white p-3 rounded-lg w-full shadow-md break-words">
            <p class="whitespace-pre-wrap" v-html="renderMarkdown(msg.content)"></p>
            <div class="text-right text-xs text-primary-200 mt-1">{{ msg.timestamp }}</div>
          </div>
          <div class="w-8 h-8 ml-2 rounded-full bg-primary-600 text-white flex items-center justify-center font-bold">U</div>
        </div>

        <!-- Agent / Assistant Message -->
        <div v-else class="flex items-start w-full mt-2">
          <div class="w-8 h-8 mr-2 rounded-full bg-gray-700 text-white flex items-center justify-center">
            <i :class="getAgentIcon(msg.agent)" />
          </div>
          <div class="agent-bubble p-3 rounded-lg w-full shadow-md whitespace-pre-wrap break-words">
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
          class="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <i class="fas fa-paper-plane"></i>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue';
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
  prompts?: any[];
}>();

const emit = defineEmits<{
  sendMessage: [message: string];
}>();

const scrollArea = ref<HTMLElement | null>(null);
const chatInput = ref('');
const isChatProcessing = ref(false);
const showPrompts = ref(false);

const togglePrompts = () => {
  showPrompts.value = !showPrompts.value;
  scrollToBottom();
};

// Build a mapping of agent → prompt message once prompts are available
const promptMessagesByAgent = computed<Record<string, Message>>(() => {
  if (!showPrompts.value || !props.prompts) return {};
  const map: Record<string, Message> = {};
  props.prompts.forEach((p: any) => {
    let promptContent: string;
    if (Array.isArray(p.prompt)) {
      promptContent = p.prompt.map((m: any) => `${m.role}: ${m.content}`).join('\n');
    } else {
      promptContent = typeof p.prompt === 'object' ? JSON.stringify(p.prompt, null, 2) : String(p.prompt);
    }
    map[p.agent] = {
      role: 'agent',
      agent: `prompt → ${p.agent}`,
      content: `\u0060\u0060\u0060\n${promptContent}\n\u0060\u0060\u0060`,
      // Use p.timestamp if provided, else current time to preserve ordering roughly
      timestamp: p.timestamp || new Date().toLocaleTimeString(),
      isStreaming: false
    } as Message;
  });
  return map;
});

// Interleave prompts directly before the first message of their agent
const displayMessages = computed<Message[]>(() => {
  if (!showPrompts.value) return props.messages;

  const inserted = new Set<string>();
  const result: Message[] = [];

  props.messages.forEach((msg) => {
    if (
      msg.role === 'agent' &&
      msg.agent &&
      promptMessagesByAgent.value[msg.agent] &&
      !inserted.has(msg.agent)
    ) {
      result.push(promptMessagesByAgent.value[msg.agent]);
      inserted.add(msg.agent);
    }
    result.push(msg);
  });

  // In case some prompts have no corresponding agent messages yet (e.g., translator prompt before response)
  Object.keys(promptMessagesByAgent.value).forEach((agent) => {
    if (!inserted.has(agent)) {
      result.push(promptMessagesByAgent.value[agent]);
    }
  });

  return result;
});

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
    system: 'fas fa-cogs',
    prompt: 'far fa-file-alt'
  };
  if (agentName && agentName.startsWith('prompt')) return icons['prompt'];
  return icons[agentName || 'system'] || 'fas fa-robot';
};

watch(() => props.prompts, scrollToBottom, { deep: true });
watch(showPrompts, scrollToBottom);
watch(() => props.messages, scrollToBottom, { deep: true });
watch(() => props.loading, scrollToBottom);

</script>

<style scoped>
/* Agent / Assistant bubble styling */
.agent-bubble {
  @apply bg-blue-600 text-white;
  /* Slightly opaque for readability but less translucent than other blues */
  background-color: rgba(37, 99, 235, 0.9); /* Tailwind blue-600 with 90% opacity */
}

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