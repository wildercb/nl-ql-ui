<template>
  <div class="h-full flex flex-col bg-gray-800 rounded-none border border-gray-700">
    <!-- Header -->
    <div class="px-4 py-2 border-b border-gray-700 flex items-center">
      <h3 class="text-md font-semibold text-gray-200">{{ title || 'Agent Stream' }}</h3>
      <button
        class="ml-auto text-xs px-3 py-1 rounded-md transition-colors duration-200 focus:outline-none"
        :class="props.showPrompts ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600 text-gray-300'"
        @click="togglePrompts"
      >
        <i class="fas fa-code mr-1" />{{ props.showPrompts ? 'Hide Prompts' : 'Show Prompts' }}
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
          <div 
            class="agent-bubble p-3 rounded-lg w-full shadow-md whitespace-pre-wrap break-words"
            :class="msg.isPrompt ? 'bg-purple-800 border-l-4 border-purple-400' : ''"
          >
            <p class="font-bold text-purple-400 text-sm capitalize mb-1">
              {{ msg.isPrompt ? 'Prompt' : msg.agent }}
            </p>
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
import { ref, onMounted, watch, nextTick, computed } from 'vue';
import { storeToRefs } from 'pinia';
import { useAuthStore } from '@/stores/auth';
import { mcpClient } from '@/services/mcpClient';
import MarkdownIt from 'markdown-it';

const props = defineProps({
  messages: {
    type: Array as () => any[],
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  title: {
    type: String,
    default: 'Agent Stream',
  },
  selectedModel: {
    type: String,
    default: 'phi3:mini',
  },
  prompts: {
    type: Array as () => any[],
    default: () => [],
  },
  showPrompts: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(['sendMessage', 'toggle-prompts']);

const authStore = useAuthStore();
const { isAuthenticated, user } = storeToRefs(authStore);

const chatInput = ref('');
const scrollArea = ref<HTMLElement | null>(null);

const md = new MarkdownIt();

const renderMarkdown = (content: string) => {
  return md.render(content || '');
};

const displayMessages = computed(() => {
  let allMessages = [...props.messages];
  
  // If showPrompts is enabled, interleave prompts with messages
  if (props.showPrompts && props.prompts.length > 0) {
    const messagesWithPrompts: any[] = [];
    
    // Group prompts by agent for better organization
    const promptsByAgent = props.prompts.reduce((acc, prompt) => {
      if (!acc[prompt.agent]) {
        acc[prompt.agent] = [];
      }
      acc[prompt.agent].push(prompt);
      return acc;
    }, {} as Record<string, any[]>);
    
    // Add regular messages and prompts
    props.messages.forEach(msg => {
      messagesWithPrompts.push(msg);
      
      // If this is an agent message, add its prompts after it
      if (msg.agent && promptsByAgent[msg.agent]) {
        promptsByAgent[msg.agent].forEach(prompt => {
          messagesWithPrompts.push({
            role: 'system',
            agent: `${msg.agent}_prompt`,
            content: `**Prompt for ${prompt.agent}:**\n\n\`\`\`\n${prompt.prompt}\n\`\`\``,
            timestamp: prompt.timestamp,
            isPrompt: true
          });
        });
        // Remove the prompts we just added so they don't get duplicated
        delete promptsByAgent[msg.agent];
      }
    });
    
    // Add any remaining prompts that weren't associated with messages
    Object.entries(promptsByAgent).forEach(([agent, prompts]) => {
      prompts.forEach(prompt => {
        messagesWithPrompts.push({
          role: 'system',
          agent: `${agent}_prompt`,
          content: `**Prompt for ${prompt.agent}:**\n\n\`\`\`\n${prompt.prompt}\n\`\`\``,
          timestamp: prompt.timestamp,
          isPrompt: true
        });
      });
    });
    
    allMessages = messagesWithPrompts;
  }
  
  return allMessages.map(msg => ({
    ...msg,
    timestamp: msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : '',
  }));
});

const isChatProcessing = computed(() => props.loading);

const getAgentIcon = (agentName: string) => {
  const icons: { [key: string]: string } = {
    user: 'fas fa-user',
    assistant: 'fas fa-robot',
    system: 'fas fa-cogs',
    mcp_pipeline: 'fas fa-project-diagram',
    agent: 'fas fa-microchip',
    translator: 'fas fa-language',
    rewriter: 'fas fa-pen-fancy',
    reviewer: 'fas fa-check-double',
    data_query: 'fas fa-database',
    default: 'fas fa-question-circle',
  };
  
  // Handle prompt messages
  if (agentName.endsWith('_prompt')) {
    return 'fas fa-code';
  }
  
  return icons[agentName] || icons.default;
};

const togglePrompts = () => {
  emit('toggle-prompts');
};

const sendMessage = () => {
  if (chatInput.value.trim()) {
    emit('sendMessage', chatInput.value);
    chatInput.value = '';
  }
};

const scrollToBottom = () => {
  nextTick(() => {
    if (scrollArea.value) {
      scrollArea.value.scrollTop = scrollArea.value.scrollHeight;
    }
  });
};

watch(() => props.messages, scrollToBottom, { deep: true, immediate: true });

onMounted(() => {
  scrollToBottom();
});
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