<template>
  <div class="flex h-screen bg-gray-900 text-white">
    <!-- Sidebar -->
    <div class="w-64 bg-gray-800 p-4 flex flex-col">
      <h1 class="text-2xl font-bold mb-4">MPPW-MCP</h1>
      <nav class="flex flex-col space-y-2">
        <router-link to="/" class="text-gray-300 hover:text-white"><i class="fas fa-home mr-2"></i>Home</router-link>
        <router-link to="/history" class="text-gray-300 hover:text-white"><i class="fas fa-history mr-2"></i>History</router-link>
        <router-link to="/docs" class="text-gray-300 hover:text-white"><i class="fas fa-book mr-2"></i>Docs</router-link>
      </nav>
      <div class="mt-auto">
        <div v-if="authStore.isAuthenticated" class="flex items-center">
          <span class="text-sm">Welcome, {{ authStore.isGuest ? 'Guest' : authStore.currentUser?.email }}</span>
          <button @click="handleLogout" class="ml-auto text-red-400 hover:text-red-300">
            <i class="fas fa-sign-out-alt"></i>
          </button>
        </div>
        <button v-else @click="handleSignIn" class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
          Sign In
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex flex-col p-6 overflow-hidden">
      <!-- Input Area -->
      <div class="mb-4">
        <textarea
          v-model="naturalQuery"
          @keyup.enter="runPipeline('standard')"
          :disabled="isProcessing"
          class="w-full p-3 bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          placeholder="Enter your natural language query here..."
        ></textarea>
        <div class="flex space-x-2 mt-2 items-center">
          <button @click="runPipeline('fast')" :disabled="isProcessing" class="btn-primary">
            <i class="fas fa-bolt mr-2"></i> Translate
          </button>
          <button @click="runPipeline('standard')" :disabled="isProcessing" class="btn-primary">
            <i class="fas fa-users-cog mr-2"></i> Multi-Agent
          </button>
          <button @click="runPipeline('comprehensive')" :disabled="isProcessing" class="btn-primary">
            <i class="fas fa-rocket mr-2"></i> Enhanced Agents
          </button>
          <select v-model="selectedModel" class="ml-4 p-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500" :disabled="isProcessing">
            <option value="phi3:mini">phi3:mini</option>
            <option value="gemma3:4b">gemma3:4b</option>
            <option value="llama4:16x17b">llama4:16x17b</option>
          </select>
        </div>
      </div>

      <!-- Results Area -->
      <div class="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden">
        <!-- Left Panel: Chat Stream -->
        <div class="flex flex-col overflow-hidden">
          <ChatStream 
            :messages="chatMessages" 
            :loading="isProcessing" 
            title="Agent Stream" 
            :selectedModel="selectedModel"
            @sendMessage="handleChatMessage"
          />
        </div>

        <!-- Right Panel: GraphQL and Data -->
        <div class="flex flex-col space-y-6 overflow-hidden">
          <GraphQLQueryBox :query="finalGraphQLQuery" @send="runDataQuery" />
          <DataResults :results="dataQueryResults" :loading="isDataLoading" />
        </div>
      </div>
    </div>

    <AuthModal :show="showAuthModal" @close="showAuthModal = false" @authenticated="onAuthenticated" @guest="onGuestSession" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';
import { useHistoryStore } from '../stores/history';
import AuthModal from '../components/AuthModal.vue';
import ChatStream from '../components/ChatStream.vue';
import GraphQLQueryBox from '../components/GraphQLQueryBox.vue';
import DataResults from '../components/DataResults.vue';

const authStore = useAuthStore();
const historyStore = useHistoryStore();
const router = useRouter();

// Model selection
const selectedModel = ref('phi3:mini');

// Pipeline state
const naturalQuery = ref('');
const chatMessages = ref<any[]>([]);
const isProcessing = ref(false);
const showAuthModal = ref(false);

const finalGraphQLQuery = ref('');
const isDataLoading = ref(false);
const dataQueryResults = ref<any[]>([]);

// Agent prompts and context
const agentPrompts = ref<any[]>([]);
const agentContext = ref<string>('');

const runPipeline = async (strategy: string) => {
  if (!naturalQuery.value.trim() || isProcessing.value) return;

  console.log('🚀 Starting pipeline with strategy:', strategy, 'using model:', selectedModel.value);
  isProcessing.value = true;
  chatMessages.value = [];
  finalGraphQLQuery.value = '';
  dataQueryResults.value = [];
  agentPrompts.value = [];

  chatMessages.value.push({ role: 'user', content: naturalQuery.value, timestamp: new Date().toLocaleTimeString() });

  try {
    console.log('📡 Making request to /api/multiagent/process/stream');
    const response = await fetch('/api/multiagent/process/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query: naturalQuery.value, 
        pipeline_strategy: strategy,
        translator_model: selectedModel.value,
        pre_model: selectedModel.value,
        review_model: selectedModel.value
      }),
    });

    console.log('📥 Response status:', response.status, 'headers:', Object.fromEntries(response.headers.entries()));

    if (!response.body) throw new Error("Response body is null");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let eventCount = 0;

    console.log('🔄 Starting to read stream...');
    while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('✅ Stream complete, processed', eventCount, 'events');
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        console.log('📦 Raw chunk received:', chunk.length, 'bytes');
        console.log('📦 Raw chunk content:', JSON.stringify(chunk));
        buffer += chunk;
        
        // Handle both \r\n and \n line endings for SSE
        const eventMessages = buffer.split(/\r?\n\r?\n/);
        console.log('🔍 Buffer after adding chunk:', JSON.stringify(buffer));
        console.log('🔍 Split into', eventMessages.length, 'potential messages');
        buffer = eventMessages.pop() || '';

        console.log('🔍 Processing', eventMessages.length, 'event messages');
        for (const msg of eventMessages) {
            console.log('📝 Processing message:', JSON.stringify(msg));
            console.log('📝 Raw message:', msg);
            const lines = msg.split(/\r?\n/);
            console.log('📝 Split into lines:', lines);
            let eventType = 'message';
            let dataStr = '';
            for (const line of lines) {
                if (line.startsWith('event:')) {
                    eventType = line.slice(6).trim();
                    console.log('🎯 Found event type:', eventType);
                } else if (line.startsWith('data:')) {
                    dataStr += line.slice(5).trim();
                    console.log('🎯 Found data line:', line.slice(5).trim());
                }
            }
            console.log('🎯 Parsed event type:', eventType, 'data length:', dataStr.length);
            if (!dataStr) {
              console.log('⚠️ No data found, skipping');
              continue;
            }
            try {
                const parsed = JSON.parse(dataStr);
                console.log('✅ Successfully parsed SSE event:', parsed);
                eventCount++;
                handleStreamEvent(parsed);
            } catch (e) {
                console.error('❌ Failed to parse SSE data', dataStr, e);
            }
        }
    }
  } catch (error) {
    console.error('💥 Pipeline error:', error);
  } finally {
    console.log('🏁 Pipeline finished, setting isProcessing to false');
    isProcessing.value = false;
  }
};

const handleStreamEvent = (event: any) => {
  console.log('🎮 Handling stream event:', event);
  
  const eventType = event.event;
  const data = event.data || event;
  
  console.log('📊 Event type:', eventType, 'Event data:', data);
  
  switch (eventType) {
    case 'agent_start':
      console.log('🚀 Agent starting:', data.agent);
      const startMessage = {
        role: 'agent',
        agent: data.agent,
        content: `Agent [${data.agent}] started...`,
        timestamp: new Date().toLocaleTimeString(),
        isStreaming: true,
      };
      chatMessages.value.push(startMessage);
      console.log('💬 Added start message, total messages:', chatMessages.value.length);
      break;
      
    case 'agent_token':
      console.log('🔤 Token received for agent:', data.agent, 'token:', data.token);
      const lastMessage = chatMessages.value[chatMessages.value.length - 1];
      if (lastMessage && lastMessage.agent === data.agent && lastMessage.isStreaming) {
        if (lastMessage.content.endsWith('...')) {
          lastMessage.content = `Agent [${data.agent}] says: `;
        }
        lastMessage.content += data.token;
        console.log('📝 Updated message content:', lastMessage.content);
      } else {
        console.log('⚠️ Could not find streaming message for agent:', data.agent);
      }
      break;
      
    case 'agent_complete':
      console.log('✅ Agent completed:', data.agent, 'result:', data.result);
      const agentMessage = chatMessages.value.find(m => m.agent === data.agent && m.isStreaming);
      if (agentMessage) {
        agentMessage.isStreaming = false;
        if (data.agent === 'reviewer') {
          agentMessage.content = formatReviewResult(data.result);
          // If reviewer suggests a new query, update the GraphQL query in the UI
          if (data.result && data.result.suggested_query) {
            console.log('📝 Reviewer suggested new GraphQL query:', data.result.suggested_query);
            finalGraphQLQuery.value = data.result.suggested_query;
          }
        } else if (data.agent === 'rewriter') {
          agentMessage.content = data.result?.rewritten_query || '';
        } else if (data.agent === 'translator' && data.result?.graphql_query) {
          console.log('🔍 Setting GraphQL query:', data.result.graphql_query);
          finalGraphQLQuery.value = data.result.graphql_query;
        }
        console.log('💬 Marked agent as completed, kept output or set result');
      } else {
        console.log('⚠️ Could not find agent message for completion:', data.agent);
      }
      
      // Store agent prompts for context
      if (data.prompt) {
        agentPrompts.value.push({
          agent: data.agent,
          prompt: data.prompt,
          result: data.result
        });
      }
      break;
      
    case 'pipeline_complete':
      console.log('🏁 Pipeline completed, final data:', data);
      if (data.translation?.graphql_query && !finalGraphQLQuery.value) {
        console.log('🔍 Setting GraphQL query from pipeline complete:', data.translation.graphql_query);
        finalGraphQLQuery.value = data.translation.graphql_query;
      }
      chatMessages.value.push({
        role: 'agent',
        agent: 'system',
        content: 'Pipeline completed.',
        timestamp: new Date().toLocaleTimeString(),
      });
      console.log('💬 Added pipeline complete message');
      
      // Build context for chat
      buildChatContext();
      break;
      
    case 'error':
      console.log('❌ Error event received:', data.error);
      chatMessages.value.push({
        role: 'agent',
        agent: 'system',
        content: `Error: ${data.error || 'Unknown error'}`,
        timestamp: new Date().toLocaleTimeString(),
      });
      console.log('💬 Added error message');
      break;
      
    default:
      console.log('❓ Unknown event type:', eventType);
  }
  
  console.log('📊 Current chat messages:', chatMessages.value);
  console.log('🔍 Current GraphQL query:', finalGraphQLQuery.value);
};

const buildChatContext = () => {
  // Build context from agent responses and prompts
  let context = `Original Query: ${naturalQuery.value}\n\n`;
  
  // Add agent responses
  context += 'Agent Responses:\n';
  chatMessages.value.forEach(msg => {
    if (msg.role === 'agent' && msg.agent) {
      context += `${msg.agent}: ${msg.content}\n`;
    }
  });
  
  // Add agent prompts
  if (agentPrompts.value.length > 0) {
    context += '\nAgent Prompts:\n';
    agentPrompts.value.forEach(prompt => {
      context += `${prompt.agent}:\n`;
      if (Array.isArray(prompt.prompt)) {
        prompt.prompt.forEach((msg: any) => {
          context += `${msg.role}: ${msg.content}\n`;
        });
      } else {
        context += `${prompt.prompt}\n`;
      }
      context += `Result: ${JSON.stringify(prompt.result)}\n\n`;
    });
  }
  
  agentContext.value = context;
};

const handleChatMessage = async (message: string) => {
  console.log('💬 Handling chat message:', message);
  
  // Add user message to chat
  chatMessages.value.push({
    role: 'user',
    content: message,
    timestamp: new Date().toLocaleTimeString()
  });
  
  // Add assistant message placeholder
  const assistantMessage = {
    role: 'agent',
    agent: 'assistant',
    content: '',
    timestamp: new Date().toLocaleTimeString(),
    isStreaming: true
  };
  chatMessages.value.push(assistantMessage);
  
  // Get reference to the message for updates
  const messageIndex = chatMessages.value.length - 1;
  
  try {
    // Prepare messages for the chat
    const messages = [];
    
    // Add context if available
    if (agentContext.value) {
      messages.push({
        role: 'system',
        content: `You are a helpful assistant. Use the following context from the pipeline execution to provide informed responses:\n\n${agentContext.value}`
      });
    }
    
    // Add previous chat messages
    chatMessages.value.forEach(msg => {
      if (msg.role === 'user' || (msg.role === 'agent' && msg.agent === 'assistant')) {
        messages.push({
          role: msg.role === 'agent' ? 'assistant' : msg.role,
          content: msg.content
        });
      }
    });
    
    // Make request to chat endpoint with all required fields
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages,
        model: selectedModel.value,
        stream: true,
        temperature: 0.7,
        max_tokens: 2048
      })
    });
    
    if (!response.body) throw new Error("Response body is null");
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        console.log('📝 Processing SSE line:', line);
        
        if (line.startsWith('event: ')) {
          const eventType = line.slice(7);
          console.log('🎯 Event type:', eventType);
        } else if (line.startsWith('data: ')) {
          const data = line.slice(6);
          console.log('📊 Event data:', data);
          
          if (data === '[DONE]') continue;
          
          try {
            const parsed = JSON.parse(data);
            console.log('🔍 Parsed data:', parsed);
            
            if (parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content) {
              const content = parsed.choices[0].delta.content;
              console.log('📝 Adding content:', content);
              // Update the message in the reactive array
              chatMessages.value[messageIndex].content += content;
            }
          } catch (e) {
            console.error('Failed to parse chat response:', e);
          }
        }
      }
    }
    
    chatMessages.value[messageIndex].isStreaming = false;
    
  } catch (error) {
    console.error('Chat error:', error);
    chatMessages.value[messageIndex].content = 'Sorry, I encountered an error. Please try again.';
    chatMessages.value[messageIndex].isStreaming = false;
  }
};

const runDataQuery = async () => {
    if (!finalGraphQLQuery.value.trim()) return;
    isDataLoading.value = true;
    dataQueryResults.value = [];
    try {
        const response = await fetch('/api/data/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: finalGraphQLQuery.value }),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch data');
        }
        const results = await response.json();
        dataQueryResults.value = results.data;
    } catch (error) {
        console.error('Data query error:', error);
        dataQueryResults.value = [{ error: String(error) }];
    } finally {
        isDataLoading.value = false;
    }
}

// Auth Handlers
const handleSignIn = () => { showAuthModal.value = true; };
const handleLogout = () => { authStore.logout(); router.push('/'); };
const onAuthenticated = (authData: any) => { authStore.setAuthenticated(authData); showAuthModal.value = false; };
const onGuestSession = (sessionData: any) => { authStore.setGuestSession(sessionData); showAuthModal.value = false; };

// Lifecycle
onMounted(() => {
  authStore.loadPersistedSession();
});

function formatReviewResult(result) {
  if (!result) return '';
  // Show comments and improvements as pretty JSON
  return `<pre>${JSON.stringify(result, null, 2)}</pre>`;
}
</script>

<style scoped>
.btn-primary {
  @apply bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed;
}
</style>