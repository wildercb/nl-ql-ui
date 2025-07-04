<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800">
    <!-- Authentication Modal -->
    <AuthModal 
      :visible="showAuthModal" 
      @close="showAuthModal = false"
      @authenticated="handleAuthenticated"
      @guest-session="handleGuestSession"
    />

    <!-- Header -->
    <header class="bg-black bg-opacity-50 backdrop-blur-md border-b border-green-600 border-opacity-30">
      <div class="container mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-4">
            <div class="w-10 h-10 bg-gradient-to-r from-green-400 to-green-600 rounded-lg flex items-center justify-center">
              <i class="fas fa-brain text-white text-lg"></i>
            </div>
            <div>
              <h1 class="text-2xl font-bold text-white">NL - GraphQL UI</h1>
              <p class="text-green-400 text-sm">Natural Language to GraphQL Translation</p>
            </div>
          </div>
          
          <!-- User Info / Auth Button -->
          <div class="flex items-center space-x-4">
            <div v-if="currentUserRef" class="flex items-center space-x-3">
              <div class="text-right">
                <p class="text-white font-medium">{{ currentUserRef.username }}</p>
                <p class="text-green-400 text-xs">{{ currentUserRef.total_queries || 0 }} queries</p>
              </div>
              <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                <i class="fas fa-user text-white text-sm"></i>
              </div>
              <button 
                @click="logout"
                class="text-gray-400 hover:text-red-400 transition-colors duration-200"
                title="Logout"
              >
                <i class="fas fa-sign-out-alt"></i>
              </button>
            </div>
            <div v-else-if="isGuestSession" class="flex items-center space-x-3">
              <div class="text-right">
                <p class="text-gray-400 font-medium">Guest Session</p>
                <p class="text-green-400 text-xs">{{ sessionQueryCount }} queries</p>
              </div>
              <div class="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                <i class="fas fa-user-secret text-white text-sm"></i>
              </div>
              <button 
                @click="showAuthModal = true"
                class="text-green-400 hover:text-green-300 transition-colors duration-200"
                title="Sign In"
              >
                <i class="fas fa-sign-in-alt"></i>
              </button>
            </div>
            <div v-else>
              <button 
                @click="showAuthModal = true"
                class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-all duration-200 flex items-center space-x-2"
              >
                <i class="fas fa-sign-in-alt"></i>
                <span>Sign In</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>

    <div class="container mx-auto px-4 py-8">
      <div class="grid grid-cols-1 xl:grid-cols-3 gap-8">
        
        <!-- Left Panel - Translation Interface -->
        <div class="xl:col-span-2 space-y-6">
          
          <!-- Query Input Section -->
          <div class="bg-gradient-to-r from-gray-800 to-gray-900 rounded-2xl border border-green-600 border-opacity-30 shadow-2xl overflow-hidden">
            <div class="bg-gradient-to-r from-green-600 to-green-700 p-4">
              <h2 class="text-xl font-bold text-white flex items-center">
                <i class="fas fa-edit mr-3"></i>
                Natural Language Query
              </h2>
              <p class="text-green-100 text-sm mt-1">Describe what data you want in plain English</p>
            </div>
            
            <div class="p-6">
              <div class="space-y-4">
                <div>
                  <textarea
                    v-model="naturalQuery"
                    placeholder="e.g., Get all thermal scans from _ printer that were above 60 degrees celsius"
                    class="w-full h-32 bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200 resize-none"
                    :disabled="isLoading"
                  ></textarea>
                </div>
                
                <div class="flex items-center space-x-4 flex-wrap">
                  <div class="flex-1">
                    <select
                      v-model="selectedModel"
                      class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      :disabled="isLoading"
                    >
                      <option v-for="model in models" :key="model" :value="model">
                        {{ model }}
                      </option>
                    </select>
                  </div>
                  <button
                    @click="handleTranslate"
                    :disabled="isLoading || !naturalQuery.trim()"
                    :class="[
                      'px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2',
                      isLoading || !naturalQuery.trim()
                        ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                        : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    ]"
                  >
                    <i v-if="isLoading" class="fas fa-spinner fa-spin"></i>
                    <i v-else class="fas fa-magic"></i>
                    <span>{{ isLoading ? 'Translating...' : 'Translate' }}</span>
                  </button>

                  <!-- New Multi-Agent button -->
                  <button
                    @click="handleMultiAgent"
                    :disabled="isLoading || !naturalQuery.trim()"
                    :class="[
                      'px-8 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2',
                      isLoading || !naturalQuery.trim()
                        ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                        : 'bg-purple-600 hover:bg-purple-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    ]"
                    title="Run multi-agent pipeline"
                  >
                    <i v-if="isLoading" class="fas fa-spinner fa-spin"></i>
                    <i v-else class="fas fa-project-diagram"></i>
                    <span>{{ isLoading ? 'Processing...' : 'Multi-Agent' }}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Multi-Agent Responses -->
          <div ref="multiAgentChatRef">
            <ChatStream
              :messages="multiAgentMessages"
              :loading="sseController !== null"
              title="Multi-Agent Responses"
            />
          </div>

          <!-- Translation Results -->
          <div class="bg-gray-800 bg-opacity-50 rounded-lg p-4 h-full flex flex-col">
            <h2 class="text-xl font-bold text-green-400 mb-4">Translation Results</h2>
            <div v-if="translationResults.length === 0" class="text-gray-400 italic flex-grow">
              No translations yet. Enter a query and click Translate to begin.
            </div>
            <div v-else class="space-y-6 flex-grow overflow-y-auto custom-scrollbar pr-2">
              <div v-for="(interaction, idx) in translationResults" :key="interaction.timestamp">
                <!-- Existing interaction card -->
                <div class="bg-gray-700 bg-opacity-70 rounded-lg p-4 border border-gray-600 shadow-sm mb-2">
                  <div class="flex justify-between items-start mb-2">
                    <div class="text-green-400 font-semibold">Query: {{ interaction.query }}</div>
                    <div class="text-xs text-gray-400">{{ interaction.timestamp }}</div>
                  </div>
                  <div class="mb-2">
                    <div class="text-gray-200"><b>Response:</b> {{ interaction.response }}</div>
                    <div v-if="interaction.explanation" class="text-gray-400 text-xs mt-1">Explanation: {{ interaction.explanation }}</div>
                    <div v-if="interaction.warnings && interaction.warnings.length" class="text-yellow-400 text-xs mt-1">Warnings: {{ interaction.warnings.join(', ') }}</div>
                    <div v-if="interaction.suggestedImprovements && interaction.suggestedImprovements.length" class="text-blue-400 text-xs mt-1">Suggestions: {{ interaction.suggestedImprovements.join(', ') }}</div>
                  </div>
                  <div class="flex items-center justify-between">
                    <div class="text-xs text-gray-500">Model: {{ interaction.model }} | Confidence: {{ interaction.confidence }} | Time: {{ interaction.processingTime ? interaction.processingTime.toFixed(2) : 'N/A' }}s</div>
                    <button class="ml-2 px-3 py-1 bg-green-700 hover:bg-green-600 text-white rounded text-xs" @click="toggleChatForInteraction(interaction)">Chat</button>
                  </div>
                </div>
                <!-- In-page chatbox below the selected interaction -->
                <div v-if="showChat && selectedInteraction && selectedInteraction.timestamp === interaction.timestamp" class="bg-gray-800 rounded-lg p-4 mb-4 border-t border-gray-600">
                  <div class="flex items-center mb-3 border-b border-gray-700 pb-2">
                    <span class="text-green-400 font-semibold mr-2">💬 Chat with {{ chatModel }}</span>
                    <span class="text-xs text-gray-400 mr-auto">Continue the conversation about this query</span>
                    <button class="text-gray-400 hover:text-red-400 text-sm" @click="closeChatForInteraction">✕</button>
                  </div>
                  <div class="overflow-y-auto max-h-64 mb-3 border border-gray-700 rounded p-3 bg-gray-900 custom-scrollbar">
                    <div v-if="chatMessages.length === 0" class="text-gray-500 text-sm text-center py-4">
                      Start a conversation about this query...
                    </div>
                    <div v-for="(msg, i) in chatMessages" :key="i" class="mb-3 last:mb-0">
                      <div :class="msg.role === 'user' ? 'text-green-300' : 'text-blue-300'">
                        <span class="font-semibold">{{ msg.role === 'user' ? '👤 You' : '🤖 ' + chatModel }}</span>
                        <span class="text-xs text-gray-500 ml-2">{{ msg.timestamp }}</span>
                      </div>
                      <div class="text-gray-200 mt-1 pl-6">{{ msg.content }}</div>
                    </div>
                  </div>
                  <div class="flex items-center space-x-2">
                    <input 
                      v-model="chatInput" 
                      @keyup.enter="sendChatMessage" 
                      :disabled="isChatLoading" 
                      class="flex-1 rounded px-3 py-2 bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-green-500 border border-gray-600" 
                      placeholder="Ask about this query, request modifications, or get explanations..." 
                    />
                    <button 
                      @click="sendChatMessage" 
                      :disabled="isChatLoading || !chatInput.trim()" 
                      class="px-4 py-2 rounded transition-colors"
                      :class="[
                        isChatLoading || !chatInput.trim()
                          ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-500 text-white'
                      ]"
                    >
                      <i v-if="isChatLoading" class="fas fa-spinner fa-spin"></i>
                      <span v-else>Send</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Panel - Multi-Agent, History & Chat -->
        <div class="space-y-6">
          
          <!-- Unified Multi-Agent Conversation timeline (full-width) -->
          <div ref="multiAgentSection" v-if="false" class="xl:col-span-3 bg-gradient-to-r from-purple-800 to-purple-900 rounded-2xl border border-purple-600 border-opacity-30 shadow-2xl overflow-hidden mb-8">
            <div class="bg-gradient-to-r from-purple-600 to-purple-700 p-4">
              <h3 class="text-lg font-bold text-white flex items-center">
                <i class="fas fa-project-diagram mr-2"></i>
                Multi-Agent Conversations
                <span v-if="agentDemoResults.length" class="ml-2 bg-white bg-opacity-20 px-2 py-1 rounded-full text-xs">
                  {{ agentDemoResults.length }}
                </span>
              </h3>
            </div>
            <div class="max-h-[70vh] overflow-y-auto divide-y divide-gray-700">
              <div v-if="agentDemoResults.length === 0" class="p-6 text-center">
                <i class="fas fa-robot text-4xl text-purple-600 mb-4"></i>
                <p class="text-gray-400">Run the multi-agent pipeline to see conversations here.</p>
              </div>
              <div v-else>
                <div v-for="demo in agentDemoResults" :key="demo.timestamp" class="p-6 space-y-4">
                  <!-- Original Query -->
                  <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white">U</div>
                    <div class="flex-1">
                      <p class="font-medium text-gray-200 mb-1">Original Query</p>
                      <p class="text-gray-300 break-words whitespace-pre-wrap">{{ demo.originalQuery }}</p>
                    </div>
                  </div>

                  <!-- Rewritten Query -->
                  <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white">P</div>
                    <div class="flex-1">
                      <p class="font-medium text-gray-200 mb-1">Rewritten Query</p>
                      <p class="text-gray-300 break-words whitespace-pre-wrap">{{ demo.rewrittenQuery }}</p>
                    </div>
                  </div>

                  <!-- GraphQL -->
                  <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-yellow-600 rounded-full flex items-center justify-center text-white text-xs">GQL</div>
                    <div class="flex-1 overflow-x-auto">
                      <p class="font-medium text-gray-200 mb-1">GraphQL</p>
                      <pre class="bg-gray-900 text-green-400 p-3 rounded whitespace-pre overflow-x-auto text-xs"><code>{{ demo.graphqlQuery }}</code></pre>
                      <p class="text-gray-400 text-xs mt-1">Confidence: {{ (demo.confidence * 100).toFixed(1) }}%</p>
                    </div>
                  </div>

                  <!-- Reviewer -->
                  <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white">R</div>
                    <div class="flex-1">
                      <p class="font-medium text-gray-200 mb-1">Reviewer Feedback</p>
                      <p class="text-gray-300">Passed: <span :class="demo.reviewPassed ? 'text-green-400' : 'text-red-400'">{{ demo.reviewPassed ? 'Yes' : 'No' }}</span></p>
                      <div v-if="demo.reviewComments.length" class="text-yellow-400 text-xs mt-1"><b>Comments:</b> {{ demo.reviewComments.join(', ') }}</div>
                      <div v-if="demo.reviewSuggestions.length" class="text-blue-400 text-xs mt-1"><b>Suggestions:</b> {{ demo.reviewSuggestions.join(', ') }}</div>
                    </div>
                  </div>

                  <div class="text-xs text-gray-500">Processed in {{ demo.processingTime.toFixed(2) }}s</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Query History -->
          <div class="bg-gradient-to-r from-gray-800 to-gray-900 rounded-2xl border border-green-600 border-opacity-30 shadow-2xl overflow-hidden">
            <div class="bg-gradient-to-r from-green-600 to-green-700 p-4">
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-bold text-white flex items-center">
                  <i class="fas fa-history mr-2"></i>
                  Query History
                </h3>
                <div class="flex space-x-2">
                  <button
                    @click="loadHistory"
                    class="text-green-100 hover:text-white transition-colors duration-200"
                    title="Refresh"
                  >
                    <i class="fas fa-sync-alt"></i>
                  </button>
                  <button
                    v-if="queryHistory.length > 0"
                    @click="clearHistory"
                    class="text-green-100 hover:text-red-300 transition-colors duration-200"
                    title="Clear History"
                  >
                    <i class="fas fa-trash"></i>
                  </button>
                </div>
              </div>
            </div>
            
            <div class="max-h-96 overflow-y-auto">
              <div v-if="isLoadingHistory" class="p-6 text-center">
                <i class="fas fa-spinner fa-spin text-green-400 text-xl mb-2"></i>
                <p class="text-gray-400">Loading history...</p>
              </div>
              <div v-else-if="queryHistory.length === 0" class="p-6 text-center">
                <i class="fas fa-clock text-4xl text-gray-600 mb-4"></i>
                <p class="text-gray-400">No query history yet</p>
                <p class="text-gray-500 text-sm mt-2">{{ currentUserRef ? 'Your queries will appear here' : 'Sign in to save your query history' }}</p>
              </div>
              <div v-else class="divide-y divide-gray-700">
                <div
                  v-for="query in queryHistory"
                  :key="query.id"
                  @click="loadHistoryQuery(query)"
                  class="p-4 hover:bg-gray-700 cursor-pointer transition-colors duration-200"
                >
                  <div class="flex items-start justify-between">
                    <div class="flex-1 min-w-0">
                      <p class="text-white font-medium truncate">{{ query.natural_query }}</p>
                      <p class="text-gray-400 text-sm mt-1">{{ formatDate(query.timestamp) }}</p>
                    </div>
                    <div class="ml-2 flex items-center space-x-2">
                      <span :class="[
                        'w-2 h-2 rounded-full',
                        query.is_successful ? 'bg-green-400' : 'bg-red-400'
                      ]"></span>
                      <span class="text-xs text-gray-500">{{ (query.confidence * 100).toFixed(0) }}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Live Interactions -->
          <div class="bg-gradient-to-r from-gray-800 to-gray-900 rounded-2xl border border-green-600 border-opacity-30 shadow-2xl overflow-hidden">
            <div class="bg-gradient-to-r from-green-600 to-green-700 p-4">
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-bold text-white flex items-center">
                  <i class="fas fa-broadcast-tower mr-2"></i>
                  Live Model Interactions
                  <span v-if="liveInteractions.length > 0" class="ml-2 bg-white bg-opacity-20 px-2 py-1 rounded-full text-xs">
                    {{ liveInteractions.length }}
                  </span>
                </h3>
                <div class="flex space-x-2">
                  <button
                    @click="loadRecentInteractions"
                    class="text-green-100 hover:text-white transition-colors duration-200"
                    title="Refresh"
                  >
                    <i class="fas fa-sync-alt"></i>
                  </button>
                  <button
                    @click="simulateInteraction"
                    class="text-green-100 hover:text-white transition-colors duration-200"
                    title="Test Interaction"
                  >
                    <i class="fas fa-flask"></i>
                  </button>
                </div>
              </div>
            </div>
            
            <div class="max-h-96 overflow-y-auto">
              <div v-if="liveInteractions.length === 0" class="p-6 text-center">
                <i class="fas fa-satellite-dish text-4xl text-gray-600 mb-4"></i>
                <p class="text-gray-400">No live interactions</p>
                <p class="text-gray-500 text-sm mt-2">
                  Run a translation to see live model interactions here
                </p>
              </div>
              <div v-else class="divide-y divide-gray-700">
                <div
                  v-for="interaction in liveInteractions"
                  :key="interaction.id"
                  class="p-4 hover:bg-gray-700 transition-colors duration-200"
                >
                  <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                      <span class="text-green-400 font-medium">{{ interaction.model }}</span>
                      <span :class="[
                        'px-2 py-1 rounded-full text-xs',
                        interaction.status === 'completed' ? 'bg-green-600 text-white' : 'bg-yellow-600 text-white'
                      ]">
                        {{ interaction.status }}
                      </span>
                    </div>
                    <span class="text-gray-500 text-xs">{{ formatTime(new Date(interaction.timestamp * 1000)) }}</span>
                  </div>
                  <p class="text-gray-300 text-sm mb-2 line-clamp-2">
                    {{ interaction.prompt_data?.user_prompt || 'No prompt' }}
                  </p>
                  <div class="flex items-center justify-between">
                    <div class="flex space-x-4 text-xs text-gray-500">
                      <span>{{ interaction.processing_time?.toFixed(2) }}s</span>
                      <span>{{ ((interaction.response_data?.confidence || 0) * 100).toFixed(0) }}%</span>
                      <span>{{ interaction.metrics?.total_tokens || 0 }} tokens</span>
                    </div>
                    <button 
                      @click="toggleChatForInteraction(interaction)"
                      class="text-green-400 hover:text-green-300 text-sm"
                    >
                      <i class="fas fa-comment mr-1"></i>
                      Chat
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useAuthStore } from '../stores/auth'
import { useThemeStore } from '../stores/theme'
import AuthModal from '../components/AuthModal.vue'
import { useRouter } from 'vue-router'
import ChatStream from '../components/ChatStream.vue'

const authStore = useAuthStore()
const { isDark, toggleTheme } = useThemeStore()
const router = useRouter()

// Reactive state
const naturalQuery = ref('')
const selectedModel = ref('phi3:mini')
const isLoading = ref(false)
const activeTab = ref('results')
const showAuthModal = ref(false)
const liveInteractions = ref([])
const isConnected = ref(false)
const currentUserRef = ref<any>(null)
const showChat = ref(false)
const chatModel = ref('')
const chatMessages = ref<any[]>([])
const chatInput = ref('')
const isChatLoading = ref(false)
const interactionDetails = ref<any>(null)
const showInteractionDetails = ref(false)
const models = ref(['phi3:mini'])
const selectedInteraction = ref<any>(null)
const isLoadingHistory = ref(false)
const multiAgentSection = ref<HTMLElement | null>(null)
const multiAgentChatRef = ref<HTMLElement | null>(null)
const multiAgentMessages = ref<any[]>([])

// Translation results
const translationResults = ref([])
const agentDemoResults = ref([])

// User history and stats
const userHistory = ref([])
const userStats = reactive({
  totalQueries: 0,
  successfulQueries: 0,
  averageConfidence: 0,
  favoriteModel: ''
})

// SSE for live interactions
const eventSource = ref<EventSource | null>(null)

// Computed properties
const formattedConfidence = computed(() => 
  Math.round(translationResults.value[0]?.confidence * 100)
)

const confidenceColor = computed(() => {
  const conf = translationResults.value[0]?.confidence
  if (conf >= 0.8) return 'text-green-400'
  if (conf >= 0.6) return 'text-yellow-400'
  return 'text-red-400'
})

const isAuthenticated = computed(() => authStore.isAuthenticated)
const currentUser = computed(() => authStore.user)

// Query history that shows appropriate data based on authentication status
const queryHistory = computed(() => {
  if (authStore.isAuthenticated) {
    return userHistory.value
  } else {
    // For guests, show the translation results as history
    return translationResults.value.map((result, index) => ({
      id: `session-${index}`,
      natural_query: result.query,
      timestamp: result.timestamp,
      is_successful: !result.response.startsWith('Error:'),
      confidence: result.confidence,
      model_used: result.model
    }))
  }
})

// History stats for display
const historyStats = computed(() => {
  if (authStore.isAuthenticated) {
    return {
      total_queries: userStats.totalQueries,
      success_rate: userStats.totalQueries > 0 ? Math.round((userStats.successfulQueries / userStats.totalQueries) * 100) : 0
    }
  } else {
    const sessionQueries = translationResults.value
    const totalQueries = sessionQueries.length
    const successfulQueries = sessionQueries.filter(q => !q.response.startsWith('Error:')).length
    return {
      total_queries: totalQueries,
      success_rate: totalQueries > 0 ? Math.round((successfulQueries / totalQueries) * 100) : 0
    }
  }
})

// Methods
const handleTranslate = async () => {
  if (!naturalQuery.value.trim()) {
    console.log('Please enter a query to translate');
    return;
  }
  console.log('Translate button clicked', naturalQuery.value, selectedModel.value);

  isLoading.value = true;
  const startTime = Date.now();
  let result = {
    query: naturalQuery.value,
    response: 'Processing...',
    timestamp: new Date().toLocaleTimeString(),
    model: selectedModel.value || 'Unknown model',
    explanation: '',
    confidence: 0,
    warnings: [],
    suggestedImprovements: [],
    processingTime: 0
  };
  translationResults.value.unshift(result);

  try {
    console.log('Sending request to /api/translate with model:', selectedModel.value);
    const response = await fetch('/api/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.accessToken || authStore.guestSessionToken || ''}`
      },
      body: JSON.stringify({
        natural_query: naturalQuery.value,
        model: selectedModel.value
      })
    });
    console.log('Response received:', response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text();
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = { detail: errorText };
      }
      
      let reasonableError = 'Translation failed';
      if (errorData.detail) {
        if (typeof errorData.detail === 'string') {
          reasonableError = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          reasonableError = errorData.detail.map(err => err.msg || err).join(', ');
        }
      }
      
      // Add helpful context for common errors
      if (reasonableError.includes('not clear') || reasonableError.includes('unclear')) {
        reasonableError += '. Try being more specific about what data you want and from which entities/tables.';
      } else if (reasonableError.includes('timeout')) {
        reasonableError += '. The model took too long to respond. Try a simpler query.';
      } else if (reasonableError.includes('model')) {
        reasonableError += '. The AI model may be unavailable. Try again in a moment.';
      }
      
      throw new Error(reasonableError);
    }

    const data = await response.json();
    console.log('Translation data:', data);

    // Update the result with the actual response
    result.response = data.graphql_query || 'No GraphQL query was generated';
    result.explanation = data.explanation || 'The model processed your request but provided no explanation';
    result.confidence = data.confidence !== undefined ? data.confidence : 0;
    result.warnings = data.warnings || [];
    result.suggestedImprovements = data.suggested_improvements || [];
    result.processingTime = (Date.now() - startTime) / 1000;
    
    // If the response indicates the query wasn't clear, provide more context
    if (result.response.includes('not available') || result.response.includes('unclear')) {
      result.explanation = `The model couldn't understand your request clearly. ${result.explanation}. Try being more specific about: 1) What data you want, 2) From which entities/tables, 3) Any relationships between them.`;
    }
    
    // Refresh user history if authenticated
    if (authStore.isAuthenticated) {
      loadUserHistory();
    }
  } catch (error) {
    console.error('Error translating query:', error);
    console.log(`Failed to translate query: ${error.message}`);
    result.response = `Error: ${error.message}`;
    result.explanation = 'The translation request failed. Check your internet connection and try again.';
    result.processingTime = (Date.now() - startTime) / 1000;
  } finally {
    isLoading.value = false;
  }
};

// SSE controller for multi-agent streaming
const sseController = ref<EventSource | null>(null)

const handleMultiAgent = () => {
  if (!naturalQuery.value.trim()) return

  // Reset message list and add the user's query as the opening message
  multiAgentMessages.value = [{
    role: 'user',
    agent: 'User',
    content: naturalQuery.value,
    timestamp: new Date().toLocaleTimeString()
  }]

  // Close previous stream if running
  if (sseController.value) {
    sseController.value.close()
    sseController.value = null
  }

  const startTime = Date.now()
  const placeholder: any = {
    originalQuery: naturalQuery.value,
    rewrittenQuery: '',
    graphqlQuery: '',
    confidence: 0,
    reviewPassed: false,
    reviewComments: [],
    reviewSuggestions: [],
    processingTime: 0,
    timestamp: new Date().toLocaleTimeString(),
    model: selectedModel.value || 'default'
  }
  agentDemoResults.value.unshift(placeholder)

  // Build streaming URL
  const qs = new URLSearchParams({
    natural_query: naturalQuery.value,
    translator_model: selectedModel.value
  })
  const es = new EventSource(`/api/agent-demo/stream?${qs.toString()}`)
  sseController.value = es

  const applyPayload = (type: string, payload: any) => {
    const ts = new Date().toLocaleTimeString()
    if (type === 'rewrite') {
      placeholder.rewrittenQuery = payload.rewritten_query
      multiAgentMessages.value.push({ role: 'agent', agent: 'Rewriter', content: payload.rewritten_query, timestamp: ts })
    } else if (type === 'graphql') {
      placeholder.graphqlQuery = payload.graphql_query
      placeholder.confidence = payload.confidence
      multiAgentMessages.value.push({ role: 'agent', agent: 'Translator', content: payload.graphql_query, timestamp: ts })
    } else if (type === 'review') {
      placeholder.reviewPassed = payload.passed
      placeholder.reviewComments = payload.comments
      placeholder.reviewSuggestions = payload.suggestions
      placeholder.processingTime = (Date.now() - startTime) / 1000
      multiAgentMessages.value.push({
        role: 'agent',
        agent: 'Reviewer',
        content: `Passed: ${payload.passed ? 'Yes' : 'No'}\nComments: ${payload.comments.join(', ')}\nSuggestions: ${payload.suggestions.join(', ')}`,
        timestamp: ts
      })
    }
  }

  es.addEventListener('rewrite', (e: MessageEvent) => applyPayload('rewrite', JSON.parse(e.data)))
  es.addEventListener('graphql', (e: MessageEvent) => applyPayload('graphql', JSON.parse(e.data)))
  es.addEventListener('review', (e: MessageEvent) => applyPayload('review', JSON.parse(e.data)))
  es.addEventListener('complete', () => {
    placeholder.processingTime = (Date.now() - startTime) / 1000
    es.close()
    sseController.value = null
    multiAgentMessages.value.push({ role: 'agent', agent: 'System', content: 'Multi-agent processing complete.', timestamp: new Date().toLocaleTimeString() })
  })

  es.addEventListener('error', (e: MessageEvent) => {
    console.error('Multi-agent SSE error', e)
    placeholder.graphqlQuery = 'Error during streaming.'
    es.close()
    sseController.value = null
  })

  // Scroll to view
  nextTick(() => {
    multiAgentChatRef.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

const handleSignIn = () => {
  showAuthModal.value = true
}

const loadUserHistory = async () => {
  if (!authStore.isAuthenticated) return
  
  try {
    const response = await fetch('/api/history', {
      headers: {
        'Authorization': `Bearer ${authStore.accessToken}`
      }
    })
    
    if (response.ok) {
      const data = await response.json()
      userHistory.value = data.history || []
      
      // Update stats
      userStats.totalQueries = data.stats?.total_queries || 0
      userStats.successfulQueries = data.stats?.successful_queries || 0
      userStats.averageConfidence = data.stats?.average_confidence || 0
      userStats.favoriteModel = data.stats?.favorite_model || ''
    }
  } catch (error) {
    console.error('Failed to load user history:', error)
  }
}

const clearHistory = async () => {
  if (!authStore.isAuthenticated) return
  
  if (!confirm('Are you sure you want to clear your query history?')) return
  
  try {
    const response = await fetch('/api/history/clear', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authStore.accessToken}`
      }
    })
    
    if (response.ok) {
      userHistory.value = []
      Object.assign(userStats, {
        totalQueries: 0,
        successfulQueries: 0,
        averageConfidence: 0,
        favoriteModel: ''
      })
    }
  } catch (error) {
    console.error('Failed to clear history:', error)
  }
}

// Live interactions functionality
const setupLiveInteractions = () => {
  if (eventSource.value) {
    eventSource.value.close()
  }
  
  eventSource = new EventSource('/api/interactions/stream')
  
  eventSource.onmessage = (event) => {
    if (event.data) {
      try {
        const interaction = JSON.parse(event.data)
        if (interaction && interaction.id) {
          // Add to beginning of array (most recent first)
          liveInteractions.value.unshift(interaction)
          
          // Keep only the last 20 interactions
          if (liveInteractions.value.length > 20) {
            liveInteractions.value = liveInteractions.value.slice(0, 20)
          }
        }
      } catch (error) {
        console.error('Error parsing interaction data:', error)
      }
    }
  }
  
  eventSource.onerror = (error) => {
    console.error('EventSource error:', error)
    // Automatically reconnect after 5 seconds
    setTimeout(() => {
      if (eventSource.readyState === EventSource.CLOSED) {
        setupLiveInteractions()
      }
    }, 5000)
  }
}

const loadRecentInteractions = async () => {
  try {
    const response = await fetch('/api/interactions/recent?limit=10')
    if (response.ok) {
      const data = await response.json()
      liveInteractions.value = data.interactions || []
    }
  } catch (error) {
    console.error('Failed to load recent interactions:', error)
  }
}

const simulateInteraction = async () => {
  try {
    await fetch('/api/interactions/simulate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: selectedModel.value,
        prompt: naturalQuery.value || 'Test simulation query'
      })
    })
  } catch (error) {
    console.error('Failed to simulate interaction:', error)
  }
}

const openInteractionDetails = (interaction) => {
  selectedInteraction.value = interaction
}

const closeInteractionDetails = () => {
  selectedInteraction.value = null
}

const copyInteractionData = async () => {
  if (!selectedInteraction.value) return
  
  try {
    const data = JSON.stringify(selectedInteraction.value, null, 2)
    await navigator.clipboard.writeText(data)
    // Could add a toast notification here
  } catch (error) {
    console.error('Failed to copy interaction data:', error)
  }
}

// Chat functionality
const startChatWithInteraction = (interaction: any) => {
  selectedInteraction.value = interaction;
  chatMessages.value = [
    { role: 'user', content: interaction.query, timestamp: new Date().toLocaleString() },
    { role: 'assistant', content: interaction.response, timestamp: new Date().toLocaleString() }
  ];
  chatModel.value = interaction.model || selectedModel.value;
  showChat.value = true;
};

const closeChat = () => {
  showChat.value = false
  chatMessages.value = []
  chatInput.value = ''
  chatModel.value = ''
}

const sendChatMessage = async () => {
  if (!chatInput.value.trim()) {
    console.log('Please enter a message');
    return;
  }
  console.log('Sending chat message', chatInput.value, chatModel.value);

  isChatLoading.value = true;
  const newMessage = {
    role: 'user',
    content: chatInput.value,
    timestamp: new Date().toLocaleString()
  };
  chatMessages.value.push(newMessage);

  try {
    console.log('Sending request to /api/chat with model:', chatModel.value);
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.accessToken || authStore.guestSessionToken || ''}`
      },
      body: JSON.stringify({
        message: chatInput.value,
        model: chatModel.value,
        history: chatMessages.value.slice(0, -1) // Send all previous messages except the current one
      })
    });
    console.log('Response received:', response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorText}`);
    }

    const data = await response.json();
    console.log('Chat response data:', data);

    // Add the assistant's response to the chat
    chatMessages.value.push({
      role: 'assistant',
      content: data.response || 'Error: No response received',
      timestamp: new Date().toLocaleString()
    });
  } catch (error) {
    console.error('Error sending chat message:', error);
    console.log(`Failed to send chat message: ${error.message}`);
    chatMessages.value.push({
      role: 'assistant',
      content: `Error: ${error.message}`,
      timestamp: new Date().toLocaleString()
    });
  } finally {
    isChatLoading.value = false;
    chatInput.value = '';
  }
};

  // Utility functions
  const formatTime = (date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).format(date)
  }

  const formatDate = (dateString) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(dateString))
  }

  // Authentication event handling
  const handleAuthenticated = (authData: any) => {
    console.log('Authentication successful:', authData);
    authStore.setAuthenticated(authData);
    showAuthModal.value = false;
    loadUserHistory();
    currentUserRef.value = authStore.user;
  }

  const handleGuestSession = (sessionData: any) => {
    console.log('Guest session created:', sessionData);
    authStore.setGuestSession(sessionData);
    showAuthModal.value = false;
    currentUserRef.value = null;
  }

  const logout = () => {
    authStore.logout();
    currentUserRef.value = null;
    userHistory.value = [];
    Object.assign(userStats, {
      totalQueries: 0,
      successfulQueries: 0,
      averageConfidence: 0,
      favoriteModel: ''
    });
  }

  // Missing functions implementation
  const loadHistory = async () => {
    isLoadingHistory.value = true;
    try {
      if (authStore.isAuthenticated) {
        await loadUserHistory();
      }
      // For guests, the computed queryHistory will handle showing session data
    } finally {
      isLoadingHistory.value = false;
    }
  }

  const connectToStream = () => {
    setupLiveInteractions();
  }

  // Add missing computed properties
  const isGuestSession = computed(() => authStore.isGuestSession)
  const sessionQueryCount = computed(() => translationResults.value.length)

  const loadHistoryQuery = (query: any) => {
    // Load the query into the input field
    naturalQuery.value = query.natural_query;
    selectedModel.value = query.model_used || selectedModel.value;
    
    // Create a translation result object to display the historical result
    const historicalResult = {
      query: query.natural_query,
      response: query.graphql_query || 'No GraphQL query available',
      timestamp: new Date(query.timestamp).toLocaleTimeString(),
      model: query.model_used || 'Unknown model',
      explanation: `Historical query from ${new Date(query.timestamp).toLocaleDateString()}`,
      confidence: query.confidence || 0,
      warnings: [],
      suggestedImprovements: [],
      processingTime: query.processing_time || 0
    };
    
    // Add to translation results if not already there
    const existingIndex = translationResults.value.findIndex(r => 
      r.query === historicalResult.query && r.timestamp === historicalResult.timestamp
    );
    
    if (existingIndex === -1) {
      translationResults.value.unshift(historicalResult);
    }
    
    // Switch to results tab to show the loaded query
    activeTab.value = 'results';
    
    console.log('Loaded historical query:', query.natural_query);
  }

  // Lifecycle hooks
  onMounted(() => {
    authStore.loadPersistedSession();
    loadHistory();
    loadRecentInteractions();
    connectToStream();
    currentUserRef.value = authStore.user;
  })

  onUnmounted(() => {
    if (eventSource.value) {
      eventSource.value.close()
    }
  })

  // New methods for chat functionality
  const toggleChatForInteraction = (interaction) => {
    console.log('Opening chat for interaction:', interaction);
    
    // Set the selected interaction and model
    selectedInteraction.value = interaction;
    chatModel.value = interaction.model || selectedModel.value;
    
    // Initialize chat messages with the query and response
    const initialMessages = [];
    
    // For translation results
    if (interaction.query && interaction.response) {
      initialMessages.push({
        role: 'user',
        content: interaction.query,
        timestamp: new Date().toLocaleString()
      });
      
      initialMessages.push({
        role: 'assistant', 
        content: interaction.response,
        timestamp: new Date().toLocaleString()
      });
    }
    // For live interactions (different structure)
    else if (interaction.prompt_data && interaction.response_data) {
      initialMessages.push({
        role: 'user',
        content: interaction.prompt_data.user_prompt || 'Query not available',
        timestamp: new Date(interaction.timestamp * 1000).toLocaleString()
      });
      
      initialMessages.push({
        role: 'assistant',
        content: interaction.response_data.raw_response || interaction.response_data.processed_response || 'Response not available',
        timestamp: new Date(interaction.timestamp * 1000).toLocaleString()
      });
    }
    
    chatMessages.value = initialMessages;
    showChat.value = true;
    
    console.log('Chat opened with', initialMessages.length, 'initial messages');
  }

  const closeChatForInteraction = () => {
    showChat.value = false;
    chatMessages.value = [];
    chatInput.value = '';
    chatModel.value = '';
    selectedInteraction.value = null;
  }

  const handleChatMessage = async () => {
    if (!chatInput.value.trim()) return;
    console.log('Sending chat message:', chatInput.value);
    
    const newMessage = {
      sender: 'user',
      content: chatInput.value,
      timestamp: new Date().toISOString()
    };
    currentChatMessages.value.push(newMessage);
    
    // Add to chat history for context
    chatHistory.value.push({
      role: 'user',
      content: chatInput.value
    });
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authStore.accessToken}`
        },
        body: JSON.stringify({
          message: chatInput.value,
          model: selectedModel.value,
          history: chatHistory.value
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Chat response:', data);
      
      const aiMessage = {
        sender: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString()
      };
      currentChatMessages.value.push(aiMessage);
      
      // Add AI response to history
      chatHistory.value.push({
        role: 'assistant',
        content: data.response
      });
      
      // Keep only last 10 messages in history to prevent token overload
      if (chatHistory.value.length > 10) {
        chatHistory.value = chatHistory.value.slice(-10);
      }
    } catch (error) {
      console.error('Error sending chat message:', error);
      currentChatMessages.value.push({
        sender: 'assistant',
        content: 'Error: Could not send message. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      });
    } finally {
      chatInput.value = '';
    }
  };
</script>

<style scoped>
/* Custom scrollbar for better appearance */
.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: #374151;
  border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: #10b981;
  border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background: #059669;
}

/* Animation for new interactions */
@keyframes slideInDown {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.live-interaction {
  animation: slideInDown 0.3s ease-out;
}
</style>