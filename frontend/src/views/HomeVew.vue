const continuationInput = ref('')
const activePipeline = ref<string | null>(null)
const showDebug = ref(false)

// SSE for live interactions - DEPRECATED
// let eventSource = ref<EventSource | null>(null) // This is now handled by the fetchStream function

// Computed properties
const formattedConfidence = computed(() => {
// ... existing code ...
  }
  return '—'
})

const finalGraphQLQuery = computed(() => {
  const completeMessage = multiAgentMessages.value.find(m => m.event === 'complete')
  if (completeMessage && completeMessage.data) {
    return completeMessage.data.result?.translation?.graphql_query || ''
  }
  return ''
})

const finalResultData = computed(() => {
  const completeMessage = multiAgentMessages.value.find(m => m.event === 'complete')
  if (completeMessage && completeMessage.data) {
    return completeMessage.data.result
  }
  return null
})

const isPipelineComplete = computed(() => {
  return multiAgentMessages.value.some(m => m.event === 'complete' || m.event === 'error')
})

// Methods

const handleTranslate = async () => {
  if (!naturalQuery.value.trim()) return

  isLoading.value = true
  translationResults.value = []
  activePipeline.value = 'translate'

  try {
    const response = await fetch(`/api/translate?natural_query=${encodeURIComponent(naturalQuery.value)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })
    if (!response.ok) throw new Error('Network response was not ok')
    const data = await response.json()
    translationResults.value = data.results
  } catch (error) {
    console.error('Translation error:', error)
    // Handle error display
  } finally {
    isLoading.value = false
  }
}

const handleMultiAgent = async () => {
  if (!naturalQuery.value.trim()) return

  clearAllStreams()
  isLoading.value = true
  activePipeline.value = 'multi_agent_legacy'

  const startMessage = {
    role: 'system',
    content: '🚀 Starting Legacy Multi-Agent Pipeline...',
    timestamp: new Date().toLocaleTimeString(),
  }
  multiAgentMessages.value = [startMessage]

  const url = `/api/agent-demo/stream?natural_query=${encodeURIComponent(naturalQuery.value)}`
  const eventSource = new EventSource(url)

  eventSource.addEventListener('rewrite', (event) => {
    const data = JSON.parse(event.data)
    multiAgentMessages.value.push({
      role: 'assistant',
      agent: 'Rewriter',
      content: `Query rewritten: ${data.rewritten_query}`,
      timestamp: new Date().toLocaleTimeString()
    })
  })

  eventSource.addEventListener('graphql', (event) => {
    const data = JSON.parse(event.data)
    multiAgentMessages.value.push({
      role: 'assistant',
      agent: 'Translator',
      content: `GraphQL generated with confidence: ${data.confidence.toFixed(2)}`,
      data: data,
      timestamp: new Date().toLocaleTimeString()
    })
  })

  eventSource.addEventListener('review', (event) => {
    const data = JSON.parse(event.data)
    multiAgentMessages.value.push({
      role: 'assistant',
      agent: 'Reviewer',
      content: `Review complete. Passed: ${data.passed}`,
      data: data,
      timestamp: new Date().toLocaleTimeString()
    })
  })

  eventSource.addEventListener('complete', () => {
    isLoading.value = false
    eventSource.close()
  })

  eventSource.addEventListener('error', (error) => {
    console.error('SSE Error:', error)
    multiAgentMessages.value.push({
      role: 'system',
      content: 'An error occurred with the live connection.',
      timestamp: new Date().toLocaleTimeString()
    })
    isLoading.value = false
    eventSource.close()
  })
}

const fetchStream = async (url: string, options: RequestInit) => {
  try {
    const response = await fetch(url, options);
    if (!response.body) {
      throw new Error("Response has no body");
    }

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += value;
      let boundary = buffer.indexOf('\\n\\n');

      while (boundary !== -1) {
        const chunk = buffer.substring(0, boundary);
        buffer = buffer.substring(boundary + 2);

        let event = 'message';
        let data = '';

        const eventMatch = chunk.match(/^event: (.*)$/m);
        if (eventMatch) {
          event = eventMatch[1];
        }

        const dataMatch = chunk.match(/^data: (.*)$/m);
        if (dataMatch) {
          data = dataMatch[1];
        }
        
        if (event !== 'ping') {
            const customEvent = new CustomEvent(event, { detail: data });
            window.dispatchEvent(customEvent);
        }

        boundary = buffer.indexOf('\\n\\n');
      }
    }
  } catch (error) {
    console.error("Streaming fetch failed:", error);
    const customEvent = new CustomEvent('error', { detail: JSON.stringify({ message: 'Connection failed' }) });
    window.dispatchEvent(customEvent);
  }
};


const handleEnhancedAgent = async (strategy: string) => {
  console.log(`Enhanced Agent button clicked with strategy: ${strategy}`);
  if (!naturalQuery.value.trim()) return;

  clearAllStreams();
  isLoading.value = true;
  showEnhancedDropdown.value = false;
  activePipeline.value = 'enhanced';

  const startMessage = {
    role: 'system',
    event: 'system_start',
    content: `🚀 Starting Enhanced Agent Pipeline with ${strategy.toUpperCase()} strategy`,
    timestamp: new Date().toLocaleTimeString(),
  };
  multiAgentMessages.value = [startMessage];

  // Define event listeners
  const handleSessionStart = (e: CustomEvent) => {
    const data = JSON.parse(e.detail);
    console.log('Session started:', data);
  };

  const handleAgentStart = (e: CustomEvent) => {
    const data = JSON.parse(e.detail);
    multiAgentMessages.value.push({
      role: 'assistant',
      agent: data.agent,
      event: 'agent_start',
      content: `Agent [${data.agent}] started (Step ${data.step}/${data.total_steps})`,
      timestamp: new Date().toLocaleTimeString(),
      data: data,
    });
  };

  const handleAgentComplete = (e: CustomEvent) => {
    const data = JSON.parse(e.detail);
    const existingMessageIndex = multiAgentMessages.value.findIndex(
      m => m.agent === data.agent && m.event === 'agent_start'
    );
    if (existingMessageIndex !== -1) {
      multiAgentMessages.value[existingMessageIndex] = {
        ...multiAgentMessages.value[existingMessageIndex],
        event: 'agent_complete',
        content: `Agent [${data.agent}] completed.`,
        data: data, // includes prompt
      };
    }
  };

  const handleComplete = (e: CustomEvent) => {
    const data = JSON.parse(e.detail);
    multiAgentMessages.value.push({
      role: 'system',
      event: 'complete',
      content: `✅ Pipeline Finished.`,
      timestamp: new Date().toLocaleTimeString(),
      data: data
    });
    cleanup();
  };

  const handleError = (e: CustomEvent) => {
    const data = JSON.parse(e.detail);
    multiAgentMessages.value.push({
      role: 'system',
      event: 'error',
      content: `❌ Pipeline Error: ${data.error}`,
      timestamp: new Date().toLocaleTimeString(),
      data: data
    });
    cleanup();
  };

  const cleanup = () => {
    isLoading.value = false;
    window.removeEventListener('session_start', handleSessionStart as EventListener);
    window.removeEventListener('agent_start', handleAgentStart as EventListener);
    window.removeEventListener('agent_complete', handleAgentComplete as EventListener);
    window.removeEventListener('complete', handleComplete as EventListener);
    window.removeEventListener('error', handleError as EventListener);
  };
  
  // Add event listeners
  window.addEventListener('session_start', handleSessionStart as EventListener);
  window.addEventListener('agent_start', handleAgentStart as EventListener);
  window.addEventListener('agent_complete', handleAgentComplete as EventListener);
  window.addEventListener('complete', handleComplete as EventListener);
  window.addEventListener('error', handleError as EventListener);

  const authStore = useAuthStore()
  const sessionId = authStore.isAuthenticated ? authStore.accessToken : authStore.guestSessionId;

  fetchStream('/api/multiagent/process/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: naturalQuery.value,
      pipeline_strategy: strategy,
      user_id: authStore.user?.id,
      session_id: sessionId,
    }),
  });
};

const handleContinuation = async () => {
// ... existing code ...
  }
};

const closeEventSource = () => {
  // This function is now effectively deprecated by the fetchStream architecture
  // but we keep it to avoid breaking anything that might still reference it.
  isLoading.value = false;
};

const clearAllStreams = () => {
  // No active connection to close in fetch-based approach
  // We just reset the state
  multiAgentMessages.value = [];
  streamingComplete.value = false;
  activePipeline.value = null;
  isLoading.value = false;
};

const handleSignIn = () => {
// ... existing code ...
// ... existing code ...
        <div 
          v-if="activePipeline === 'enhanced' || activePipeline === 'multi_agent_legacy'" 
          class="flex-grow overflow-y-auto p-4 space-y-4"
        >
          <div v-for="(msg, index) in multiAgentMessages" :key="index" class="chat-message">
            <div v-if="msg.role === 'system'" class="text-center text-sm text-gray-500 my-2">
              --- {{ msg.content }} ---
            </div>
            <div v-else class="flex items-start space-x-3">
              <div class="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center">
                A
              </div>
              <div class="flex-grow">
                <div class="font-bold text-sm">{{ msg.agent || 'Agent' }}</div>
                <div>{{ msg.content }}</div>
                <div v-if="msg.event === 'agent_complete' && msg.data.prompt" class="mt-2">
                    <button @click="msg.showPrompt = !msg.showPrompt" class="text-xs text-blue-500 hover:underline">
                        {{ msg.showPrompt ? 'Hide' : 'Show' }} Prompt
                    </button>
                    <div v-if="msg.showPrompt" class="mt-1 p-2 bg-gray-800 text-white rounded-md text-xs overflow-x-auto">
                        <pre><code>{{ JSON.stringify(msg.data.prompt, null, 2) }}</code></pre>
                    </div>
                </div>
                <div class="text-xs text-gray-400 mt-1">{{ msg.timestamp }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Final Result Display -->
        <div v-if="isPipelineComplete && finalResultData" class="p-4 border-t border-gray-700">
          <h3 class="text-lg font-semibold mb-2">Final Result</h3>
          <div class="bg-gray-800 p-3 rounded-lg text-sm">
            <p><strong>Strategy:</strong> {{ finalResultData.pipeline_strategy }}</p>
            <p><strong>Processing Time:</strong> {{ finalResultData.processing_time.toFixed(2) }}s</p>
            <p><strong>Review Passed:</strong> {{ finalResultData.review.passed ? '✅' : '❌' }}</p>
            <div v-if="finalGraphQLQuery" class="mt-2">
              <h4 class="font-semibold">GraphQL Query:</h4>
              <pre class="bg-black text-white p-2 rounded mt-1 overflow-x-auto"><code>{{ finalGraphQLQuery }}</code></pre>
            </div>
          </div>
        </div>

        <!-- Chat Continuation Input -->
        <div v-if="isPipelineComplete" class="p-4 border-t border-gray-700">
          <div class="relative">
            <input
// ... existing code ...

</rewritten_file> 