<template>
  <div class="flex-1 bg-gray-50 dark:bg-gray-900">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Advanced Query Translator
        </h1>
        <p class="text-lg text-gray-600 dark:text-gray-300">
          Real-time natural language to GraphQL translation with detailed agentic interactions
        </p>
      </div>
      
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Input Section -->
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Natural Language Query
            </h2>
          </div>
          <div class="card-body">
            <textarea
              v-model="query"
              class="textarea h-32"
              placeholder="Enter your natural language query here..."
            ></textarea>
            <div class="mt-4 space-y-2">
              <div class="flex items-center space-x-2">
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Model:</label>
                <select v-model="selectedModel" class="select">
                  <option value="phi3:mini">phi3:mini (Fast)</option>
                  <option value="llama2">llama2 (Balanced)</option>
                  <option value="llama2:13b">llama2:13b (High Quality)</option>
                </select>
              </div>
              <div class="flex items-center space-x-2">
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Temperature:</label>
                <input 
                  v-model.number="temperature" 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1" 
                  class="flex-1"
                >
                <span class="text-sm text-gray-600 dark:text-gray-400 w-8">{{ temperature }}</span>
              </div>
              <div class="flex justify-end space-x-2">
                <button
                  @click="clearAll"
                  class="btn-outline"
                >
                  Clear
                </button>
                <button
                  @click="translate"
                  :disabled="!query.trim() || loading"
                  class="btn-primary"
                >
                  <span v-if="loading" class="flex items-center">
                    <div class="spinner-sm mr-2"></div>
                    Translating...
                  </span>
                  <span v-else>Translate</span>
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Output Section -->
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              GraphQL Result
            </h2>
          </div>
          <div class="card-body">
            <div v-if="loading" class="flex items-center justify-center h-32">
              <div class="spinner"></div>
              <span class="ml-2 text-gray-600 dark:text-gray-300">Translating...</span>
            </div>
            <div v-else-if="result" class="h-32 overflow-auto">
              <pre class="text-sm text-gray-800 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 p-4 rounded-md">{{ result }}</pre>
            </div>
            <div v-else class="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
              Translation will appear here
            </div>
          </div>
        </div>

        <!-- Metrics Section -->
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Performance Metrics
            </h2>
          </div>
          <div class="card-body">
            <div v-if="metrics" class="space-y-3">
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
                <span class="text-sm font-medium" :class="confidenceColor">{{ (metrics.confidence * 100).toFixed(1) }}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Processing Time:</span>
                <span class="text-sm font-medium">{{ metrics.processing_time.toFixed(2) }}s</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Model Used:</span>
                <span class="text-sm font-medium">{{ metrics.model_used }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Tokens Used:</span>
                <span class="text-sm font-medium">{{ metrics.tokens_used || 'N/A' }}</span>
              </div>
              <div v-if="metrics.warnings && metrics.warnings.length > 0" class="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                <div class="text-sm font-medium text-yellow-800 dark:text-yellow-200">Warnings:</div>
                <ul class="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                  <li v-for="warning in metrics.warnings" :key="warning">• {{ warning }}</li>
                </ul>
              </div>
            </div>
            <div v-else class="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
              Metrics will appear here
            </div>
          </div>
        </div>
      </div>
      
      <!-- Agentic Interaction Log -->
      <div class="mt-8 card">
        <div class="card-header">
          <div class="flex justify-between items-center">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Agentic Interaction Log
            </h2>
            <div class="flex space-x-2">
              <button @click="clearLog" class="btn-outline-sm">Clear Log</button>
              <button @click="exportLog" class="btn-outline-sm">Export</button>
            </div>
          </div>
        </div>
        <div class="card-body">
          <div class="h-96 overflow-auto bg-gray-100 dark:bg-gray-800 rounded-md p-4">
            <div v-if="interactionLog.length === 0" class="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
              No interactions logged yet
            </div>
            <div v-else class="space-y-4">
              <div 
                v-for="(entry, index) in interactionLog" 
                :key="index"
                class="border-l-4 border-blue-500 pl-4 py-2"
              >
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sm font-medium text-gray-900 dark:text-white">
                    {{ entry.timestamp }}
                  </span>
                  <span class="text-xs px-2 py-1 rounded-full" :class="entryTypeClass(entry.type)">
                    {{ entry.type }}
                  </span>
                </div>
                <div class="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  {{ entry.message }}
                </div>
                <div v-if="entry.details" class="text-xs text-gray-600 dark:text-gray-400">
                  <pre class="whitespace-pre-wrap">{{ entry.details }}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Detailed Analysis -->
      <div class="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Prompt Analysis -->
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Prompt Analysis
            </h2>
          </div>
          <div class="card-body">
            <div v-if="promptAnalysis" class="space-y-3">
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">System Prompt:</label>
                <pre class="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded mt-1 overflow-auto max-h-32">{{ promptAnalysis.system_prompt }}</pre>
              </div>
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">User Prompt:</label>
                <pre class="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded mt-1 overflow-auto max-h-32">{{ promptAnalysis.user_prompt }}</pre>
              </div>
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Parameters:</label>
                <pre class="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded mt-1">{{ JSON.stringify(promptAnalysis.parameters, null, 2) }}</pre>
              </div>
            </div>
            <div v-else class="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
              Prompt analysis will appear here
            </div>
          </div>
        </div>

        <!-- Response Analysis -->
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Response Analysis
            </h2>
          </div>
          <div class="card-body">
            <div v-if="responseAnalysis" class="space-y-3">
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Raw Response:</label>
                <pre class="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded mt-1 overflow-auto max-h-32">{{ responseAnalysis.raw_response }}</pre>
              </div>
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Extracted GraphQL:</label>
                <pre class="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded mt-1 overflow-auto max-h-32">{{ responseAnalysis.extracted_graphql }}</pre>
              </div>
              <div>
                <label class="text-sm font-medium text-gray-700 dark:text-gray-300">Processing Steps:</label>
                <div class="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  <div v-for="(step, index) in responseAnalysis.processing_steps" :key="index" class="mb-1">
                    <span class="font-medium">{{ index + 1 }}.</span> {{ step }}
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="h-32 flex items-center justify-center text-gray-500 dark:text-gray-400">
              Response analysis will appear here
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

// Reactive state
const query = ref('')
const result = ref('')
const loading = ref(false)
const selectedModel = ref('phi3:mini')
const temperature = ref(0.7)

// Interaction logging
const interactionLog = ref<Array<{
  timestamp: string
  type: string
  message: string
  details?: string
}>>([])

// Analysis data
const metrics = ref<any>(null)
const promptAnalysis = ref<any>(null)
const responseAnalysis = ref<any>(null)

// Computed properties
const confidenceColor = computed(() => {
  if (!metrics.value) return ''
  const confidence = metrics.value.confidence
  if (confidence >= 0.8) return 'text-green-600 dark:text-green-400'
  if (confidence >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
})

// Helper functions
const addLogEntry = (type: string, message: string, details?: string) => {
  const timestamp = new Date().toLocaleTimeString()
  interactionLog.value.unshift({
    timestamp,
    type,
    message,
    details
  })
}

const entryTypeClass = (type: string) => {
  switch (type) {
    case 'INFO': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300'
    case 'WARNING': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300'
    case 'ERROR': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300'
    case 'SUCCESS': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300'
    default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }
}

const clearAll = () => {
  query.value = ''
  result.value = ''
  metrics.value = null
  promptAnalysis.value = null
  responseAnalysis.value = null
  addLogEntry('INFO', 'Cleared all data')
}

const clearLog = () => {
  interactionLog.value = []
}

const exportLog = () => {
  const logData = {
    timestamp: new Date().toISOString(),
    query: query.value,
    result: result.value,
    metrics: metrics.value,
    interactions: interactionLog.value
  }
  
  const blob = new Blob([JSON.stringify(logData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `translation-log-${new Date().toISOString().split('T')[0]}.json`
  a.click()
  URL.revokeObjectURL(url)
  
  addLogEntry('INFO', 'Exported interaction log')
}

// Main translation function
const translate = async () => {
  if (!query.value.trim()) return
  
  addLogEntry('INFO', `Starting translation for: "${query.value}"`)
  addLogEntry('INFO', `Using model: ${selectedModel.value}, temperature: ${temperature.value}`)
  
  loading.value = true
  result.value = ''
  metrics.value = null
  promptAnalysis.value = null
  responseAnalysis.value = null
  
  const startTime = Date.now()
  
  try {
    addLogEntry('INFO', 'Sending request to streaming translation API...')
    
    // Try streaming first, fallback to regular endpoint
    let useStreaming = true
    let finalData: any = null
    
    try {
      // Use the streaming endpoint
      const response = await fetch('http://localhost:8000/translate/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          natural_query: query.value,
          model: selectedModel.value,
          temperature: temperature.value,
          stream: true
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body reader available')
      }
      
      const decoder = new TextDecoder()
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6)
            if (dataStr === '[DONE]') {
              break
            }
            
            try {
              const data = JSON.parse(dataStr)
              
              // Handle different steps
              switch (data.step) {
                case 'init':
                  addLogEntry('INFO', data.message)
                  break
                case 'validate':
                  addLogEntry('INFO', data.message)
                  break
                case 'model':
                  addLogEntry('INFO', data.message)
                  break
                case 'prompt':
                  addLogEntry('INFO', data.message)
                  break
                case 'generate':
                  addLogEntry('INFO', data.message)
                  break
                case 'process':
                  addLogEntry('INFO', data.message)
                  break
                case 'extract':
                  addLogEntry('INFO', data.message)
                  break
                case 'complete':
                  addLogEntry('SUCCESS', data.message)
                  break
                case 'result':
                  finalData = data.data
                  addLogEntry('SUCCESS', 'Final result received')
                  break
                case 'error':
                  addLogEntry('ERROR', data.message)
                  throw new Error(data.message)
              }
            } catch (parseError) {
              console.warn('Failed to parse streaming data:', parseError)
            }
          }
        }
      }
      
    } catch (streamingError) {
      addLogEntry('WARNING', `Streaming failed: ${streamingError.message}, falling back to regular endpoint`)
      useStreaming = false
    }
    
    // Fallback to regular endpoint if streaming failed
    if (!useStreaming || !finalData) {
      addLogEntry('INFO', 'Using regular translation endpoint...')
      
      const response = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          natural_query: query.value,
          model: selectedModel.value,
          temperature: temperature.value
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      addLogEntry('SUCCESS', 'Translation completed via regular endpoint')
      
      // Convert regular response to streaming format
      finalData = {
        graphql_query: data.graphql_query,
        confidence: data.confidence,
        explanation: data.explanation,
        model: data.model,
        processing_time: data.processing_time,
        warnings: data.warnings || [],
        suggested_improvements: data.suggested_improvements || [],
        prompt_analysis: {
          system_prompt: "You are an expert GraphQL translator. Convert natural language queries to valid GraphQL syntax.",
          user_prompt: query.value,
          parameters: {
            model: selectedModel.value,
            temperature: temperature.value,
            max_tokens: 1000
          }
        },
        response_analysis: {
          raw_response: data.explanation || 'No explanation provided',
          extracted_graphql: data.graphql_query,
          processing_steps: [
            'Received natural language query',
            'Analyzed query intent and structure',
            'Generated GraphQL syntax',
            'Validated query structure',
            'Extracted confidence score',
            'Applied post-processing'
          ]
        }
      }
    }
    
    if (!finalData) {
      throw new Error('No final data received from translation response')
    }
    
    // Update metrics
    metrics.value = {
      confidence: finalData.confidence,
      processing_time: finalData.processing_time,
      model_used: finalData.model,
      tokens_used: finalData.tokens_used,
      warnings: finalData.warnings || []
    }
    
    // Update prompt analysis
    if (finalData.prompt_analysis) {
      promptAnalysis.value = finalData.prompt_analysis
    }
    
    // Update response analysis
    if (finalData.response_analysis) {
      responseAnalysis.value = finalData.response_analysis
    }
    
    // Format the result
    if (finalData.graphql_query && !finalData.graphql_query.startsWith('# Error:')) {
      result.value = `query {
  ${finalData.graphql_query}
}

// Confidence: ${(finalData.confidence * 100).toFixed(1)}%
// Model: ${finalData.model}
// Processing Time: ${finalData.processing_time.toFixed(2)}s
${finalData.explanation ? `// Explanation: ${finalData.explanation}` : ''}
${finalData.warnings && finalData.warnings.length > 0 ? `// Warnings: ${finalData.warnings.join(', ')}` : ''}
${finalData.suggested_improvements && finalData.suggested_improvements.length > 0 ? `// Suggestions: ${finalData.suggested_improvements.join(', ')}` : ''}`
      
      addLogEntry('SUCCESS', `Generated GraphQL query with ${(finalData.confidence * 100).toFixed(1)}% confidence`)
    } else {
      result.value = `${finalData.graphql_query || 'No GraphQL query generated'}

// Error Details:
${finalData.explanation ? `// ${finalData.explanation}` : ''}
${finalData.warnings && finalData.warnings.length > 0 ? `// Warnings: ${finalData.warnings.join(', ')}` : ''}
${finalData.suggested_improvements && finalData.suggested_improvements.length > 0 ? `// Suggestions: ${finalData.suggested_improvements.join(', ')}` : ''}
// Model: ${finalData.model}
// Processing Time: ${finalData.processing_time.toFixed(2)}s`
      
      addLogEntry('WARNING', 'Translation completed with errors or low confidence')
    }
    
    // Log detailed metrics
    addLogEntry('INFO', `Processing time: ${finalData.processing_time.toFixed(2)}s`, 
      `Model: ${finalData.model}\nConfidence: ${(finalData.confidence * 100).toFixed(1)}%\nTokens: ${finalData.tokens_used || 'N/A'}`)
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    addLogEntry('ERROR', `Translation failed: ${errorMessage}`)
    
    result.value = `Error: Failed to translate query
${errorMessage}`
    
    // Set error metrics
    metrics.value = {
      confidence: 0,
      processing_time: (Date.now() - startTime) / 1000,
      model_used: selectedModel.value,
      tokens_used: 0,
      warnings: [errorMessage]
    }
  } finally {
    loading.value = false
    addLogEntry('INFO', 'Translation process completed')
  }
}

// Component lifecycle
onMounted(() => {
  addLogEntry('INFO', 'Advanced Translator UI initialized')
  addLogEntry('INFO', 'Ready to process natural language queries')
})
</script>

<style scoped>
.spinner-sm {
  @apply animate-spin rounded-full h-4 w-4 border-2 border-gray-300 border-t-blue-600;
}

.btn-outline-sm {
  @apply px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors;
}
</style> 