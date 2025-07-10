<template>
  <div class="flex flex-col h-screen bg-gray-900 text-white">
    <div class="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 overflow-hidden">
      <!-- Left Column: NL Query Input + Agent Stream -->
      <div class="flex flex-col h-full min-h-0 border border-gray-700 rounded-none p-4">
        <!-- Header row with title + controls -->
        <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
          <h2 class="text-lg font-semibold text-gray-200">Enter Natural Language Query</h2>

          <!-- Controls (model, pipeline, send) -->
          <div class="flex flex-wrap items-center gap-2">
            <!-- Model selection (compact, uniform height) -->
            <select v-model="selectedModel" class="h-8 px-3 text-xs bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500" :disabled="isProcessing">
              <option value="phi3:mini">phi3:mini (Ollama)</option>
              <option value="gemma3:4b">gemma3:4b (Ollama)</option>
              <option value="gemma3n:e2b">gemma3n:e2b (Ollama)</option>
              <option value="gemma3n:e4b">gemma3n:e4b (Ollama)</option>
              <option value="gemma:7b">gemma:7b (Ollama)</option>
              <option value="llama3:8b">llama3:8b (Ollama)</option>
              <!-- Groq models -->
              <option value="groq::llama3-8b-8192">Groq – Llama3-8B-8192</option>
              <option value="groq::llama3-70b-8192">Groq – Llama3-70B-8192</option>
              <option value="groq::mixtral-8x7b-32768">Groq – Mixtral-8x7B-32K</option>
              <!-- OpenRouter models -->
              <option value="openrouter::meta-llama/llama-3-8b-instruct">OpenRouter – Llama3-8B-Instruct</option>
              <option value="openrouter::mistralai/mixtral-8x7b">OpenRouter – Mixtral-8x7B</option>
              <option value="openrouter::google/gemma-7b-it">OpenRouter – Gemma-7B-IT</option>
              <option value="openrouter::thudm/chatglm3-6b-32k">OpenRouter – ChatGLM3-6B-32K</option>
            </select>

            <!-- Pipeline strategy dropdown (same height) -->
            <select
              v-model="selectedPipeline"
              :disabled="isProcessing"
              class="h-8 px-3 text-xs bg-primary-600 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 cursor-pointer"
            >
              <option v-for="opt in pipelineOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>

            <!-- Send button (compact with icon) -->
            <button
              @click="runPipeline(selectedPipeline)"
              :disabled="isProcessing || !naturalQuery.trim()"
              class="flex items-center justify-center gap-1 h-8 px-3 text-xs bg-primary-600 hover:bg-primary-700 text-white rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <i class="fas fa-paper-plane"></i>
            </button>
          </div>
        </div>

        <!-- NL Query Input (fixed height) -->
        <textarea
          v-model="naturalQuery"
          @keyup.enter="runPipeline(selectedPipeline)"
          :disabled="isProcessing"
          class="w-full resize-none p-3 bg-gray-700 rounded-none border border-gray-600 focus:outline-none focus:ring-0 focus:border-green-600 text-base min-h-[90px] max-h-[200px]"
          placeholder="Enter your natural language query here..."
        ></textarea>

        <!-- Agent Stream Chat (fills remaining space, scrollable) -->
        <div class="flex-1 min-h-0">
          <ChatStream 
            :messages="chatMessages" 
            :loading="isProcessing" 
            title="Agent Stream" 
            :selectedModel="selectedModel"
            :prompts="agentPrompts"
            :show-prompts="showPrompts"
            @toggle-prompts="showPrompts = !showPrompts"
            @sendMessage="handleChatMessage"
          />
        </div>
      </div>

      <!-- Right Column: GraphQL Query + Results -->
      <div class="flex flex-col h-full min-h-0 space-y-0 border border-gray-700 rounded-none p-4">
        <!-- Header with Download button -->
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-lg font-semibold text-gray-200">Get Results Here</h2>
          <!-- Small green download button -->
          <button
            @click="downloadResults"
            :disabled="!dataQueryResults.length"
            class="bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold px-3 py-1 rounded-md flex items-center"
          >
            <i class="fas fa-download mr-1"></i> Download
          </button>
        </div>
        <!-- GraphQL Query Box -->
        <div class="flex-none">
          <GraphQLQueryBox :query="finalGraphQLQuery" @update:query="finalGraphQLQuery = $event" @send="runDataQuery" />
        </div>
        <!-- Results -->
        <div class="flex-1 flex flex-col min-h-0">
          <DataResults :results="dataQueryResults" :loading="isDataLoading" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';
import { useHistoryStore } from '../stores/history';
import ChatStream from '../components/ChatStream.vue';
import GraphQLQueryBox from '../components/GraphQLQueryBox.vue';
import DataResults from '../components/DataResults.vue';
import { mcpClient, type MCPQueryRequest } from '../services/mcpClient';

const authStore = useAuthStore();
const historyStore = useHistoryStore();
const router = useRouter();

// Model & pipeline selections
const selectedModel = ref('gemma3:4b');
const selectedPipeline = ref('comprehensive');

const pipelineOptions = [
  { label: 'MCP Fast', value: 'fast', description: 'Translation only - fastest processing' },
  { label: 'MCP Standard', value: 'standard', description: 'Rewrite → Translate → Review pipeline' },
  { label: 'MCP Comprehensive', value: 'comprehensive', description: 'All agents + optimization + data review' },
  { label: 'MCP Adaptive', value: 'adaptive', description: 'Strategy auto-selected by query complexity' },
] as const;

// Pipeline state
const naturalQuery = ref('');
const chatMessages = ref<any[]>([]);
const isProcessing = ref(false);

const finalGraphQLQuery = ref('');
const isDataLoading = ref(false);
const dataQueryResults = ref<any[]>([]);

// Agent prompts and context
const agentPrompts = ref<any[]>([]);
const agentContext = ref<string>('');
const showPrompts = ref(false);

const isValidGraphQL = (q: string) => {
  return /\{[\s\S]*\}/.test(q);
};

const sanitizeGraphQL = (q: string) => {
  if (!q) return '';
  q = q.replace(/```graphql|```/g, '').trim();
  try {
    const obj = JSON.parse(q);
    if (typeof obj === 'string') return obj;
    if (obj.graphql) return obj.graphql;
    if (obj.query) return obj.query;
    if (obj.suggested_query) return obj.suggested_query;
  } catch {}
  const match = q.match(/(query|mutation)?[\s\S]*\{[\s\S]*\}/);
  return match ? match[0] : '';
};

const runPipeline = async (strategy: string) => {
  if (!naturalQuery.value.trim() || isProcessing.value) return;

  console.log('🚀 Starting MCP pipeline with strategy:', strategy, 'using model:', selectedModel.value);
  isProcessing.value = true;
  chatMessages.value = [];
  finalGraphQLQuery.value = '';
  dataQueryResults.value = [];
  agentPrompts.value = [];

  chatMessages.value.push({ role: 'user', content: naturalQuery.value, timestamp: new Date().toLocaleTimeString() });

  // Placeholder chat message that we will update as events stream in
  const streamingMsg = {
    role: 'agent',
    agent: 'mcp_pipeline',
    content: '⏳ Starting pipeline...',
    timestamp: new Date().toLocaleTimeString(),
    isStreaming: true
  };
  chatMessages.value.push(streamingMsg);

  try {
    console.log('📡 Streaming request to MCP server...');

    // Build MCP request
    const mcpRequest: MCPQueryRequest = {
      query: naturalQuery.value,
      pipeline_strategy: strategy as any,
      translator_model: selectedModel.value,
      user_id: 'frontend_user'
    };

    console.log('📤 MCP Request:', mcpRequest);

    let finalResult: any = null;
    let eventCount = 0;

    // No longer tracking message indices since we removed processing messages

    for await (const evt of mcpClient.processQueryStream(mcpRequest)) {
      eventCount++;
      console.log(`🌊 MCP event #${eventCount}:`, evt);
      console.log(`📊 Event type: ${evt.event}`);
      console.log(`📊 Event data:`, JSON.stringify(evt.data, null, 2));

      switch (evt.event) {
        case 'start':
          streamingMsg.content = '⏳ Pipeline started...';
          break;

        case 'agent_start': {
          const agent = evt.data?.data?.agent || evt.data?.agent || 'agent';
          console.log(`🚀 Agent started: ${agent}`);
          // Don't create any processing message - just wait for the response
          break;
        }

        case 'agent_prompt': {
          const agent = evt.data?.data?.agent || evt.data?.agent || 'agent';
          const prompt = evt.data?.data?.prompt || evt.data?.prompt || '';
          console.log(`📝 Agent prompt for ${agent}:`, prompt);
          
          // Store the prompt for Show Prompts toggle (only)
          agentPrompts.value.push({
            agent,
            prompt,
            result: null,
            timestamp: new Date().toLocaleTimeString()
          });
          
          // Add the prompt to the chat stream as plain text
          chatMessages.value.push({
            role: 'agent',
            agent: agent,
            content: `PROMPT:\n${prompt}`,
            timestamp: new Date().toLocaleTimeString(),
            isStreaming: false
          });
          
          break;
        }

        case 'agent_token': {
          const agent = evt.data?.data?.agent || evt.data?.agent || 'agent';
          const token = evt.data?.data?.token || evt.data?.token || '';
          console.log(`🎯 Agent token for ${agent}:`, token);
          
          // Don't show tokens in the chat stream - just log them
          break;
        }

        case 'agent_complete': {
          const agent = evt.data?.data?.agent || evt.data?.agent || 'agent';
          const result = evt.data?.data?.result || evt.data?.result || {};
          const prompt = evt.data?.data?.prompt || evt.data?.prompt;
          const rawOutput = evt.data?.data?.raw_output || evt.data?.raw_output;
          const success = evt.data?.data?.success || evt.data?.success;
          const error = evt.data?.data?.error || evt.data?.error;

          console.log(`✅ Agent completed: ${agent}`, evt.data?.data?.result || evt.data?.result);
          console.log(`📝 Agent prompt:`, evt.data?.data?.prompt || evt.data?.prompt);
          console.log(`📊 Agent result data:`, JSON.stringify(evt.data?.data?.result || evt.data?.result, null, 2));
          console.log(`🔍 Raw output:`, evt.data?.data?.raw_output || evt.data?.raw_output);
          console.log(`✅ Success:`, evt.data?.data?.success || evt.data?.success);
          console.log(`❌ Error:`, evt.data?.data?.error || evt.data?.error);
          console.log(`🎯 Full event data:`, JSON.stringify(evt.data, null, 2));

          // Show the response in the chat stream (same as console)
          let responseContent = '';
          
          // Use the exact same data that was logged to console
          const consoleResult = evt.data?.data?.result || evt.data?.result;
          const consoleRawOutput = evt.data?.data?.raw_output || evt.data?.raw_output;
          const consoleSuccess = evt.data?.data?.success || evt.data?.success;
          const consoleError = evt.data?.data?.error || evt.data?.error;
          
          if (consoleError) {
            responseContent = `${agent.toUpperCase()} ERROR:\n\nError: ${consoleError}\n\nRaw Output: ${consoleRawOutput || 'No raw output'}\n\nResult: ${JSON.stringify(consoleResult, null, 2)}`;
          } else {
            // Format the response as plain text
            if (consoleRawOutput && typeof consoleRawOutput === 'object') {
              responseContent = `${agent.toUpperCase()} RESPONSE:\n\n${JSON.stringify(consoleRawOutput, null, 2)}`;
            } else if (consoleRawOutput) {
              responseContent = `${agent.toUpperCase()} RESPONSE:\n\n${consoleRawOutput}`;
            } else if (consoleResult) {
              responseContent = `${agent.toUpperCase()} RESPONSE:\n\n${JSON.stringify(consoleResult, null, 2)}`;
            } else {
              responseContent = `${agent.toUpperCase()} RESPONSE:\n\nNo response data available`;
            }
          }

          // Add the response to the chat stream immediately after the prompt
          chatMessages.value.push({
            role: 'agent',
            agent: agent,
            content: responseContent,
            timestamp: new Date().toLocaleTimeString(),
            isStreaming: false
          });

          // Update the agent prompts with the result
          const existingPromptIndex = agentPrompts.value.findIndex(p => p.agent === agent && p.result === null);
          if (existingPromptIndex !== -1) {
            agentPrompts.value[existingPromptIndex].result = evt.data?.data?.result || evt.data?.result || {};
          }

          // Handle translator's GraphQL query
          if (agent === 'translator') {
            const graphql = (evt.data?.data?.result?.graphql_query || evt.data?.result?.graphql_query || evt.data?.data?.result?.graphql || evt.data?.result?.graphql || '');
            if (graphql) {
              finalGraphQLQuery.value = sanitizeGraphQL(graphql);
              // Automatically fetch data for the new query
              runDataQuery();
            }
          }

          // Force Vue reactivity update
          chatMessages.value = [...chatMessages.value];
          break;
        }

        case 'complete':
          finalResult = evt.data?.result || evt.data || {};
          streamingMsg.isStreaming = false;
          streamingMsg.content = '✅ Pipeline completed – processing results...';
          console.log('🎉 Pipeline completed:', finalResult);
          break;

        case 'error':
          streamingMsg.isStreaming = false;
          streamingMsg.content = `❌ Error: ${evt.data?.error || 'unknown'}`;
          console.error('❌ Pipeline error:', evt.data);
          break;
      }
    }

    if (!finalResult) throw new Error('MCP stream finished without a result');

    console.log('✅ MCP final result:', finalResult);

    // Handle translation result from final summary
    if (finalResult.translation?.graphql_query) {
      finalGraphQLQuery.value = sanitizeGraphQL(finalResult.translation.graphql_query);

      // Only add translation message if we don't already have one from agent_complete
      const hasTranslatorMessage = chatMessages.value.some(msg => msg.agent === 'translator');
      if (!hasTranslatorMessage) {
        chatMessages.value.push({
          role: 'agent',
          agent: 'translator',
          content: `TRANSLATION COMPLETE\n\nGraphQL Query:\n${finalResult.translation.graphql_query}\n\nConfidence: ${finalResult.translation.confidence || 'N/A'}\n\nExplanation: ${finalResult.translation.explanation || 'No explanation provided'}\n\n${finalResult.translation.suggestions?.length ? `Suggestions:\n${finalResult.translation.suggestions.map((s: string) => `- ${s}`).join('\n')}` : ''}`,
          timestamp: new Date().toLocaleTimeString(),
          isStreaming: false
        });
      }
      // Fetch data automatically for the final translation
      runDataQuery();
    }

    // Handle review from final summary
    if (finalResult.review && Object.keys(finalResult.review).length > 0) {
      const hasReviewerMessage = chatMessages.value.some(msg => msg.agent === 'reviewer');
      if (!hasReviewerMessage) {
        chatMessages.value.push({
          role: 'agent',
          agent: 'mcp_reviewer',
          content: formatReviewResult(finalResult.review),
          timestamp: new Date().toLocaleTimeString(),
          isStreaming: false
        });
      }
    }

    // Completion summary
    chatMessages.value.push({
      role: 'agent',
      agent: 'system',
      content: `Pipeline completed in ${finalResult.processing_time?.toFixed?.(2) || '?.??'}s using ${finalResult.pipeline_strategy || 'unknown'} strategy`,
      timestamp: new Date().toLocaleTimeString()
    });

    // Store prompts if present in final result
    if (finalResult.prompts) {
      Object.entries(finalResult.prompts).forEach(([agent, prompt]: [string, any]) => {
        agentPrompts.value.push({
          agent: agent,
          prompt,
          result: finalResult[agent]?.result || {},
          timestamp: new Date().toLocaleTimeString()
        });
      });
    }

    buildChatContext();

  } catch (error) {
    console.error('💥 MCP Pipeline error:', error);
    
    // Update the streaming message with error
    streamingMsg.isStreaming = false;
    streamingMsg.content = `❌ Error: ${error.message || 'Unknown error occurred'}`;
    
    // Add additional error message
    chatMessages.value.push({
      role: 'agent',
      agent: 'system',
      content: `❌ Pipeline failed: ${error.message || 'Unknown error occurred'}`,
      timestamp: new Date().toLocaleTimeString(),
    });
  } finally {
    console.log('🏁 MCP Pipeline finished');
    isProcessing.value = false;
  }
};

// The old handleStreamEvent function has been removed since we're now using MCP client directly

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
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            if (parsed.token) {
              chatMessages.value[messageIndex].content += parsed.token;
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
    if (!isValidGraphQL(finalGraphQLQuery.value)) {
      console.warn('⚠️ Ignoring invalid GraphQL query:', finalGraphQLQuery.value);
      return;
    }
    
    isDataLoading.value = true;
    dataQueryResults.value = [];
    
    // Add a streaming message for data query
    const dataQueryMsg = {
      role: 'agent',
      agent: 'data_query',
      content: '🔍 Executing GraphQL query...',
      timestamp: new Date().toLocaleTimeString(),
      isStreaming: true
    };
    chatMessages.value.push(dataQueryMsg);
    
    try {
        const response = await fetch('/api/data/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ graphql_query: finalGraphQLQuery.value }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch data');
        }
        
        const results = await response.json();
        dataQueryResults.value = results.results || [];
        
        // Update the streaming message with results
        dataQueryMsg.content = `Data Query Complete\n\nResults: ${dataQueryResults.value.length} records found\n\nQuery:\n${finalGraphQLQuery.value}`;
        dataQueryMsg.isStreaming = false;
        
    } catch (error) {
        console.error('Data query error:', error);
        dataQueryResults.value = [{ error: String(error) }];
        
        // Update the streaming message with error
        dataQueryMsg.content = `Data Query Failed\n\nError: ${error.message || 'Unknown error'}\n\nQuery:\n${finalGraphQLQuery.value}`;
        dataQueryMsg.isStreaming = false;
    } finally {
        isDataLoading.value = false;
        // Force Vue reactivity update
        chatMessages.value = [...chatMessages.value];
    }
}

// Auth is now handled globally in App.vue

// Lifecycle
onMounted(() => {
  authStore.loadPersistedSession();
});

function formatReviewResult(result) {
  if (!result) return 'No review result available';
  
  let content = `**Review Status:** ${result.passed ? '✅ Passed' : '❌ Failed'}\n\n`;
  
  if (result.comments && result.comments.length > 0) {
    content += `**Comments:**\n${result.comments.map(c => `- ${c}`).join('\n')}\n\n`;
  }
  
  if (result.suggested_improvements && result.suggested_improvements.length > 0) {
    content += `**Suggested Improvements:**\n${result.suggested_improvements.map(s => `- ${s}`).join('\n')}\n\n`;
  }
  
  if (result.security_concerns && result.security_concerns.length > 0) {
    content += `**Security Concerns:**\n${result.security_concerns.map(s => `- ${s}`).join('\n')}\n\n`;
  }
  
  if (result.performance_score !== undefined) {
    content += `**Performance Score:** ${result.performance_score}/10\n\n`;
  }
  
  if (result.suggested_query) {
    content += `**Suggested Query:**\n\`\`\`graphql\n${result.suggested_query}\n\`\`\``;
  }
  
  return content;
}

function formatDataReviewResult(result) {
  if (!result) return 'No data review result available';
  
  let content = `**Data Review Analysis**\n\n`;
  
  // Satisfaction status
  content += `**Satisfied:** ${result.satisfied ? '✅ Yes' : '❌ No'}\n\n`;
  
  // Accuracy score
  if (result.accuracy_score !== undefined) {
    content += `**Accuracy Score:** ${result.accuracy_score}/10\n\n`;
  }
  
  // Data quality
  if (result.data_quality) {
    const qualityEmoji = {
      'excellent': '🌟',
      'good': '✅',
      'poor': '⚠️',
      'failed': '❌'
    };
    content += `**Data Quality:** ${qualityEmoji[result.data_quality] || '❓'} ${result.data_quality}\n\n`;
  }
  
  // Issues found
  if (result.issues_found && result.issues_found.length > 0) {
    content += `**Issues Found:**\n${result.issues_found.map(issue => `- ${issue}`).join('\n')}\n\n`;
  }
  
  // Suggestions
  if (result.suggestions && result.suggestions.length > 0) {
    content += `**Suggestions:**\n${result.suggestions.map(suggestion => `- ${suggestion}`).join('\n')}\n\n`;
  }
  
  // Explanation
  if (result.explanation) {
    content += `**Analysis:**\n${result.explanation}\n\n`;
  }
  
  // Iteration info
  if (result.iteration) {
    content += `**Iteration:** ${result.iteration}\n\n`;
  }
  
  // Suggested query
  if (result.suggested_query) {
    content += `**Improved Query:**\n\`\`\`graphql\n${result.suggested_query}\n\`\`\`\n\n`;
  }
  
  // Error handling
  if (result.error) {
    content += `**Error:** ${result.error}\n\n`;
  }
  
  if (result.status === 'failed') {
    content += `**Status:** ❌ Failed\n\n`;
  } else if (result.status === 'skipped') {
    content += `**Status:** ⏭️ Skipped - ${result.reason || 'No reason provided'}\n\n`;
  }
  
  return content;
}

function formatAgentResult(result) {
  if (!result) return 'No result available';
  return `<pre>${JSON.stringify(result, null, 2)}</pre>`;
}

// 📥 Download results (JSON + multimodal links) – mirrors the button inside DataResults
const downloadResults = () => {
  if (!dataQueryResults.value.length) return;
  const blob = new Blob([JSON.stringify(dataQueryResults.value, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'results.json';
  link.click();
  URL.revokeObjectURL(url);
};
</script>

<style scoped>
.btn-primary {
  @apply bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed;
}
</style>