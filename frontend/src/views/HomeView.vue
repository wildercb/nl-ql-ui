<template>
  <div class="flex flex-col h-screen bg-gray-900 text-white">
    <div class="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 overflow-hidden">
      <!-- Left Column: NL Query Input + Agent Stream -->
      <div class="flex flex-col h-full min-h-0 border border-gray-700 rounded-none p-4">
        <h2 class="text-lg font-semibold text-gray-200 mb-2">Enter Natural Language Query Here</h2>
        <!-- NL Query Input (fixed height) -->
        <div class="flex-none">
          <textarea
            v-model="naturalQuery"
            @keyup.enter="runPipeline('standard')"
            :disabled="isProcessing"
            class="w-full resize-none p-3 bg-gray-700 rounded-none focus:outline-none focus:ring-2 focus:ring-purple-500 text-base min-h-[90px] max-h-[200px]"
            placeholder="Enter your natural language query here..."
          ></textarea>
          <div class="flex flex-wrap justify-center items-center gap-2 mt-2">
            <button @click="runPipeline('fast')" :disabled="isProcessing" class="btn-primary">
              <i class="fas fa-bolt mr-2"></i> Translate
            </button>
            <button @click="runPipeline('standard')" :disabled="isProcessing" class="btn-primary">
              <i class="fas fa-users-cog mr-2"></i> Multi-Agent
            </button>
            <button @click="runPipeline('comprehensive')" :disabled="isProcessing" class="btn-primary">
              <i class="fas fa-rocket mr-2"></i> Enhanced Agents
            </button>
            <select v-model="selectedModel" class="p-2 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500" :disabled="isProcessing">
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
          </div>
        </div>
        <!-- Agent Stream Chat (fills remaining space, scrollable) -->
        <div class="flex-1 min-h-0">
          <ChatStream 
            :messages="chatMessages" 
            :loading="isProcessing" 
            title="Agent Stream" 
            :selectedModel="selectedModel"
            :prompts="agentPrompts"
            @sendMessage="handleChatMessage"
          />
        </div>
      </div>

      <!-- Right Column: GraphQL Query + Results -->
      <div class="flex flex-col h-full min-h-0 space-y-0 border border-gray-700 rounded-none p-4">
        <h2 class="text-lg font-semibold text-gray-200 mb-2">Get Results Here</h2>
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

const authStore = useAuthStore();
const historyStore = useHistoryStore();
const router = useRouter();

// Model selection
const selectedModel = ref('phi3:mini');

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

      // Ensure a placeholder prompt is recorded so it appears immediately when prompts are toggled
      const existingPromptIdx = agentPrompts.value.findIndex(p => p.agent === data.agent);
      if (data.prompt) {
        if (existingPromptIdx !== -1) {
          agentPrompts.value[existingPromptIdx].prompt = data.prompt;
        } else {
          agentPrompts.value.push({
            agent: data.agent,
            prompt: data.prompt,
            result: null,
            timestamp: new Date().toLocaleTimeString()
          });
        }
      } else if (existingPromptIdx === -1) {
        // no prompt yet, create placeholder
        agentPrompts.value.push({
          agent: data.agent,
          prompt: 'Generating prompt…',
          result: null,
          timestamp: new Date().toLocaleTimeString()
        });
      }
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
            finalGraphQLQuery.value = sanitizeGraphQL(data.result.suggested_query);

            // Automatically execute the newly suggested query so results stream in live
            try {
              // Fire and forget – we don't need to await here
              runDataQuery();
            } catch (e) {
              console.error('Failed to auto-run data query:', e);
            }
          }
        } else if (data.agent === 'data_reviewer') {
          agentMessage.content = formatDataReviewResult(data.result);
          // If data reviewer suggests a new query, update and execute it
          if (data.result && data.result.suggested_query) {
            console.log('🔍 Data reviewer suggested new GraphQL query:', data.result.suggested_query);
            finalGraphQLQuery.value = sanitizeGraphQL(data.result.suggested_query);
            
            // Show the query execution results in the data reviewer message
            if (data.result.query_result) {
              const queryResult = data.result.query_result;
              if (queryResult.success) {
                agentMessage.content += `\n\n📊 **Query Results:**\n\`\`\`json\n${JSON.stringify(queryResult.data, null, 2)}\n\`\`\``;
              } else {
                agentMessage.content += `\n\n❌ **Query Failed:**\n\`\`\`\n${queryResult.error || 'Unknown error'}\n\`\`\``;
                if (queryResult.errors && queryResult.errors.length > 0) {
                  agentMessage.content += `\n\n**Errors:**\n${queryResult.errors.map(e => `- ${e}`).join('\n')}`;
                }
              }
            }

            // Automatically execute the newly suggested query
            try {
              console.log('🚀 Auto-executing data reviewer suggested query');
              runDataQuery();
            } catch (e) {
              console.error('Failed to auto-run data reviewer query:', e);
            }
          }
        } else if (data.agent === 'rewriter') {
          agentMessage.content = data.result?.rewritten_query || '';
        } else if (data.agent === 'translator' && data.result?.graphql_query) {
          console.log('🔍 Setting GraphQL query:', data.result.graphql_query);
          finalGraphQLQuery.value = sanitizeGraphQL(data.result.graphql_query);
        } else if (data.agent === 'data_reviewer') {
          console.log('🔍 Data reviewer completed:', data.result);
          agentMessage.content = formatDataReviewResult(data.result);
          
          // If data reviewer refined the query after analyzing results, update and re-run
          if (data.result?.final_query && data.result.final_query !== finalGraphQLQuery.value) {
            console.log('📝 Data reviewer refined GraphQL query:', data.result.final_query);
            finalGraphQLQuery.value = sanitizeGraphQL(data.result.final_query);
            
            // Auto-execute the refined query
            try {
              runDataQuery();
            } catch (e) {
              console.error('Failed to auto-run refined data query:', e);
            }
          }
          
          // Show iteration details in the chat
          if (data.result?.iterations) {
            data.result.iterations.forEach((iteration, idx) => {
              chatMessages.value.push({
                role: 'agent',
                agent: 'data_reviewer',
                content: `Iteration ${iteration.iteration}: ${iteration.analysis?.reasoning || 'Analyzing results...'}`,
                timestamp: new Date().toLocaleTimeString(),
                isStreaming: false
              });
            });
          }
        }
        console.log('💬 Marked agent as completed, kept output or set result');
      } else {
        console.log('⚠️ Could not find agent message for completion:', data.agent);
      }
      
      // Store agent prompts for context
      if (data.prompt) {
        const idx = agentPrompts.value.findIndex(p => p.agent === data.agent);
        if (idx !== -1) {
          agentPrompts.value[idx].prompt = data.prompt;
          agentPrompts.value[idx].result = data.result;
        } else {
          agentPrompts.value.push({
            agent: data.agent,
            prompt: data.prompt,
            result: data.result,
            timestamp: new Date().toLocaleTimeString()
          });
        }
      }
      break;
      
    case 'pipeline_complete':
      console.log('🏁 Pipeline completed, final data:', data);
      if (data.translation?.graphql_query && !finalGraphQLQuery.value) {
        console.log('🔍 Setting GraphQL query from pipeline complete:', data.translation.graphql_query);
        finalGraphQLQuery.value = sanitizeGraphQL(data.translation.graphql_query);
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
    if (!isValidGraphQL(finalGraphQLQuery.value)) {
      console.warn('⚠️ Ignoring invalid GraphQL query:', finalGraphQLQuery.value);
      return;
    }
    isDataLoading.value = true;
    dataQueryResults.value = [];
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
    } catch (error) {
        console.error('Data query error:', error);
        dataQueryResults.value = [{ error: String(error) }];
    } finally {
        isDataLoading.value = false;
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
</script>

<style scoped>
.btn-primary {
  @apply bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed;
}
</style>