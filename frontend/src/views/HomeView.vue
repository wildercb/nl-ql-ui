<template>
  <div class="home-view">
    <h1>Natural Language to GraphQL Translator</h1>
    
    <!-- Query Input Section -->
    <div class="query-section">
      <textarea
        v-model="naturalQuery"
        placeholder="Enter your natural language query here (e.g., 'get all users')"
        rows="3"
        class="query-input"
      ></textarea>
      <div class="query-actions">
        <button @click="translateQuery" :disabled="isLoading" class="translate-btn">
          {{ isLoading ? 'Translating...' : 'Translate to GraphQL' }}
        </button>
        <div class="model-selection">
          <label for="model">Model:</label>
          <select v-model="selectedModel" id="model" class="model-dropdown">
            <option v-for="model in models" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
        </div>
      </div>
    </div>

    <!-- Translation Results Section -->
    <div class="results-section" v-if="translationResult">
      <h2>Translation Results</h2>
      <div class="result-card">
        <h3>GraphQL Query</h3>
        <pre><code>{{ translationResult.graphql_query || 'Not available' }}</code></pre>
        <div class="result-meta">
          <span>Confidence: {{ translationResult.confidence.toFixed(2) }}</span>
          <span>Processing Time: {{ translationResult.processing_time.toFixed(2) }}s</span>
          <span>Model: {{ translationResult.model_used }}</span>
        </div>
        <div class="explanation" v-if="translationResult.explanation">
          <h4>Explanation</h4>
          <p>{{ translationResult.explanation }}</p>
        </div>
        <div class="warnings" v-if="translationResult.warnings && translationResult.warnings.length > 0">
          <h4>Warnings</h4>
          <ul>
            <li v-for="warning in translationResult.warnings" :key="warning">{{ warning }}</li>
          </ul>
        </div>
        <div class="suggestions" v-if="translationResult.suggested_improvements && translationResult.suggested_improvements.length > 0">
          <h4>Suggestions</h4>
          <ul>
            <li v-for="suggestion in translationResult.suggested_improvements" :key="suggestion">{{ suggestion }}</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Query History Section -->
    <div class="history-section">
      <h2>Query History</h2>
      <div v-if="queryHistory.length > 0" class="history-list">
        <div v-for="query in queryHistory" :key="query.id" class="history-item" @click="loadQueryToChat(query)">
          <div class="history-query">{{ query.original_query }}</div>
          <div class="history-meta">
            <span>Confidence: {{ query.confidence.toFixed(2) }}</span>
            <span>Time: {{ new Date(query.timestamp).toLocaleString() }}</span>
          </div>
        </div>
      </div>
      <p v-else>No queries in history.</p>
      <button @click="fetchHistory" :disabled="isLoadingHistory" class="refresh-btn">
        {{ isLoadingHistory ? 'Loading...' : 'Refresh History' }}
      </button>
    </div>

    <!-- Live Model Interactions Section -->
    <div class="live-interactions-section">
      <h2>Live Model Interactions</h2>
      <div v-if="liveInteractions.length > 0" class="interactions-list">
        <div v-for="interaction in liveInteractions" :key="interaction.id" class="interaction-item">
          <div class="interaction-agent">Agent: {{ interaction.agent || 'Unknown' }}</div>
          <div class="interaction-message">{{ interaction.message }}</div>
          <div class="interaction-time">{{ new Date(interaction.timestamp).toLocaleTimeString() }}</div>
          <button @click="loadInteractionToChat(interaction)" class="chat-btn">Chat from here</button>
        </div>
      </div>
      <p v-else>No live interactions currently.</p>
    </div>

    <!-- Chat Interface Section -->
    <div class="chat-section" v-if="isChatActive">
      <h2>Chat with Model</h2>
      <div class="chat-messages">
        <div v-for="msg in chatMessages" :key="msg.id" :class="['chat-message', msg.sender]">
          <div class="message-content">{{ msg.content }}</div>
          <div class="message-time">{{ new Date(msg.timestamp).toLocaleTimeString() }}</div>
        </div>
      </div>
      <div class="chat-input">
        <textarea v-model="chatInput" placeholder="Type your message..." rows="2" class="chat-textarea"></textarea>
        <button @click="sendChatMessage" :disabled="isChatSending" class="send-btn">
          {{ isChatSending ? 'Sending...' : 'Send' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import { ref, onMounted, onUnmounted } from 'vue';

export default {
  name: 'HomeView',
  setup() {
    const naturalQuery = ref('');
    const translationResult = ref(null);
    const isLoading = ref(false);
    const models = ref(['phi3:mini']);
    const selectedModel = ref('phi3:mini');
    const queryHistory = ref([]);
    const isLoadingHistory = ref(false);
    const liveInteractions = ref([]);
    const isChatActive = ref(false);
    const chatMessages = ref([]);
    const chatInput = ref('');
    const isChatSending = ref(false);
    let eventSource = null;

    const fetchHistory = async () => {
      isLoadingHistory.value = true;
      try {
        const response = await axios.get('/api/history', { params: { limit: 50 } });
        queryHistory.value = response.data;
      } catch (error) {
        console.error('Failed to fetch query history:', error);
        alert('Failed to load history. Please try again.');
      } finally {
        isLoadingHistory.value = false;
      }
    };

    const translateQuery = async () => {
      if (!naturalQuery.value.trim()) {
        alert('Please enter a query.');
        return;
      }

      isLoading.value = true;
      translationResult.value = null;

      try {
        const response = await axios.post('/api/translate/stream', {
          query: naturalQuery.value,
          model: selectedModel.value
        });
        translationResult.value = response.data;
        fetchHistory(); // Refresh history after new query
      } catch (error) {
        console.error('Translation error:', error);
        translationResult.value = {
          graphql_query: '',
          confidence: 0,
          explanation: 'Translation failed due to an error.',
          model_used: selectedModel.value,
          processing_time: 0,
          original_query: naturalQuery.value,
          suggested_improvements: ['Try again or check server status.'],
          warnings: [error.message || 'Unknown error']
        };
      } finally {
        isLoading.value = false;
      }
    };

    const setupLiveInteractions = () => {
      eventSource = new EventSource('/api/interactions/stream');
      eventSource.onmessage = (event) => {
        const interaction = JSON.parse(event.data);
        liveInteractions.value.push(interaction);
        if (liveInteractions.value.length > 50) {
          liveInteractions.value.shift(); // Keep only the latest 50 interactions
        }
      };
      eventSource.onerror = () => {
        console.error('Error in live interactions stream');
        eventSource.close();
        setTimeout(setupLiveInteractions, 5000); // Reconnect after 5s
      };
    };

    const loadQueryToChat = (query) => {
      isChatActive.value = true;
      chatMessages.value = [{
        id: Date.now().toString(),
        sender: 'user',
        content: query.original_query,
        timestamp: new Date().toISOString()
      }, {
        id: (Date.now() + 1).toString(),
        sender: 'model',
        content: query.graphql_query || 'No GraphQL query generated.',
        timestamp: new Date().toISOString()
      }];
    };

    const loadInteractionToChat = (interaction) => {
      isChatActive.value = true;
      chatMessages.value = [{
        id: Date.now().toString(),
        sender: interaction.agent || 'agent',
        content: interaction.message,
        timestamp: new Date().toISOString()
      }];
    };

    const sendChatMessage = async () => {
      if (!chatInput.value.trim()) return;

      const userMessage = {
        id: Date.now().toString(),
        sender: 'user',
        content: chatInput.value,
        timestamp: new Date().toISOString()
      };
      chatMessages.value.push(userMessage);
      chatInput.value = '';
      isChatSending.value = true;

      try {
        const response = await axios.post('/api/chat', {
          query: userMessage.content,
          model: selectedModel.value,
          context: chatMessages.value
        });
        const modelMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'model',
          content: response.data.response || 'No response from model.',
          timestamp: new Date().toISOString()
        };
        chatMessages.value.push(modelMessage);
      } catch (error) {
        console.error('Chat error:', error);
        const errorMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'model',
          content: 'Error: Could not get response from model.',
          timestamp: new Date().toISOString()
        };
        chatMessages.value.push(errorMessage);
      } finally {
        isChatSending.value = false;
      }
    };

    onMounted(() => {
      fetchHistory();
      setupLiveInteractions();
    });

    onUnmounted(() => {
      if (eventSource) {
        eventSource.close();
      }
    });

    return {
      naturalQuery,
      translationResult,
      isLoading,
      models,
      selectedModel,
      translateQuery,
      queryHistory,
      fetchHistory,
      isLoadingHistory,
      liveInteractions,
      isChatActive,
      chatMessages,
      chatInput,
      sendChatMessage,
      isChatSending
    };
  }
};
</script>

<style scoped>
.home-view {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

h1, h2 {
  color: #333;
  margin-bottom: 20px;
}

.query-section, .results-section, .history-section, .live-interactions-section, .chat-section {
  margin-bottom: 40px;
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.query-input {
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 15px;
  resize: vertical;
}

.query-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.translate-btn, .refresh-btn, .chat-btn, .send-btn {
  background-color: #007bff;
  color: white;
  border: none;
  padding: 10px 15px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.translate-btn:hover, .refresh-btn:hover, .chat-btn:hover, .send-btn:hover {
  background-color: #0056b3;
}

.translate-btn:disabled, .refresh-btn:disabled, .send-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.model-selection {
  display: flex;
  align-items: center;
}

.model-dropdown {
  margin-left: 10px;
  padding: 8px;
  font-size: 14px;
  border-radius: 4px;
  border: 1px solid #ddd;
}

.result-card {
  border: 1px solid #eee;
  padding: 15px;
  border-radius: 6px;
  background: #f9f9f9;
}

.result-card pre {
  background: #e9ecef;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
}

.result-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 14px;
  color: #666;
}

.history-item, .interaction-item {
  border-bottom: 1px solid #eee;
  padding: 10px 0;
  cursor: pointer;
  transition: background 0.2s;
}

.history-item:hover, .interaction-item:hover {
  background: #f5f5f5;
}

.history-query, .interaction-message {
  font-weight: bold;
  margin-bottom: 5px;
}

.history-meta, .interaction-time {
  font-size: 12px;
  color: #888;
}

.interaction-agent {
  font-weight: bold;
  color: #007bff;
}

.chat-messages {
  border: 1px solid #ddd;
  height: 300px;
  overflow-y: auto;
  padding: 10px;
  margin-bottom: 15px;
  background: #f9f9f9;
  border-radius: 4px;
}

.chat-message {
  margin-bottom: 10px;
  padding: 8px;
  border-radius: 4px;
  max-width: 80%;
}

.chat-message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
}

.chat-message.model, .chat-message.agent {
  background: #e9ecef;
  margin-right: auto;
}

.message-content {
  word-wrap: break-word;
}

.message-time {
  font-size: 11px;
  margin-top: 5px;
  opacity: 0.8;
}

.chat-input {
  display: flex;
  gap: 10px;
}

.chat-textarea {
  flex: 1;
  padding: 10px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: none;
}
</style> 