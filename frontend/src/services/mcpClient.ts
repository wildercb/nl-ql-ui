/**
 * MCP Client Service
 * 
 * Handles communication with the Enhanced MCP Server
 * Provides a clean abstraction layer for calling MCP tools
 */

export interface MCPQueryRequest {
  query: string;
  pipeline_strategy?: 'standard' | 'fast' | 'comprehensive' | 'adaptive';
  translator_model?: string;
  user_id?: string;
  metadata?: Record<string, any>;
}

export interface MCPBatchRequest {
  queries: string[];
  pipeline_strategy?: 'standard' | 'fast' | 'comprehensive' | 'adaptive';
  max_concurrent?: number;
  translator_model?: string;
  user_id?: string;
}

export interface MCPResponse {
  original_query: string;
  rewritten_query?: string;
  translation: {
    graphql_query: string;
    confidence: number;
    explanation: string;
    warnings: string[];
    suggestions: string[];
  };
  review: Record<string, any>;
  processing_time: number;
  session_id: string;
  pipeline_strategy: string;
  events_count: number;
}

export interface MCPBatchResponse {
  successful: number;
  total_queries: number;
  results: MCPResponse[];
  failed_queries: Array<{ query: string; error: string }>;
  processing_time: number;
  session_id: string;
}

export class MCPClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api/mcp') {
    this.baseUrl = baseUrl;
  }

  /**
   * Process a single query using the specified pipeline strategy
   */
  async processQuery(request: MCPQueryRequest): Promise<MCPResponse> {
    const toolName = this.getToolNameForStrategy(request.pipeline_strategy || 'standard');
    
    const response = await fetch(`${this.baseUrl}/tools/${toolName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: request.query,
        translator_model: request.translator_model || 'gemma3:4b',
        user_id: request.user_id || 'frontend_user',
        metadata: request.metadata || {}
      }),
    });

    if (!response.ok) {
      throw new Error(`MCP request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Process multiple queries in batch
   */
  async processBatch(request: MCPBatchRequest): Promise<MCPBatchResponse> {
    const response = await fetch(`${this.baseUrl}/tools/batch_process_queries`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        queries: request.queries,
        pipeline_strategy: request.pipeline_strategy || 'standard',
        max_concurrent: request.max_concurrent || 3,
        translator_model: request.translator_model || 'gemma3:4b',
        user_id: request.user_id || 'frontend_user'
      }),
    });

    if (!response.ok) {
      throw new Error(`MCP batch request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get available tools from MCP server
   */
  async getAvailableTools(): Promise<Array<{ name: string; description: string }>> {
    const response = await fetch(`${this.baseUrl}/tools`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get MCP tools: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Get server information
   */
  async getServerInfo(): Promise<{ name: string; version: string; capabilities: string[] }> {
    const response = await fetch(`${this.baseUrl}/info`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Failed to get MCP server info: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Stream query processing with real-time updates
   */
  async *processQueryStream(request: MCPQueryRequest): AsyncGenerator<{
    event: string;
    data: any;
  }> {
    const toolName = this.getToolNameForStrategy(request.pipeline_strategy || 'standard');
    
    const response = await fetch(`${this.baseUrl}/tools/${toolName}/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: request.query,
        translator_model: request.translator_model || 'gemma3:4b',
        user_id: request.user_id || 'frontend_user',
        metadata: request.metadata || {}
      }),
    });

    if (!response.ok) {
      throw new Error(`MCP stream request failed: ${response.status} ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('No response body for streaming');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              // Handle server-sent events format
              if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                yield { event: data.event || 'data', data };
              } else {
                // Handle plain JSON events
                const event = JSON.parse(line);
                yield { event: event.event || 'data', data: event };
              }
            } catch (error) {
              console.warn('Failed to parse streaming event:', line, error);
              // Continue processing other lines even if one fails
            }
          }
        }
      }
      
      // Process any remaining buffer content
      if (buffer.trim()) {
        try {
          if (buffer.startsWith('data: ')) {
            const data = JSON.parse(buffer.slice(6));
            yield { event: data.event || 'data', data };
          } else {
            const event = JSON.parse(buffer);
            yield { event: event.event || 'data', data: event };
          }
        } catch (error) {
          console.warn('Failed to parse final buffer:', buffer, error);
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw new Error(`Streaming failed: ${error.message}`);
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Map pipeline strategy to MCP tool name
   */
  private getToolNameForStrategy(strategy: string): string {
    const toolMap: Record<string, string> = {
      'fast': 'process_query_fast',
      'standard': 'process_query_standard', 
      'comprehensive': 'process_query_comprehensive',
      'adaptive': 'process_query_adaptive'
    };

    return toolMap[strategy] || 'process_query_standard';
  }
}

// Export singleton instance
export const mcpClient = new MCPClient(); 