/**
 * Frontend Debug Utilities
 * Comprehensive monitoring and logging for the Vue.js frontend
 */

export interface DebugConfig {
  enabled: boolean;
  apiLogging: boolean;
  userInteractionLogging: boolean;
  performanceLogging: boolean;
  consoleOutput: boolean;
  storageLogging: boolean;
}

export interface APICallLog {
  timestamp: string;
  method: string;
  url: string;
  requestData?: any;
  responseData?: any;
  responseTime: number;
  status: number;
  error?: string;
}

export interface UserInteractionLog {
  timestamp: string;
  event: string;
  element?: string;
  data?: any;
  path: string;
}

export interface PerformanceLog {
  timestamp: string;
  type: 'navigation' | 'render' | 'api' | 'interaction';
  name: string;
  duration: number;
  details?: any;
}

class DebugManager {
  private config: DebugConfig;
  private apiLogs: APICallLog[] = [];
  private userLogs: UserInteractionLog[] = [];
  private performanceLogs: PerformanceLog[] = [];

  constructor() {
    this.config = {
      enabled: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1',
      apiLogging: true,
      userInteractionLogging: true,
      performanceLogging: true,
      consoleOutput: true,
      storageLogging: true
    };

    if (this.config.enabled) {
      this.setupGlobalErrorHandling();
      this.setupPerformanceMonitoring();
      this.log('🐛 Debug mode enabled', 'init');
    }
  }

  private setupGlobalErrorHandling() {
    window.addEventListener('error', (event) => {
      this.log(`❌ Global Error: ${event.error?.message}`, 'error', {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.log(`❌ Unhandled Promise Rejection: ${event.reason}`, 'error', {
        reason: event.reason
      });
    });
  }

  private setupPerformanceMonitoring() {
    if ('performance' in window) {
      // Monitor navigation timing
      window.addEventListener('load', () => {
        const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        this.logPerformance('navigation', 'page-load', perfData.loadEventEnd, {
          domContentLoaded: perfData.domContentLoadedEventEnd,
          firstPaint: this.getFirstPaint(),
          firstContentfulPaint: this.getFirstContentfulPaint()
        });
      });
    }
  }

  private getFirstPaint(): number | null {
    const paintEntries = performance.getEntriesByType('paint');
    const firstPaint = paintEntries.find(entry => entry.name === 'first-paint');
    return firstPaint ? firstPaint.startTime : null;
  }

  private getFirstContentfulPaint(): number | null {
    const paintEntries = performance.getEntriesByType('paint');
    const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint');
    return fcp ? fcp.startTime : null;
  }

  // API Call Logging
  logAPICall(
    method: string,
    url: string,
    requestData?: any,
    responseData?: any,
    responseTime?: number,
    status?: number,
    error?: string
  ) {
    if (!this.config.enabled || !this.config.apiLogging) return;

    const logEntry: APICallLog = {
      timestamp: new Date().toISOString(),
      method: method.toUpperCase(),
      url,
      requestData,
      responseData,
      responseTime: responseTime || 0,
      status: status || 0,
      error
    };

    this.apiLogs.push(logEntry);
    this.trimLogs('api');

    if (this.config.consoleOutput) {
      const emoji = status && status >= 200 && status < 300 ? '✅' : '❌';
      console.group(`${emoji} API ${method.toUpperCase()} ${url}`);
      console.log('📤 Request:', requestData);
      console.log('📥 Response:', responseData);
      console.log('⏱️ Time:', `${responseTime}ms`);
      console.log('📊 Status:', status);
      if (error) console.log('❌ Error:', error);
      console.groupEnd();
    }

    this.saveToStorage();
  }

  // User Interaction Logging
  logUserInteraction(event: string, element?: string, data?: any) {
    if (!this.config.enabled || !this.config.userInteractionLogging) return;

    const logEntry: UserInteractionLog = {
      timestamp: new Date().toISOString(),
      event,
      element,
      data,
      path: window.location.pathname
    };

    this.userLogs.push(logEntry);
    this.trimLogs('user');

    if (this.config.consoleOutput) {
      console.log(`👤 User ${event}:`, { element, data, path: logEntry.path });
    }

    this.saveToStorage();
  }

  // Performance Logging
  logPerformance(type: PerformanceLog['type'], name: string, duration: number, details?: any) {
    if (!this.config.enabled || !this.config.performanceLogging) return;

    const logEntry: PerformanceLog = {
      timestamp: new Date().toISOString(),
      type,
      name,
      duration,
      details
    };

    this.performanceLogs.push(logEntry);
    this.trimLogs('performance');

    if (this.config.consoleOutput) {
      console.log(`⚡ Performance ${type}:`, { name, duration: `${duration}ms`, details });
    }

    this.saveToStorage();
  }

  // Generic Logging
  log(message: string, type: string = 'info', data?: any) {
    if (!this.config.enabled || !this.config.consoleOutput) return;

    const timestamp = new Date().toISOString();
    const emoji = this.getEmojiForType(type);
    
    console.log(`${emoji} [${timestamp}] ${message}`, data || '');
  }

  private getEmojiForType(type: string): string {
    const emojiMap: Record<string, string> = {
      'info': 'ℹ️',
      'error': '❌',
      'warning': '⚠️',
      'success': '✅',
      'debug': '🐛',
      'init': '🚀',
      'api': '🌐',
      'user': '👤',
      'performance': '⚡'
    };
    return emojiMap[type] || 'ℹ️';
  }

  // Log Management
  private trimLogs(type: 'api' | 'user' | 'performance') {
    const maxLogs = 1000;
    
    switch (type) {
      case 'api':
        if (this.apiLogs.length > maxLogs) {
          this.apiLogs = this.apiLogs.slice(-maxLogs);
        }
        break;
      case 'user':
        if (this.userLogs.length > maxLogs) {
          this.userLogs = this.userLogs.slice(-maxLogs);
        }
        break;
      case 'performance':
        if (this.performanceLogs.length > maxLogs) {
          this.performanceLogs = this.performanceLogs.slice(-maxLogs);
        }
        break;
    }
  }

  private saveToStorage() {
    if (!this.config.storageLogging) return;

    try {
      localStorage.setItem('debug-api-logs', JSON.stringify(this.apiLogs.slice(-100)));
      localStorage.setItem('debug-user-logs', JSON.stringify(this.userLogs.slice(-100)));
      localStorage.setItem('debug-performance-logs', JSON.stringify(this.performanceLogs.slice(-100)));
    } catch (error) {
      console.warn('Failed to save debug logs to localStorage:', error);
    }
  }

  // Public Methods for Retrieving Logs
  getAPILogs(): APICallLog[] {
    return [...this.apiLogs];
  }

  getUserLogs(): UserInteractionLog[] {
    return [...this.userLogs];
  }

  getPerformanceLogs(): PerformanceLog[] {
    return [...this.performanceLogs];
  }

  getAllLogs() {
    return {
      api: this.getAPILogs(),
      user: this.getUserLogs(),
      performance: this.getPerformanceLogs()
    };
  }

  // Export logs for debugging
  exportLogs() {
    const logs = this.getAllLogs();
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `debug-logs-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Clear all logs
  clearLogs() {
    this.apiLogs = [];
    this.userLogs = [];
    this.performanceLogs = [];
    localStorage.removeItem('debug-api-logs');
    localStorage.removeItem('debug-user-logs');
    localStorage.removeItem('debug-performance-logs');
    this.log('🧹 All debug logs cleared', 'info');
  }

  // Performance measurement helper
  measurePerformance<T>(name: string, fn: () => T): T;
  measurePerformance<T>(name: string, fn: () => Promise<T>): Promise<T>;
  measurePerformance<T>(name: string, fn: () => T | Promise<T>): T | Promise<T> {
    const start = performance.now();
    const result = fn();

    if (result instanceof Promise) {
      return result.finally(() => {
        const duration = performance.now() - start;
        this.logPerformance('api', name, duration);
      });
    } else {
      const duration = performance.now() - start;
      this.logPerformance('render', name, duration);
      return result;
    }
  }
}

// Create global instance
const debugManager = new DebugManager();

// Export for global access
(window as any).debugManager = debugManager;

export default debugManager; 