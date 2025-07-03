<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Documentation
        </h1>
        <p class="text-lg text-gray-600 dark:text-gray-300">
          Learn how to use the Natural Language to GraphQL Translator
        </p>
      </div>
      
      <div class="space-y-6">
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Getting Started
            </h2>
          </div>
          <div class="card-body prose dark:prose-invert max-w-none">
            <p>
              The Natural Language to GraphQL Translator allows you to convert natural language queries into GraphQL queries using AI-powered translation.
            </p>
            <h3>How to Use</h3>
            <ol>
              <li>Navigate to the <strong>Translator</strong> page</li>
              <li>Enter your natural language query in the input field</li>
              <li>Click the <strong>Translate</strong> button</li>
              <li>Review the generated GraphQL query</li>
            </ol>
            <h3>Example Queries</h3>
            <ul>
              <li>"Get all users with their posts"</li>
              <li>"Find posts by user John published last month"</li>
              <li>"Show me comments on posts with more than 10 likes"</li>
            </ul>
          </div>
        </div>
        
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              API Reference
            </h2>
          </div>
          <div class="card-body prose dark:prose-invert max-w-none">
            <p>
              The backend API provides endpoints for translation, analytics, and model management.
            </p>
            <h3>Endpoints</h3>
            <ul>
              <li><code>/api/translate</code> - Translate natural language to GraphQL</li>
              <li><code>/api/analytics</code> - Get analytics data</li>
              <li><code>/api/models</code> - Manage AI models</li>
              <li><code>/health</code> - Health check endpoint</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Documentation view component
</script> 