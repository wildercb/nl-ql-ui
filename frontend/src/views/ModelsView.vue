<template>
  <div class="flex-1 bg-gray-50 dark:bg-gray-900">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          AI Models
        </h1>
        <p class="text-lg text-gray-600 dark:text-gray-300">
          Manage and configure AI models for translation
        </p>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div class="card">
          <div class="card-body">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Ollama Models
            </h3>
            <p class="text-gray-600 dark:text-gray-300 mb-4">
              Local AI models running via Ollama
            </p>
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-300">Status:</span>
                <span class="text-sm text-green-600 dark:text-green-400">Connected</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              OpenAI Models
            </h3>
            <p class="text-gray-600 dark:text-gray-300 mb-4">
              Cloud-based AI models via OpenAI API
            </p>
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-300">Status:</span>
                <span class="text-sm text-gray-500 dark:text-gray-400">Not configured</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Anthropic Models
            </h3>
            <p class="text-gray-600 dark:text-gray-300 mb-4">
              Claude models via Anthropic API
            </p>
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-300">Status:</span>
                <span class="text-sm text-gray-500 dark:text-gray-400">Not configured</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Models view component
</script> 