<template>
  <div class="flex-1 bg-gray-50 dark:bg-gray-900">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Settings
        </h1>
        <p class="text-lg text-gray-600 dark:text-gray-300">
          Configure your translation preferences and API settings
        </p>
      </div>
      
      <div class="space-y-6">
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              API Configuration
            </h2>
          </div>
          <div class="card-body space-y-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Backend API URL
              </label>
              <input
                type="text"
                class="input"
                placeholder="http://localhost:8000"
                value="http://localhost:8000"
                readonly
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                MongoDB Connection
              </label>
              <input
                type="text"
                class="input"
                placeholder="mongodb://localhost:27017"
                value="mongodb://localhost:27017"
                readonly
              />
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-header">
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">
              Theme Settings
            </h2>
          </div>
          <div class="card-body">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
                Dark Mode
              </span>
              <button class="btn-outline">
                Toggle Theme
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Settings view component
</script> 