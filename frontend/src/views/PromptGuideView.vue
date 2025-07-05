<template>
  <div class="flex-1 bg-gray-50 dark:bg-gray-900">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <article
        class="prose dark:prose-invert max-w-none"
        v-html="htmlContent"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { marked } from 'marked';
// Import the markdown file as a raw string
import guideMarkdown from '@/assets/docs/PromptEngineeringGuide.md?raw';

// Convert markdown to HTML at runtime
const htmlContent = ref<string>(marked.parse(guideMarkdown));
</script> 