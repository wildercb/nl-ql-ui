import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: 'Natural Language to GraphQL Translator' }
  },
  {
    path: '/translator',
    name: 'Translator',
    component: () => import('@/views/TranslatorView.vue'),
    meta: { title: 'Query Translator' }
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('@/views/ModelsView.vue'),
    meta: { title: 'AI Models' }
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/HistoryView.vue'),
    meta: { title: 'Query History' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { title: 'Settings' }
  },
  {
    path: '/docs',
    name: 'Documentation',
    component: () => import('@/views/DocsView.vue'),
    meta: { title: 'Documentation' }
  },
  {
    path: '/prompt-guide',
    name: 'PromptGuide',
    component: () => import('@/views/PromptGuideView.vue'),
    meta: { title: 'Prompt Engineering Guide' }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: 'Page Not Found' }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  }
})

// Global navigation guard for page titles
router.beforeEach((to, from, next) => {
  const title = to.meta.title as string
  if (title) {
    document.title = `${title} | MPPW MCP`
  }
  next()
})

export default router 