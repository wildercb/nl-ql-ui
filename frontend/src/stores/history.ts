import { defineStore } from 'pinia';

export const useHistoryStore = defineStore('history', {
  state: () => ({
    selectedHistoryItem: null as any | null,
  }),
  actions: {
    setSelectedHistoryItem(item: any) {
      this.selectedHistoryItem = item;
    },
    clearSelectedHistoryItem() {
      this.selectedHistoryItem = null;
    },
  },
}); 