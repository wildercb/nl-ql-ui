import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface User {
  id: string
  username: string
  full_name?: string
  subscription_tier?: string
  total_queries?: number
  successful_queries?: number
}

interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  guestSessionId: string | null
  guestSessionToken: string | null
  isAuthenticated: boolean
  isGuest: boolean
  guestHistory: any[]
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    user: null,
    accessToken: null,
    refreshToken: null,
    guestSessionId: null,
    guestSessionToken: null,
    isAuthenticated: false,
    isGuest: false,
    guestHistory: []
  }),
  
  getters: {
    currentUser: (state) => state.user,
    isLoggedIn: (state) => state.isAuthenticated && !state.isGuest,
    isGuestSession: (state) => state.isAuthenticated && state.isGuest,
    hasValidSession: (state) => state.isAuthenticated && (state.accessToken || state.guestSessionToken)
  },
  
  actions: {
    addGuestHistoryItem(item: any) {
      if (this.isGuest) {
        this.guestHistory.unshift(item);
      }
    },
    setAuthenticated(authData: any) {
      console.log('Setting authenticated user:', authData);
      
      if (authData.user) {
        this.user = authData.user;
        this.accessToken = authData.access_token;
        this.refreshToken = authData.refresh_token;
        this.isAuthenticated = true;
        this.isGuest = false;
        
        // Store in localStorage
        localStorage.setItem('access_token', authData.access_token);
        if (authData.refresh_token) {
          localStorage.setItem('refresh_token', authData.refresh_token);
        }
        localStorage.setItem('user_data', JSON.stringify(authData.user));
        
        console.log('User authenticated successfully');
      }
    },
    
    setGuestSession(sessionData: any) {
      console.log('Setting guest session:', sessionData);
      
      this.user = null;
      this.accessToken = null;
      this.refreshToken = null;
      this.guestSessionId = sessionData.session_id;
      this.guestSessionToken = sessionData.session_token;
      this.isAuthenticated = true;
      this.isGuest = true;
      this.guestHistory = [];
      
      // Store in localStorage
      localStorage.setItem('guest_session_id', sessionData.session_id);
      localStorage.setItem('guest_session_token', sessionData.session_token);
      
      console.log('Guest session created successfully');
    },
    
    logout() {
      console.log('Logging out user');
      
      this.user = null;
      this.accessToken = null;
      this.refreshToken = null;
      this.guestSessionId = null;
      this.guestSessionToken = null;
      this.isAuthenticated = false;
      this.isGuest = false;
      this.guestHistory = [];
      
      // Clear localStorage
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user_data');
      localStorage.removeItem('guest_session_id');
      localStorage.removeItem('guest_session_token');
      
      console.log('User logged out successfully');
    },
    
    loadPersistedSession() {
      console.log('Loading persisted session...');
      
      // Try to load authenticated user session
      const accessToken = localStorage.getItem('access_token');
      const userDataStr = localStorage.getItem('user_data');
      
      if (accessToken && userDataStr) {
        try {
          const userData = JSON.parse(userDataStr);
          this.user = userData;
          this.accessToken = accessToken;
          this.refreshToken = localStorage.getItem('refresh_token');
          this.isAuthenticated = true;
          this.isGuest = false;
          
          console.log('Loaded authenticated user session:', userData.username);
          return;
        } catch (error) {
          console.error('Failed to parse user data:', error);
        }
      }
      
      // Try to load guest session
      const guestSessionId = localStorage.getItem('guest_session_id');
      const guestSessionToken = localStorage.getItem('guest_session_token');
      
      if (guestSessionId && guestSessionToken) {
        this.guestSessionId = guestSessionId;
        this.guestSessionToken = guestSessionToken;
        this.isAuthenticated = true;
        this.isGuest = true;
        this.guestHistory = [];
        
        console.log('Loaded guest session:', guestSessionId);
        return;
      }
      
      console.log('No persisted session found');
    },
    
    getAuthHeaders() {
      if (this.accessToken) {
        return {
          'Authorization': `Bearer ${this.accessToken}`
        };
      } else if (this.guestSessionToken) {
        return {
          'X-Session-Token': this.guestSessionToken
        };
      }
      return {};
    },
    
    async refreshAccessToken() {
      if (!this.refreshToken) {
        console.log('No refresh token available');
        this.logout();
        return false;
      }
      
      try {
        const response = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.refreshToken}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          this.accessToken = data.access_token;
          localStorage.setItem('access_token', data.access_token);
          
          if (data.refresh_token) {
            this.refreshToken = data.refresh_token;
            localStorage.setItem('refresh_token', data.refresh_token);
          }
          
          console.log('Access token refreshed successfully');
          return true;
        } else {
          console.log('Failed to refresh token, logging out');
          this.logout();
          return false;
        }
      } catch (error) {
        console.error('Error refreshing token:', error);
        this.logout();
        return false;
      }
    },
    
    async updateUserStats(stats: any) {
      if (this.user) {
        this.user = { ...this.user, ...stats };
        localStorage.setItem('user_data', JSON.stringify(this.user));
      }
    }
  }
}) 