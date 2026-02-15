// API Routes
export const API_ROUTES = {
  CREATE_UPLOAD: '/api/create-upload',
  TRIGGER_PROCESS: '/api/trigger-process',
  STATUS: '/api/status',
} as const;

// App Configuration
export const APP_CONFIG = {
  POLL_INTERVAL: 1500, // Poll every 1.5 seconds
  MAX_VIDEO_SIZE: 500 * 1024 * 1024, // 500MB
  SUPPORTED_VIDEO_FORMATS: ['.mp4', '.mov', '.avi'],
} as const;

// Court colors (tennis court visualization)
export const COURT_COLORS = {
  PRIMARY: '#90C641',
  SECONDARY: '#A4D902',
  LINE: '#FFFFFF',
} as const;

// Theme colors
export const THEME_COLORS = {
  PRIMARY: '#11052C',
  SECONDARY: '#182338',
  ACCENT: '#8bc34a',
} as const;
