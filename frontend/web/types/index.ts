// User types
export interface User {
  name: string;
  id: string;
  image: string;
}

// Navigation types
export type NavigationItem =
  | 'Dashboard'
  | 'Upload Video'
  | 'Match Stats'
  | 'Overall Stats'
  | 'Opponents'
  | 'Recordings'
  | 'Settings';

// Game types
export type GameId = 'Game_01' | 'Game_02' | 'Game_03';

// Video processing types
export interface VideoProcessingStatus {
  match_id: string;
  status: 'pending' | 'processing' | 'done' | 'failed';
  progress: number;
  error?: string;
  videoUrl?: string;
  fps?: number;
  num_frames?: number;
}

// Stats types
export interface PlayerDetection {
  track_id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface BallPosition {
  x: number | null;
  y: number | null;
  frame: number;
}

export interface CourtKeypoint {
  x: number;
  y: number;
  visible: boolean;
}

// Set selection types
export type SetSelection = 'set1' | 'set2' | 'set3' | 'overall';

// Tab types
export type TabSelection = 'shots' | 'player';
