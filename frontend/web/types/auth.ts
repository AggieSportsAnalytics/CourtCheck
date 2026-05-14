import type { User } from '@supabase/supabase-js';

export type AuthUser = User;

export interface SignUpMetadata {
  name?: string;
  first_name?: string;
  last_name?: string;
  role?: string;
  team?: string;
  notify_match_ready?: boolean;
}

export interface AuthContextType {
  user: AuthUser | null;
  loading: boolean;
  signUp: (email: string, password: string, metadata?: SignUpMetadata) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}