"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { createBrowserClient } from "@supabase/ssr";
import Link from "next/link";

export default function SettingsPage() {
  const { user, signOut } = useAuth();

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  const currentName = user?.user_metadata?.name || user?.email?.split("@")[0] || "";
  const email = user?.email || "";

  const [displayName, setDisplayName] = useState(currentName);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [confirmSignOut, setConfirmSignOut] = useState(false);

  const handleSaveName = async () => {
    if (!displayName.trim() || displayName.trim() === currentName) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const { error } = await supabase.auth.updateUser({
        data: { name: displayName.trim() },
      });
      if (error) throw error;
      setSaveMsg({ type: "ok", text: "Display name updated." });
    } catch (e) {
      setSaveMsg({ type: "err", text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const initials = currentName
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((p: string) => p[0]?.toUpperCase())
    .join("");

  const cardStyle = {
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.07)',
  } as React.CSSProperties;

  const dividerStyle = {
    borderTop: '1px solid rgba(255,255,255,0.07)',
  } as React.CSSProperties;

  return (
    <div className="px-6 py-8 max-w-2xl">
      {/* Page header */}
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
          Account
        </p>
        <h1 className="text-3xl font-black tracking-tight text-white">Settings</h1>
        <p className="text-sm mt-1" style={{ color: '#5A5A66' }}>Manage your account preferences</p>
      </div>

      {/* Profile section */}
      <div className="rounded-2xl mb-4 overflow-hidden" style={cardStyle}>
        <div className="px-5 py-4" style={dividerStyle}>
          <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#5A5A66' }}>Profile</h3>
        </div>

        {/* Avatar preview */}
        <div className="px-5 py-4 flex items-center gap-4" style={dividerStyle}>
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center text-sm font-bold shrink-0"
            style={{ background: '#B4F000', color: '#07070A' }}
          >
            {initials || "P"}
          </div>
          <div>
            <p className="text-sm text-white font-medium">{currentName}</p>
            <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>{email}</p>
          </div>
        </div>

        {/* Display name */}
        <div className="px-5 py-4" style={dividerStyle}>
          <label className="block text-xs font-medium mb-1.5" style={{ color: '#5A5A66' }}>
            Display Name
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Your name"
              className="flex-1 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none transition-colors"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.1)',
              }}
              onFocus={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'rgba(180,240,0,0.4)'; }}
              onBlur={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.1)'; }}
            />
            <button
              onClick={handleSaveName}
              disabled={saving || !displayName.trim() || displayName.trim() === currentName}
              className="px-4 py-2 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                background: 'rgba(180,240,0,0.1)',
                border: '1px solid rgba(180,240,0,0.2)',
                color: '#B4F000',
              }}
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
          {saveMsg && (
            <p className={`text-xs mt-1.5 ${saveMsg.type === "ok" ? "text-green-400" : "text-red-400"}`}>
              {saveMsg.text}
            </p>
          )}
        </div>

        {/* Email (read only) */}
        <div className="px-5 pb-5" style={dividerStyle}>
          <label className="block text-xs font-medium mb-1.5" style={{ color: '#5A5A66' }}>
            Email <span style={{ color: '#3A3A44' }}>(read only)</span>
          </label>
          <input
            type="email"
            value={email}
            readOnly
            className="w-full rounded-lg px-3 py-2 text-sm cursor-not-allowed"
            style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.06)',
              color: '#5A5A66',
            }}
          />
        </div>
      </div>

      {/* App section */}
      <div className="rounded-2xl mb-4 overflow-hidden" style={cardStyle}>
        <div className="px-5 py-4" style={dividerStyle}>
          <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#5A5A66' }}>App</h3>
        </div>
        <div className="px-5 py-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-white">Dashboard</p>
            <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>Return to your analytics overview</p>
          </div>
          <Link
            href="/"
            className="text-xs font-semibold transition-colors"
            style={{ color: '#B4F000' }}
          >
            Go to dashboard →
          </Link>
        </div>
        <div className="px-5 py-4 flex items-center justify-between" style={dividerStyle}>
          <div>
            <p className="text-sm text-white">Profile</p>
            <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>View your player profile and career stats</p>
          </div>
          <Link
            href="/profile"
            className="text-xs font-semibold transition-colors"
            style={{ color: '#B4F000' }}
          >
            View profile →
          </Link>
        </div>
      </div>

      {/* Account section */}
      <div className="rounded-2xl overflow-hidden" style={cardStyle}>
        <div className="px-5 py-4" style={dividerStyle}>
          <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#5A5A66' }}>Account</h3>
        </div>

        {!confirmSignOut ? (
          <div className="px-5 py-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-white">Sign Out</p>
              <p className="text-xs mt-0.5" style={{ color: '#5A5A66' }}>Sign out of your account on this device</p>
            </div>
            <button
              onClick={() => setConfirmSignOut(true)}
              className="text-xs rounded-lg px-3 py-1.5 transition-colors"
              style={{
                color: '#F87171',
                border: '1px solid rgba(239,68,68,0.2)',
                background: 'transparent',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.06)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
            >
              Sign Out
            </button>
          </div>
        ) : (
          <div className="px-5 py-4">
            <p className="text-sm text-white mb-3">Are you sure you want to sign out?</p>
            <div className="flex gap-2">
              <button
                onClick={signOut}
                className="px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
                style={{
                  background: 'rgba(239,68,68,0.08)',
                  border: '1px solid rgba(239,68,68,0.2)',
                  color: '#F87171',
                }}
              >
                Yes, sign out
              </button>
              <button
                onClick={() => setConfirmSignOut(false)}
                className="px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  color: '#9CA3AF',
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
