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

  return (
    <div className="px-5 py-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Settings</h2>
        <p className="text-sm text-gray-400 mt-1">Manage your account preferences</p>
      </div>

      {/* Profile section */}
      <div className="bg-secondary rounded-2xl border border-gray-700/40 mb-4 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-700/40">
          <h3 className="text-sm font-semibold text-white">Profile</h3>
        </div>

        {/* Avatar preview */}
        <div className="px-5 py-4 flex items-center gap-4 border-b border-gray-700/40">
          <div className="w-12 h-12 rounded-full bg-accent/20 border-2 border-accent/30 flex items-center justify-center text-accent font-bold text-base shrink-0">
            {initials || "P"}
          </div>
          <div>
            <p className="text-sm text-white font-medium">{currentName}</p>
            <p className="text-xs text-gray-400">{email}</p>
          </div>
        </div>

        {/* Display name */}
        <div className="px-5 py-4">
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Display Name
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Your name"
              className="flex-1 bg-primary border border-gray-700/60 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent/60 transition-colors"
            />
            <button
              onClick={handleSaveName}
              disabled={saving || !displayName.trim() || displayName.trim() === currentName}
              className="px-4 py-2 bg-accent/10 hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed border border-accent/20 rounded-lg text-sm text-accent font-medium transition-colors"
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
        <div className="px-5 pb-5">
          <label className="block text-xs font-medium text-gray-400 mb-1.5">
            Email <span className="text-gray-600">(read only)</span>
          </label>
          <input
            type="email"
            value={email}
            readOnly
            className="w-full bg-primary/50 border border-gray-700/40 rounded-lg px-3 py-2 text-sm text-gray-400 cursor-not-allowed"
          />
        </div>
      </div>

      {/* App section */}
      <div className="bg-secondary rounded-2xl border border-gray-700/40 mb-4 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-700/40">
          <h3 className="text-sm font-semibold text-white">App</h3>
        </div>
        <div className="px-5 py-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-white">Dashboard</p>
            <p className="text-xs text-gray-500">Return to your analytics overview</p>
          </div>
          <Link
            href="/"
            className="text-xs text-accent hover:underline"
          >
            Go to dashboard →
          </Link>
        </div>
        <div className="px-5 pb-4 flex items-center justify-between border-t border-gray-700/40">
          <div className="pt-4">
            <p className="text-sm text-white">Profile</p>
            <p className="text-xs text-gray-500">View your player profile and career stats</p>
          </div>
          <Link
            href="/profile"
            className="text-xs text-accent hover:underline"
          >
            View profile →
          </Link>
        </div>
      </div>

      {/* Account section */}
      <div className="bg-secondary rounded-2xl border border-gray-700/40 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-700/40">
          <h3 className="text-sm font-semibold text-white">Account</h3>
        </div>

        {!confirmSignOut ? (
          <div className="px-5 py-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-white">Sign Out</p>
              <p className="text-xs text-gray-500">Sign out of your account on this device</p>
            </div>
            <button
              onClick={() => setConfirmSignOut(true)}
              className="text-xs text-red-400 hover:text-red-300 border border-red-800/40 hover:border-red-700/60 px-3 py-1.5 rounded-lg transition-colors"
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
                className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-sm text-red-400 font-medium transition-colors"
              >
                Yes, sign out
              </button>
              <button
                onClick={() => setConfirmSignOut(false)}
                className="px-4 py-2 bg-gray-700/40 hover:bg-gray-700/60 border border-gray-700/40 rounded-lg text-sm text-gray-300 font-medium transition-colors"
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
