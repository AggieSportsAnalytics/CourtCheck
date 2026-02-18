import Link from "next/link";

export default function OpponentsPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-6 text-center">
      <div className="max-w-md">
        <div className="w-16 h-16 rounded-2xl bg-gray-800 border border-gray-700/40 flex items-center justify-center mx-auto mb-5">
          <svg viewBox="0 0 24 24" className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
            <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
          </svg>
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Opponent Analysis</h2>
        <p className="text-sm text-gray-400 mb-4 leading-relaxed">
          Opponent tracking requires the ability to distinguish between both players in a match.
          The current pipeline tracks two players but cannot automatically label which is you and
          which is your opponent.
        </p>
        <p className="text-xs text-gray-500 mb-6">
          This feature is planned for a future release. In the meantime, you can view your own
          player stats on the dashboard.
        </p>
        <div className="flex gap-3 justify-center">
          <Link
            href="/"
            className="px-4 py-2 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-xl text-sm text-accent font-medium transition-colors"
          >
            Back to Dashboard
          </Link>
          <Link
            href="/upload"
            className="px-4 py-2 bg-secondary hover:bg-white/5 border border-gray-700/40 rounded-xl text-sm text-gray-300 font-medium transition-colors"
          >
            Analyse a Match
          </Link>
        </div>
      </div>
    </div>
  );
}
