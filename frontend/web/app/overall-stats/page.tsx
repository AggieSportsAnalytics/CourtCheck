export default function OverallStatsPage() {
  return (
    <div className="px-4 py-8">
      <h2 className="text-2xl font-bold text-white mb-4">Overall Stats</h2>
      <p className="text-gray-400">Your overall performance statistics across all matches.</p>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-secondary p-6 rounded-lg">
          <div className="text-3xl font-bold text-accent mb-2">0</div>
          <div className="text-sm text-gray-400">Total Matches</div>
        </div>
        <div className="bg-secondary p-6 rounded-lg">
          <div className="text-3xl font-bold text-accent mb-2">0%</div>
          <div className="text-sm text-gray-400">Win Rate</div>
        </div>
        <div className="bg-secondary p-6 rounded-lg">
          <div className="text-3xl font-bold text-accent mb-2">0</div>
          <div className="text-sm text-gray-400">Hours Played</div>
        </div>
      </div>

      <div className="mt-8 p-8 bg-secondary rounded-lg text-center">
        <div className="text-6xl mb-4">📊</div>
        <h3 className="text-xl font-semibold text-white mb-2">Start Tracking Your Progress</h3>
        <p className="text-gray-400">
          Upload and analyze your matches to see detailed statistics.
        </p>
      </div>
    </div>
  );
}
