export default function OpponentsPage() {
  return (
    <div className="px-4 py-8">
      <h2 className="text-2xl font-bold text-white mb-4">Opponents</h2>
      <p className="text-gray-400">Track and analyze your performance against different opponents.</p>

      <div className="mt-8 p-8 bg-secondary rounded-lg text-center">
        <div className="text-6xl mb-4">🎯</div>
        <h3 className="text-xl font-semibold text-white mb-2">No Opponent Data Yet</h3>
        <p className="text-gray-400">
          As you upload and analyze matches, opponent statistics will appear here.
        </p>
      </div>
    </div>
  );
}
