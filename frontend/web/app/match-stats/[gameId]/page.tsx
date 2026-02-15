import Game01Stats from '@/components/features/match/Game01Stats';

interface MatchStatsPageProps {
  params: {
    gameId: string;
  };
}

export default function MatchStatsPage({ params }: MatchStatsPageProps) {
  const { gameId } = params;

  return (
    <div>
      {gameId === 'Game_01' ? (
        <Game01Stats />
      ) : (
        <div className="px-4 py-8">
          <h2 className="text-2xl font-bold text-white">Match Stats - {gameId}</h2>
          <p className="text-gray-400 mt-2">Statistics for this game will be available soon.</p>
        </div>
      )}
    </div>
  );
}
