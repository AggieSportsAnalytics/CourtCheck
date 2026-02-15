'use client';

import { useEffect, useState } from 'react';

const PlayerStats = () => {
  const [activePlayer, setActivePlayer] = useState<'player1' | 'player2'>('player1');
  const [hasRecordings, setHasRecordings] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchHasRecordings() {
      try {
        const res = await fetch('/api/recordings');
        if (!res.ok) {
          if (!cancelled) setHasRecordings(false);
          return;
        }
        const data = await res.json();
        const recordings = Array.isArray(data?.recordings) ? data.recordings : [];
        if (!cancelled) setHasRecordings(recordings.length > 0);
      } catch {
        if (!cancelled) setHasRecordings(false);
      }
    }
    fetchHasRecordings();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="bg-white rounded-xl p-4">
      <h3 className="text-xl font-bold mb-4 text-gray-800">Player-Movement</h3>

      {/* Player Selection */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <button
          className={`py-2 rounded-md ${activePlayer === 'player1' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActivePlayer('player1')}
          aria-label="View stats for Game 1 Player"
        >
          Game 1 Player
        </button>
        <button
          className={`py-2 rounded-md ${activePlayer === 'player2' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActivePlayer('player2')}
          aria-label="View stats for Game 2 Player"
        >
          Game 2 Player
        </button>
      </div>

      {/* Player Movement Chart */}
      <div className="mt-4">
        {hasRecordings === false ? (
          <div className="h-56 bg-gray-50 rounded-lg flex items-center justify-center px-4 text-center">
            <p className="text-sm text-gray-600">Upload a video first to see player movement stats.</p>
          </div>
        ) : (
          <div className="relative">
            {/* This would be an actual chart in a real application */}
            <div className="h-56 bg-gray-50 rounded-lg flex items-end px-2">
              {/* Mock bar chart for player movement */}
              <div className="w-1/6 h-[45%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
              <div className="w-1/6 h-[65%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
              <div className="w-1/6 h-[55%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
              <div className="w-1/6 h-[40%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
              <div className="w-1/6 h-[75%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
              <div className="w-1/6 h-[60%] mx-1 bg-accent rounded-t-md" aria-hidden="true"></div>
            </div>

            {/* X axis labels */}
            <div className="flex justify-between mt-2 px-2">
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
            </div>

            {/* Y axis labels */}
            <div className="absolute left-0 top-0 h-full flex flex-col justify-between py-2">
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
              <div className="text-xs text-gray-600">X</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PlayerStats;
