import React, { useState } from 'react';
import Logo from './Logo';

const SELECTION_IMAGES = Array.from({ length: 10 }, (_, i) => ({
  src: `/assets/player-frames/track${i + 1}.png`,
  index: i + 1
}));

const CONFIRMATION_IMAGES = [
  ...Array.from({ length: 10 }, (_, i) => ({
    src: `/assets/player-frames/confirm${i + 1}_player1.png`,
    player: 'Player 1',
    index: i + 1
  })),
  ...Array.from({ length: 10 }, (_, i) => ({
    src: `/assets/player-frames/confirm${i + 1}_player2.png`,
    player: 'Player 2',
    index: i + 1
  }))
];

const PlayerSelectionFlow = ({ onComplete }) => {
  const [phase, setPhase] = useState('selection'); // 'selection' | 'confirmation'
  const [currentSelectionIdx, setCurrentSelectionIdx] = useState(0);
  const [selections, setSelections] = useState([]); // [{p1: id, p2: id}]
  const [confirmed, setConfirmed] = useState(false);

  // Selection phase handlers
  const handleNext = () => {
    if (currentSelectionIdx < SELECTION_IMAGES.length - 1) {
      setCurrentSelectionIdx(currentSelectionIdx + 1);
    } else {
      setPhase('confirmation');
    }
  };

  // Confirmation phase
  const handleConfirm = () => {
    setConfirmed(true);
    if (onComplete) onComplete(selections);
  };

  // Selection phase UI
  if (phase === 'selection') {
    // Show all 10 images in a vertical list, each with two input boxes below
    return (
      <div className="w-full max-w-4xl mx-auto p-6">
        <div className="mb-6 bg-blue-50 rounded-xl p-4 shadow">
          <Logo size="xl" className="mx-auto mb-3" />
          <h2 className="text-3xl font-extrabold text-blue-900 mb-2 text-center drop-shadow">Player Selection (All Frames)</h2>
          <p className="text-blue-700 text-center text-lg">Enter the Temp IDs for each player as shown in each image below. Fill all to continue.</p>
        </div>
        <div className="flex flex-col gap-12 mb-8">
          {SELECTION_IMAGES.map((img, idx) => {
            const current = selections[idx] || { p1: '', p2: '' };
            return (
              <div key={img.src} className="flex flex-col items-center bg-white/90 rounded-2xl shadow-2xl p-8">
                <img
                  src={img.src}
                  alt={`Selection ${img.index}`}
                  className="rounded-2xl border-4 border-blue-300 shadow-xl mb-6 w-full object-contain bg-white"
                  style={{ maxWidth: 700, maxHeight: 480, width: '90vw', height: 'auto' }}
                />
                <div className="flex flex-col md:flex-row gap-8 w-full justify-center mb-2">
                  <div className="flex flex-col items-center w-full">
                    <label className="font-semibold text-blue-800 mb-2 text-2xl">Player 1 Temp ID</label>
                    <input
                      type="text"
                      className="border-2 border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 rounded-xl px-6 py-4 text-center w-full text-2xl font-bold text-gray-900 bg-white transition-all duration-150"
                      value={current.p1}
                      onChange={e => {
                        const newSelections = [...selections];
                        newSelections[idx] = { ...current, p1: e.target.value };
                        setSelections(newSelections);
                      }}
                      placeholder="e.g. 0"
                    />
                  </div>
                  <div className="flex flex-col items-center w-full">
                    <label className="font-semibold text-blue-800 mb-2 text-2xl">Player 2 Temp ID</label>
                    <input
                      type="text"
                      className="border-2 border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 rounded-xl px-6 py-4 text-center w-full text-2xl font-bold text-gray-900 bg-white transition-all duration-150"
                      value={current.p2}
                      onChange={e => {
                        const newSelections = [...selections];
                        newSelections[idx] = { ...current, p2: e.target.value };
                        setSelections(newSelections);
                      }}
                      placeholder="e.g. 3"
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="text-center mt-8">
          <button
            className="px-12 py-5 bg-blue-600 text-white rounded-2xl font-extrabold text-2xl shadow-lg hover:bg-blue-700 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={selections.length !== 10 || selections.some(sel => !sel?.p1 || !sel?.p2)}
            onClick={() => setPhase('confirmation')}
          >
            Review Selections
          </button>
        </div>
      </div>
    );
  }

  // Confirmation phase UI
  if (phase === 'confirmation') {
    return (
      <div className="w-full max-w-7xl mx-auto p-4 md:p-6">
        <div className="mb-4 md:mb-6 bg-green-50 rounded-xl p-3 md:p-4 shadow">
          <Logo size="lg" className="mx-auto mb-2" />
          <h2 className="text-2xl md:text-3xl font-bold text-green-900 mb-1 md:mb-2 text-center drop-shadow">Confirm Your Player Selections</h2>
          <p className="text-green-700 text-center text-base md:text-lg">Review your selections for each player below. If correct, confirm to continue.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-12">
          {/* Player 1 Confirmations */}
          <div>
            <h3 className="text-lg md:text-xl font-semibold text-blue-700 mb-3 md:mb-4 text-center tracking-wide">Player 1</h3>
            <div className="flex flex-col gap-6 md:gap-8">
              {CONFIRMATION_IMAGES.filter(img => img.player === 'Player 1').map((img, idx) => (
                <div key={img.src} className="relative rounded-xl overflow-hidden border border-blue-200 shadow bg-white flex flex-col items-center p-2 md:p-4">
                  <img src={img.src} alt={`Confirm P1 ${img.index}`} className="w-full object-contain rounded-md" style={{ maxWidth: 700, width: '100%', height: 'auto' }} />
                  <div className="absolute top-2 left-2 bg-blue-600/90 text-white text-xs md:text-sm px-3 py-1 rounded shadow font-semibold tracking-wide">
                    ID: <span className="font-bold">{selections[idx]?.p1 || '?'}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          {/* Player 2 Confirmations */}
          <div>
            <h3 className="text-lg md:text-xl font-semibold text-purple-700 mb-3 md:mb-4 text-center tracking-wide">Player 2</h3>
            <div className="flex flex-col gap-6 md:gap-8">
              {CONFIRMATION_IMAGES.filter(img => img.player === 'Player 2').map((img, idx) => (
                <div key={img.src} className="relative rounded-xl overflow-hidden border border-purple-200 shadow bg-white flex flex-col items-center p-2 md:p-4">
                  <img src={img.src} alt={`Confirm P2 ${img.index}`} className="w-full object-contain rounded-md" style={{ maxWidth: 700, width: '100%', height: 'auto' }} />
                  <div className="absolute top-2 left-2 bg-purple-600/90 text-white text-xs md:text-sm px-3 py-1 rounded shadow font-semibold tracking-wide">
                    ID: <span className="font-bold">{selections[idx]?.p2 || '?'}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-8 md:mt-12 text-center">
          <button
            className="px-8 py-3 md:px-12 md:py-5 bg-green-600 text-white rounded-xl font-bold text-lg md:text-2xl shadow-lg hover:bg-green-700 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleConfirm}
            disabled={confirmed}
          >
            {confirmed ? 'Selections Confirmed!' : 'Confirm and Continue'}
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default PlayerSelectionFlow; 