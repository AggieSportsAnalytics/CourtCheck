import React, { useState } from 'react';

const PlayerIdentification = ({ onPlayersIdentified }) => {
  const [selectedPlayers, setSelectedPlayers] = useState({
    player1: null,
    player2: null
  });

  // Mock data with actual frames and bounding boxes
  const mockFrames = [
    {
      id: 1,
      url: '/assets/player-frames/player1_frame1.jpg',
      bbox: { x: 100, y: 150, width: 200, height: 400 },
      player: 'Player 1'
    },
    {
      id: 2,
      url: '/assets/player-frames/player1_frame2.jpg',
      bbox: { x: 120, y: 160, width: 180, height: 380 },
      player: 'Player 1'
    },
    {
      id: 3,
      url: '/assets/player-frames/player2_frame1.jpg',
      bbox: { x: 500, y: 140, width: 190, height: 390 },
      player: 'Player 2'
    },
    {
      id: 4,
      url: '/assets/player-frames/player2_frame2.jpg',
      bbox: { x: 480, y: 130, width: 210, height: 410 },
      player: 'Player 2'
    }
  ];

  const handlePlayerSelect = (playerNumber, frameId) => {
    setSelectedPlayers(prev => ({
      ...prev,
      [`player${playerNumber}`]: frameId
    }));
  };

  const handleConfirm = () => {
    if (selectedPlayers.player1 && selectedPlayers.player2) {
      onPlayersIdentified(selectedPlayers);
    }
  };

  const renderFrameWithBoundingBox = (frame) => {
    const { url, bbox, player } = frame;
    return (
      <div className="relative">
        <img
          src={url}
          alt={`${player} Frame`}
          className="w-full h-auto"
        />
        <div
          className="absolute border-2 border-blue-500"
          style={{
            left: `${bbox.x}px`,
            top: `${bbox.y}px`,
            width: `${bbox.width}px`,
            height: `${bbox.height}px`
          }}
        />
      </div>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Identify Players</h2>
        <p className="text-gray-600">Select which player is which from the frames below</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Player 1 Selection */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-700">Player 1</h3>
          <div className="grid grid-cols-2 gap-4">
            {mockFrames.filter(frame => frame.player === 'Player 1').map(frame => (
              <div
                key={frame.id}
                className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all
                  ${selectedPlayers.player1 === frame.id ? 'border-blue-500' : 'border-gray-200'}`}
                onClick={() => handlePlayerSelect(1, frame.id)}
              >
                {renderFrameWithBoundingBox(frame)}
                {selectedPlayers.player1 === frame.id && (
                  <div className="absolute top-2 right-2 bg-blue-500 text-white rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Player 2 Selection */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-700">Player 2</h3>
          <div className="grid grid-cols-2 gap-4">
            {mockFrames.filter(frame => frame.player === 'Player 2').map(frame => (
              <div
                key={frame.id}
                className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all
                  ${selectedPlayers.player2 === frame.id ? 'border-blue-500' : 'border-gray-200'}`}
                onClick={() => handlePlayerSelect(2, frame.id)}
              >
                {renderFrameWithBoundingBox(frame)}
                {selectedPlayers.player2 === frame.id && (
                  <div className="absolute top-2 right-2 bg-blue-500 text-white rounded-full p-1">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8 text-center">
        <button
          onClick={handleConfirm}
          disabled={!selectedPlayers.player1 || !selectedPlayers.player2}
          className={`px-6 py-3 rounded-lg font-semibold text-white transition-colors
            ${selectedPlayers.player1 && selectedPlayers.player2
              ? 'bg-blue-500 hover:bg-blue-600'
              : 'bg-gray-300 cursor-not-allowed'}`}
        >
          Confirm Players
        </button>
      </div>
    </div>
  );
};

export default PlayerIdentification; 