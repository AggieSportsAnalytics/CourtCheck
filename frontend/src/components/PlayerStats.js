import React, { useState, useEffect } from 'react';

const PlayerStats = ({ match }) => {
  const [activePlayer, setActivePlayer] = useState('team1');
  const [randomizedZones, setRandomizedZones] = useState({
    team1: [92, 74, 58, 81, 63, 35],
    team2: [48, 64, 88, 42, 65, 78]
  });
  
  // Base player movement data
  const playerMovementData = {
    team1: {
      distance: 1872,
      sprintCount: 34,
      courtCoverage: 68.4,
      // Base values that will be randomized
      zones: [92, 74, 58, 81, 63, 35]
    },
    team2: {
      distance: 1924,
      sprintCount: 38,
      courtCoverage: 73.1,
      // Base values that will be randomized
      zones: [48, 64, 88, 42, 65, 78]
    }
  };

  // Randomize zone values on mount and when player changes
  useEffect(() => {
    const randomizeZones = () => {
      // Player 1 (Me) - moderate randomization (±15%)
      const team1Random = playerMovementData.team1.zones.map(val => {
        const variance = Math.floor(val * 0.3 * (Math.random() - 0.5));
        return Math.min(100, Math.max(20, val + variance));
      });
      
      // Player 2 (Opponent) - higher randomization (±30%)
      const team2Random = playerMovementData.team2.zones.map(val => {
        // More extreme randomization for opponent
        const variance = Math.floor(val * 0.6 * (Math.random() - 0.5));
        
        // Occasionally make a really extreme value for more visual randomness
        const extremeFactor = Math.random() < 0.2 ? 
          (Math.random() < 0.5 ? 0.8 : 1.3) : 1;
        
        return Math.min(100, Math.max(15, Math.floor(val + variance) * extremeFactor));
      });
      
      setRandomizedZones({
        team1: team1Random,
        team2: team2Random
      });
    };
    
    randomizeZones();
    
    // Randomize more frequently for opponent data
    const intervalId = setInterval(() => {
      if (activePlayer === 'team2') {
        randomizeZones();
      }
    }, 3000); // Faster updates for opponent
    
    // Regular updates regardless of active player
    const intervalIdAll = setInterval(() => {
      randomizeZones();
    }, 8000);
    
    return () => {
      clearInterval(intervalId);
      clearInterval(intervalIdAll);
    };
  }, [activePlayer]);

  // Colors for zone visualization - vibrant palette matching heatmaps
  const zoneColors = {
    team1: '#2C82FF', // Blue
    team2: '#FF5733'  // Red/orange
  };

  const currentPlayerData = match?.stats.playerMovement?.[activePlayer] || playerMovementData[activePlayer];
  const zoneData = randomizedZones[activePlayer];
  
  // Label text for the zones
  const zoneLabels = [
    'Baseline Left', 'Baseline Center', 'Baseline Right',
    'Net Left', 'Net Center', 'Net Right'
  ];
  
  return (
    <div className="bg-gray-800 rounded-xl p-5 shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-bold">Player Movement</h3>
        
        {/* Player Selection */}
        <div className="flex space-x-2">
          <button
            onClick={() => setActivePlayer('team1')}
            className={`px-3 py-1 text-sm rounded-md ${
              activePlayer === 'team1'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {match?.players.team1.name || 'Me'}
          </button>
          <button
            onClick={() => setActivePlayer('team2')}
            className={`px-3 py-1 text-sm rounded-md ${
              activePlayer === 'team2'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {match?.players.team2.name || 'Opponent'}
          </button>
        </div>
      </div>
      
      {/* Player Movement Stats */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Distance</div>
          <div className="text-xl font-bold text-white">{currentPlayerData.totalDistance || currentPlayerData.distance}m</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Sprints</div>
          <div className="text-xl font-bold text-white">{currentPlayerData.sprintCount}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Court Coverage</div>
          <div className="text-xl font-bold text-white">{currentPlayerData.courtCoverage}%</div>
        </div>
      </div>
      
      {/* Player Movement Chart - Court Zones */}
      <div className="mt-4">
        <h4 className="text-sm text-gray-400 mb-2">Zone Coverage</h4>
        <div className="relative h-48 bg-gray-900 rounded-lg p-4 border border-gray-600">
          {/* Court lines overlay */}
          <div className="absolute inset-x-0 top-1/2 h-px bg-gray-500 opacity-50"></div>
          <div className="absolute inset-y-0 left-1/2 w-px bg-gray-500 opacity-50"></div>
          
          {/* Zone bars */}
          <div className="absolute inset-x-0 bottom-0 grid grid-cols-6 h-full px-2 pb-8">
            {zoneData.map((height, index) => (
              <div key={index} className="px-1 flex flex-col items-center justify-end">
                <div 
                  className="w-full rounded-t-md transition-all duration-500 ease-in-out" 
                  style={{ 
                    height: `${height}%`,
                    backgroundColor: zoneColors[activePlayer],
                    opacity: 0.7 + (height / 300), // Makes higher values more vivid
                    boxShadow: `0 0 8px ${zoneColors[activePlayer]}40`
                  }}
                ></div>
              </div>
            ))}
          </div>
          
          {/* Zone labels */}
          <div className="absolute inset-x-0 bottom-1 grid grid-cols-6 px-2 text-center">
            {[1, 2, 3, 4, 5, 6].map(index => (
              <div key={index} className="text-[9px] text-white font-bold">
                Z{index}
              </div>
            ))}
          </div>
          
          {/* Court outline overlay */}
          <div className="absolute inset-[8%] border border-gray-500 opacity-30 pointer-events-none">
            <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-gray-500"></div>
            <div className="absolute top-1/2 left-0 right-0 h-[1px] bg-gray-500"></div>
          </div>
        </div>
        
        {/* Zone key */}
        <div className="mt-3 bg-gray-700 p-2 rounded-md">
          <div className="text-xs text-gray-300 font-medium mb-1">Zone Key:</div>
          <div className="grid grid-cols-3 gap-1 text-[10px] text-gray-400">
            {zoneLabels.map((label, i) => (
              <div key={i} className="flex items-center">
                <span className="mr-1 font-bold text-white">Z{i+1}:</span> {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlayerStats; 