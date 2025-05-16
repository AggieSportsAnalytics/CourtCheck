import React, { useState } from 'react';

const ShotPercentages = ({ match }) => {
  const [activeSet, setActiveSet] = useState('overall'); // 'set1', 'set2', 'set3', 'overall'
  const [activePlayer, setActivePlayer] = useState('team1'); // 'team1' or 'team2'

  // Mock data for shot percentages - in a real app, this would come from your match data
  const shotData = {
    team1: {
      forehand: 25,
      backhand: 28,
      slices: 22,
      lob: 9,
      drop: 16
    },
    team2: {
      forehand: 18,
      backhand: 35,
      slices: 15,
      lob: 12,
      drop: 20
    }
  };

  // Updated colors for the shot types - more diverse and vibrant
  const shotColors = {
    forehand: '#FF5733', // Bright coral
    backhand: '#337DFF', // Bright blue
    slices: '#FFBD33',   // Amber
    lob: '#9333FF',      // Purple
    drop: '#33FF57'      // Bright green
  };

  // Get the current player's shot data
  const currentPlayerShots = shotData[activePlayer];

  return (
    <div className="bg-gray-800 rounded-xl p-5 shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-bold">Shot Percentages</h3>
        
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
      
      {/* Set Selection */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <button
          className={`py-2 rounded-md text-sm ${activeSet === 'set1' ? 'bg-gray-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
          onClick={() => setActiveSet('set1')}
        >
          Set 1
        </button>
        <button
          className={`py-2 rounded-md text-sm ${activeSet === 'set2' ? 'bg-gray-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
          onClick={() => setActiveSet('set2')}
        >
          Set 2
        </button>
        <button
          className={`py-2 rounded-md text-sm ${activeSet === 'set3' ? 'bg-gray-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
          onClick={() => setActiveSet('set3')}
        >
          Set 3
        </button>
        <button
          className={`py-2 rounded-md text-sm ${activeSet === 'overall' ? 'bg-gray-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
          onClick={() => setActiveSet('overall')}
        >
          Overall
        </button>
      </div>
      
      {/* Pie Chart */}
      <div className="flex items-center justify-center mt-6">
        <div className="relative w-48 h-48">
          {/* Simple CSS pie chart */}
          <div className="absolute inset-0 rounded-full overflow-hidden">
            {/* This simplified pie chart uses CSS conic-gradient */}
            <div 
              className="w-full h-full"
              style={{
                background: `conic-gradient(
                  ${shotColors.forehand} 0% ${currentPlayerShots.forehand}%, 
                  ${shotColors.backhand} ${currentPlayerShots.forehand}% ${currentPlayerShots.forehand + currentPlayerShots.backhand}%, 
                  ${shotColors.slices} ${currentPlayerShots.forehand + currentPlayerShots.backhand}% ${currentPlayerShots.forehand + currentPlayerShots.backhand + currentPlayerShots.slices}%, 
                  ${shotColors.lob} ${currentPlayerShots.forehand + currentPlayerShots.backhand + currentPlayerShots.slices}% ${currentPlayerShots.forehand + currentPlayerShots.backhand + currentPlayerShots.slices + currentPlayerShots.lob}%, 
                  ${shotColors.drop} ${currentPlayerShots.forehand + currentPlayerShots.backhand + currentPlayerShots.slices + currentPlayerShots.lob}% 100%
                )`
              }}
            ></div>
          </div>
          
          {/* Center circle */}
          <div className="absolute w-24 h-24 bg-gray-800 rounded-full top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center border-4 border-gray-700">
            <span className="text-white font-bold">100%</span>
          </div>
        </div>
      </div>
      
      {/* Legend */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-6">
        {Object.entries(currentPlayerShots).map(([type, percentage]) => (
          <div key={type} className="flex items-center">
            <div 
              className="w-3 h-3 rounded-full mr-2" 
              style={{ backgroundColor: shotColors[type] }}
            ></div>
            <span className="text-sm capitalize text-gray-300">{type}</span>
            <span className="text-sm ml-auto text-gray-400">{percentage}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ShotPercentages; 