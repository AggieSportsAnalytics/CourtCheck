import React, { useState } from 'react';

// Court image path
const COURT_IMAGE_PATH = './assets/court_skeleton.png';
// Player movement heatmap images
const PLAYER1_MOVEMENT_HEATMAP = './assets/player1_movement_heatmap.png';
const PLAYER2_MOVEMENT_HEATMAP = './assets/player2_movement_heatmap.png';
// Player bounce heatmap images
const PLAYER1_BOUNCE_HEATMAP = './assets/player1_heatmap_bounce.png';
const PLAYER2_BOUNCE_HEATMAP = './assets/player2_heatmap_bounce.png';

const HeatMaps = ({ match }) => {
  const [activeTab, setActiveTab] = useState('shots'); // 'shots' or 'player'
  const [activeSet, setActiveSet] = useState('overall'); // 'set1', 'set2', 'set3', 'overall'
  const [activePlayer, setActivePlayer] = useState('team1'); // 'team1' or 'team2'
  const [activeMapType, setActiveMapType] = useState('bounces'); // 'bounces' or 'movement'
  const [imageError, setImageError] = useState(false);

  // Get the appropriate image based on selection
  const getHeatmapImage = () => {
    // For "Me" (team1) and "Opponent" (team2)
    if (activePlayer === 'team1') {
      // For team1 ("Me")
      return activeTab === 'shots' ? PLAYER1_BOUNCE_HEATMAP : PLAYER1_MOVEMENT_HEATMAP;
    } else {
      // For team2 ("Opponent")
      return activeTab === 'shots' ? PLAYER2_BOUNCE_HEATMAP : PLAYER2_MOVEMENT_HEATMAP;
    }
  };

  // Simple fallback court
  const renderFallbackCourt = () => (
    <div className="w-full aspect-video bg-[#8CB369] rounded-lg relative">
      <div className="absolute inset-[5%] border-[2px] border-white">
        <div className="absolute top-0 bottom-0 left-1/2 w-[2px] bg-white"></div>
        <div className="absolute top-[30%] left-0 right-0 h-[2px] bg-white"></div>
        <div className="absolute top-[30%] left-[25%] h-[40%] w-[2px] bg-white"></div>
        <div className="absolute top-[30%] left-[75%] h-[40%] w-[2px] bg-white"></div>
        <div className="absolute top-[70%] left-0 right-0 h-[2px] bg-white"></div>
      </div>
    </div>
  );

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden shadow-md">
      <div className="p-5">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold text-white">Heat Maps</h3>
          
          {/* Primary Tabs: Shots vs Player */}
          <div className="flex space-x-2">
            <button 
              className={`px-6 py-2 rounded-md text-sm ${activeTab === 'shots' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
              onClick={() => setActiveTab('shots')}
            >
              Shots
            </button>
            <button 
              className={`px-6 py-2 rounded-md text-sm ${activeTab === 'player' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
              onClick={() => setActiveTab('player')}
            >
              Player
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
        
        {/* Player Selection (for both tabs) */}
        <div className="grid grid-cols-2 gap-2 mb-4">
          <button
            className={`py-2 rounded-md text-sm ${activePlayer === 'team1' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
            onClick={() => {
              setActivePlayer('team1');
              setActiveMapType(activeTab === 'shots' ? 'bounces' : 'movement');
            }}
          >
            {match?.players.team1.name || 'Me'}
          </button>
          <button
            className={`py-2 rounded-md text-sm ${activePlayer === 'team2' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
            onClick={() => {
              setActivePlayer('team2');
              setActiveMapType(activeTab === 'shots' ? 'bounces' : 'movement');
            }}
          >
            {match?.players.team2.name || 'Opponent'}
          </button>
        </div>
      </div>
      
      {/* Tennis Court with Heat Map */}
      <div className="p-4 flex justify-center">
        <div className="relative w-full max-w-3xl mx-auto">
          {/* Tennis Court with Heatmap */}
          <div className="rounded-lg overflow-hidden border border-white">
            {imageError ? (
              renderFallbackCourt()
            ) : (
              <div className="w-full aspect-video bg-gray-900 rounded-lg relative">
                <img 
                  src={getHeatmapImage()} 
                  className="w-full h-full object-contain"
                  onError={() => setImageError(true)}
                  alt={`${activeTab === 'shots' ? 'Ball bounce' : activePlayer === 'team1' ? 'Player 1' : 'Player 2'} heatmap`}
                />
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="p-5">
        <div className="text-gray-400 text-sm">
          {imageError ? (
            <p className="italic">Note: Using placeholder tennis court. Heat map visualization could not be loaded.</p>
          ) : activeTab === 'shots' ? (
            <p>Heat map showing {activePlayer === 'team1' ? match?.players.team1.name || 'Me' : match?.players.team2.name || 'Opponent'}'s ball bounce distribution during {activeSet === 'overall' ? 'the entire match' : activeSet.replace('set', 'set ')}.</p>
          ) : (
            <p>Heat map showing {activePlayer === 'team1' ? match?.players.team1.name || 'Me' : match?.players.team2.name || 'Opponent'}'s movement patterns during {activeSet === 'overall' ? 'the entire match' : activeSet.replace('set', 'set ')}.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default HeatMaps; 