import React, { useState } from 'react';

const HeatMaps = () => {
  const [activeTab, setActiveTab] = useState('shots');
  const [activeSet, setActiveSet] = useState('set1');

  // Mock data - in a real app these would be dynamic images or generated SVGs
  const courtImageEmpty = "https://i.ibb.co/s38xVyK/empty-court.png";
  const courtImageHeatmap = "https://i.ibb.co/GQ5XqLh/heatmap-court.png";
  
  return (
    <div className="bg-secondary rounded-xl overflow-hidden">
      <div className="p-4">
        <h3 className="text-xl font-bold mb-4">Emma's Heat Maps!</h3>
        
        {/* Tabs */}
        <div className="flex gap-2 mb-4">
          <button 
            className={`px-6 py-2 rounded-md ${activeTab === 'shots' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveTab('shots')}
          >
            Shots
          </button>
          <button 
            className={`px-6 py-2 rounded-md ${activeTab === 'player' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveTab('player')}
          >
            Player
          </button>
        </div>
        
        {/* Set Selection */}
        <div className="grid grid-cols-4 gap-1 mb-4">
          <button 
            className={`py-2 rounded-md ${activeSet === 'set1' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set1')}
          >
            Set 1
          </button>
          <button 
            className={`py-2 rounded-md ${activeSet === 'set2' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set2')}
          >
            Set 2
          </button>
          <button 
            className={`py-2 rounded-md ${activeSet === 'set3' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('set3')}
          >
            Set 3
          </button>
          <button 
            className={`py-2 rounded-md ${activeSet === 'overall' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveSet('overall')}
          >
            Overall
          </button>
        </div>
      </div>
      
      {/* Court Visualization */}
      <div className="p-4 flex justify-center">
        <div className="relative w-full">
          <img 
            src={activeTab === 'shots' ? courtImageEmpty : courtImageHeatmap} 
            alt="Tennis Court Visualization" 
            className="w-full rounded-lg"
          />
          
          {/* If showing the shots tab with empty court, we could add markers for specific shots */}
          {activeTab === 'shots' && (
            <>
              {/* These are placeholders for shot markers - in a real app you'd map over real shot data */}
              <div className="absolute top-[30%] left-[20%] w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute top-[45%] left-[35%] w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute top-[60%] left-[70%] w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute top-[25%] left-[80%] w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute top-[70%] left-[25%] w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
            </>
          )}
        </div>
      </div>
      
      {/* Second Court Image - For Bottom Heat Map in UI */}
      <div className="p-4 flex justify-center">
        <div className="relative w-full">
          <img 
            src={courtImageHeatmap} 
            alt="Tennis Court Heat Map" 
            className="w-full rounded-lg"
          />
        </div>
      </div>
    </div>
  );
};

export default HeatMaps; 