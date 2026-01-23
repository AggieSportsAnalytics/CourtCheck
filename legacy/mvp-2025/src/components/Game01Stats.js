import React, { useState } from 'react';

const Game01Stats = () => {
  const [activeTab, setActiveTab] = useState('shots');
  const [activeSet, setActiveSet] = useState('set1');
  const [activeTab2, setActiveTab2] = useState('player');
  const [activeSet2, setActiveSet2] = useState('set1');

  return (
    <div className="px-4 grid gap-4 pb-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Column - Heat Maps Section */}
        <div className="grid gap-4">
          {/* First Heat Map */}
          <div className="bg-white rounded-3xl p-6">
            <div className="space-y-4">
              {/* Header and Tabs Row */}
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold text-gray-800">Heat Maps</h3>
                <div className="flex gap-2">
                  <button 
                    className={`px-8 py-2 text-sm font-medium rounded-full transition-colors ${
                      activeTab === 'shots' ? 'bg-gray-400 text-white' : 'bg-gray-200 text-gray-600'
                    }`}
                    onClick={() => setActiveTab('shots')}
                  >
                    Shots
                  </button>
                  <button 
                    className={`px-8 py-2 text-sm font-medium rounded-full transition-colors ${
                      activeTab === 'player' ? 'bg-gray-400 text-white' : 'bg-gray-200 text-gray-600'
                    }`}
                    onClick={() => setActiveTab('player')}
                  >
                    Player
                  </button>
                </div>
              </div>
              
              {/* Set Selection Tabs */}
              <div className="grid grid-cols-4 gap-2">
                {['Set 1', 'Set 2', 'Set 3', 'Overall'].map((set, index) => (
                  <button
                    key={set}
                    className={`py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeSet === `set${index + 1}` ? 'bg-gray-200 text-gray-700' : 'bg-gray-100 text-gray-600'
                    }`}
                    onClick={() => setActiveSet(`set${index + 1}`)}
                  >
                    {set}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Tennis Court */}
            <div className="mt-4">
              <div className="bg-[#90C641] rounded-2xl p-3">
                <div className="w-full aspect-[1.6] bg-[#A4D902] rounded-xl relative">
                  {/* Court outline */}
                  <div className="absolute inset-[8%] border-[2px] border-white">
                    {/* Center line */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-[2px] bg-white"></div>
                    
                    {/* Service line */}
                    <div className="absolute top-[30%] left-0 right-0 h-[2px] bg-white"></div>
                    
                    {/* Service boxes */}
                    <div className="absolute top-[30%] left-[25%] h-[40%] w-[2px] bg-white"></div>
                    <div className="absolute top-[30%] left-[75%] h-[40%] w-[2px] bg-white"></div>
                    <div className="absolute top-[70%] left-0 right-0 h-[2px] bg-white"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Second Heat Map */}
          <div className="bg-white rounded-3xl p-6">
            <div className="space-y-4">
              {/* Header and Tabs Row */}
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold text-gray-800">Heat Maps</h3>
                <div className="flex gap-2">
                  <button 
                    className="px-8 py-2 text-sm font-medium rounded-full transition-colors bg-gray-200 text-gray-600"
                    onClick={() => setActiveTab2('shots')}
                  >
                    Shots
                  </button>
                  <button 
                    className="px-8 py-2 text-sm font-medium rounded-full transition-colors bg-gray-400 text-white"
                    onClick={() => setActiveTab2('player')}
                  >
                    Player
                  </button>
                </div>
              </div>
              
              {/* Set Selection Tabs */}
              <div className="grid grid-cols-4 gap-2">
                {['Set 1', 'Set 2', 'Set 3', 'Overall'].map((set, index) => (
                  <button
                    key={set}
                    className={`py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeSet2 === `set${index + 1}` ? 'bg-gray-200 text-gray-700' : 'bg-gray-100 text-gray-600'
                    }`}
                    onClick={() => setActiveSet2(`set${index + 1}`)}
                  >
                    {set}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Tennis Court */}
            <div className="mt-4">
              <div className="bg-[#90C641] rounded-2xl p-3">
                <div className="w-full aspect-[1.6] bg-[#A4D902] rounded-xl relative">
                  {/* Court outline */}
                  <div className="absolute inset-[8%] border-[2px] border-white">
                    {/* Center line */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-[2px] bg-white"></div>
                    
                    {/* Service line */}
                    <div className="absolute top-[30%] left-0 right-0 h-[2px] bg-white"></div>
                    
                    {/* Service boxes */}
                    <div className="absolute top-[30%] left-[25%] h-[40%] w-[2px] bg-white"></div>
                    <div className="absolute top-[30%] left-[75%] h-[40%] w-[2px] bg-white"></div>
                    <div className="absolute top-[70%] left-0 right-0 h-[2px] bg-white"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="grid gap-4">
          {/* Player Movement Section */}
          <div className="bg-white rounded-xl p-4">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Player-Movement</h3>
            <div className="grid grid-cols-2 gap-2 mb-4">
              <button
                className="py-2 rounded-md bg-gray-200 text-gray-800"
              >
                Cory_Pham_22
              </button>
              <button
                className="py-2 rounded-md bg-gray-100 text-gray-600"
              >
                Game_01_Opponent
              </button>
            </div>
            
            {/* Bar Chart */}
            <div className="h-56 bg-gray-50 rounded-lg mt-4 relative">
              <div className="absolute inset-x-0 bottom-0 flex items-end justify-around h-full px-2">
                {[45, 65, 55, 40, 75, 60].map((height, index) => (
                  <div key={index} className="w-1/6 mx-1">
                    <div 
                      className="bg-accent rounded-t-md" 
                      style={{ height: `${height}%` }}
                    ></div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Shot Percentages Section */}
          <div className="bg-white rounded-xl p-4">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Shot Percentages</h3>
            <div className="grid grid-cols-4 gap-1 mb-4">
              {['Set 1', 'Set 2', 'Set 3', 'Overall'].map((set) => (
                <button
                  key={set}
                  className={`py-2 rounded-md ${
                    set === 'Set 1' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {set}
                </button>
              ))}
            </div>
            
            {/* Pie Chart */}
            <div className="flex justify-center mt-4">
              <div className="relative w-48 h-48">
                <div className="w-full h-full rounded-full border-4 border-gray-200"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xl font-bold text-gray-800">100%</span>
                </div>
              </div>
            </div>
            
            {/* Legend */}
            <div className="grid grid-cols-2 gap-4 mt-6">
              {[
                { label: 'Forehand', value: '22%', color: '#a4de02' },
                { label: 'Backhand', value: '28%', color: '#436b22' },
                { label: 'Slices', value: '9%', color: '#acc260' },
                { label: 'Lob-Shots', value: '16%', color: '#225a28' },
                { label: 'Drop-Shots', value: '25%', color: '#d4eb8f' },
              ].map((item) => (
                <div key={item.label} className="flex items-center">
                  <div 
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-sm text-gray-800">{item.label}</span>
                  <span className="text-sm text-gray-600 ml-auto">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Game01Stats; 