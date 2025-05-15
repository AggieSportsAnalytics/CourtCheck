import React, { useState } from 'react';

const GamePlay = () => {
  const [activeTab, setActiveTab] = useState('stats');
  const [timeframe, setTimeframe] = useState('30');
  
  return (
    <div className="bg-secondary rounded-xl p-4">
      <h2 className="text-xl font-bold">You Recorded 5.89 Hrs of Game-Play This Month!</h2>
      
      <div className="flex items-center justify-between mt-4">
        <div className="flex items-center">
          <span className="text-sm">Last {timeframe} days</span>
          <button className="ml-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
        
        <div className="flex gap-4">
          <button 
            className={`px-4 py-1 rounded-md text-sm ${activeTab === 'stats' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveTab('stats')}
          >
            Stats
          </button>
          <button 
            className={`px-4 py-1 rounded-md text-sm ${activeTab === 'history' ? 'bg-gray-600' : 'bg-gray-800'}`}
            onClick={() => setActiveTab('history')}
          >
            History
          </button>
        </div>
      </div>
      
      {activeTab === 'stats' && (
        <div className="mt-6">
          {/* Bar chart visualization */}
          <div className="relative h-48 flex items-end">
            {/* Y-axis labels */}
            <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-gray-400 text-xs pr-2">
              <span>2</span>
              <span>1.5</span>
              <span>1</span>
              <span>0.5</span>
              <span>0</span>
            </div>
            
            {/* Bars */}
            <div className="flex-1 flex items-end justify-around pl-8">
              <div className="flex flex-col items-center">
                <div className="w-10 h-16 bg-accent rounded-t-md"></div>
                <span className="text-xs mt-2 text-gray-400">Mar 1-7</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-10 h-32 bg-accent rounded-t-md"></div>
                <span className="text-xs mt-2 text-gray-400">Mar 8-14</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-10 h-28 bg-accent rounded-t-md"></div>
                <span className="text-xs mt-2 text-gray-400">Mar 15-21</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-10 h-30 bg-accent rounded-t-md"></div>
                <span className="text-xs mt-2 text-gray-400">Mar 22-28</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-10 h-36 bg-accent rounded-t-md"></div>
                <span className="text-xs mt-2 text-gray-400">Final Week</span>
              </div>
            </div>
          </div>
          
          {/* Win Rate Section */}
          <div className="mt-8">
            <h3 className="text-lg font-medium mb-4">Win-Rate</h3>
            <div className="flex items-center justify-center">
              {/* Win-rate circle */}
              <div className="w-32 h-32 rounded-full bg-gray-700 flex items-center justify-center relative">
                {/* Progress circle */}
                <svg className="w-32 h-32 absolute">
                  <circle
                    className="text-accent"
                    strokeWidth="8"
                    strokeDasharray="289.02652413026095"
                    strokeDashoffset="109.83"
                    stroke="currentColor"
                    fill="transparent"
                    r="46"
                    cx="64"
                    cy="64"
                  />
                </svg>
                <span className="text-2xl font-bold">62%</span>
              </div>
            </div>
            
            {/* Win/Loss count */}
            <div className="flex justify-center mt-4">
              <div className="flex items-center mr-8">
                <div className="w-3 h-3 bg-gray-400 rounded-full mr-2"></div>
                <div>
                  <span className="block text-center text-xl font-bold">8</span>
                  <span className="text-xs text-gray-400">Unsuccessful</span>
                </div>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-accent rounded-full mr-2"></div>
                <div>
                  <span className="block text-center text-xl font-bold">13</span>
                  <span className="text-xs text-gray-400">Successful</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {activeTab === 'history' && (
        <div className="mt-6 space-y-4">
          {/* Player history items */}
          {['Brian_Le', 'emma_j', 'Harsh21', 'Bum_soo05'].map((player, index) => (
            <div key={player} className="flex items-center justify-between p-2 hover:bg-gray-700 rounded-lg">
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-sm ${
                  index === 0 ? 'bg-green-800' : index === 1 ? 'bg-purple-800' : index === 2 ? 'bg-yellow-800' : 'bg-orange-800'
                }`}>
                  {player.charAt(0).toUpperCase()}
                </div>
                <span className="ml-3">{player}</span>
              </div>
              <div className="flex items-center">
                <span className="text-xs text-gray-400 mr-2">Jan 7, 12:30pm</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GamePlay; 