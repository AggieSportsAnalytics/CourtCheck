import React, { useState } from 'react';

// Component imports
import Sidebar from './components/Sidebar';
import Game01Stats from './components/Game01Stats';
import GameStatistics from './components/GameStatistics';
import HeatMaps from './components/HeatMaps';
import ShotPercentages from './components/ShotPercentages';
import GamePlay from './components/GamePlay';
import PlayerStats from './components/PlayerStats';

const App = () => {
  const [currentGame, setCurrentGame] = useState('Game_01');
  const [currentNav, setCurrentNav] = useState('Match Stats');
  const [user, setUser] = useState({
    name: 'Cory_Pham_22',
    id: '64838263',
    image: 'https://i.pravatar.cc/150?img=3'
  });

  const renderContent = () => {
    switch (currentNav) {
      case 'Dashboard':
        return (
          <div className="px-4 grid gap-4 pb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <HeatMaps />
              <div className="grid gap-4">
                <PlayerStats />
                <ShotPercentages />
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <GamePlay />
              <div className="col-span-2">
                <GameStatistics />
              </div>
            </div>
          </div>
        );
      case 'Match Stats':
        if (currentGame === 'Game_01') {
          return <Game01Stats />;
        }
        return null;
      default:
        return (
          <div className="px-4 py-8">
            <h2 className="text-2xl font-bold text-white">Coming Soon</h2>
            <p className="text-gray-400 mt-2">This section is under development.</p>
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-primary text-white overflow-hidden">
      {/* Sidebar */}
      <Sidebar 
        username={user.name} 
        onGameSelect={setCurrentGame} 
        onNavSelect={setCurrentNav}
      />
      
      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        {/* Header */}
        <header className="p-4 flex justify-between items-center">
          <div className="flex items-center">
            <span className="text-yellow-400">👋</span>
            <h1 className="ml-2 text-xl">Hey Cory!</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center">
              <span className="bg-green-500 rounded-full h-2.5 w-2.5 inline-block mr-2"></span>
              <span>Live</span>
            </div>
            
            <div className="relative">
              <button className="flex items-center bg-secondary rounded-full px-4 py-2 gap-2">
                <span>English</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </div>
            
            <div className="relative">
              <button className="flex items-center">
                <div className="relative">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                  <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-red-500"></span>
                </div>
              </button>
            </div>
            
            <div className="flex items-center gap-2">
              <img src={user.image} alt="User" className="h-10 w-10 rounded-full object-cover" />
              <div>
                <p className="text-sm font-medium">{user.name}</p>
                <p className="text-xs text-gray-400">ID: {user.id}</p>
              </div>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </header>
        
        {/* Page title */}
        <div className="px-4 mb-6 flex justify-between items-center">
          <h2 className="text-2xl font-bold">{currentNav === 'Match Stats' ? `${currentGame} Statistics` : currentNav}</h2>
          <button className="bg-secondary p-2 rounded">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
        
        {/* Dynamic content area */}
        {renderContent()}
      </div>
    </div>
  );
};

export default App; 