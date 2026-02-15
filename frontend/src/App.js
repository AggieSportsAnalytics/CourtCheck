import React, { useState } from 'react';

// Component imports
import Sidebar from './components/Sidebar';
import Game01Stats from './components/Game01Stats';
import GameStatistics from './components/GameStatistics';
import HeatMaps from './components/HeatMaps';
import ShotPercentages from './components/ShotPercentages';
import GamePlay from './components/GamePlay';
import PlayerStats from './components/PlayerStats';
import VideoUpload from './components/VideoUpload';

const App = () => {
  const [currentGame, setCurrentGame] = useState('Game_01');
  const [currentNav, setCurrentNav] = useState('Match Stats');
  const [user, setUser] = useState({
    name: 'Coach Jackson',
    id: '64838263',
    image: 'https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/ucdavisaggies.com/images/2024/12/2/Coach_Sara_Jackson_Headshot_24.jpg?width=300'
  });

  const handleUploadComplete = () => {
    // Navigate to recordings or show success message
    setCurrentNav('Recordings');
  };

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
      case 'Upload Video':
        return (
          <div className="px-4 py-8">
            <div className="max-w-4xl mx-auto">
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-white mb-2">Upload Your Tennis Match</h3>
                <p className="text-gray-400">
                  Upload a video of your tennis match to analyze ball tracking, court detection, player movement, and more.
                </p>
              </div>
              <VideoUpload onUploadComplete={handleUploadComplete} />
              <div className="mt-8 p-6 bg-secondary rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-4">What We Analyze:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-start">
                    <span className="text-2xl mr-3">🎾</span>
                    <div>
                      <h5 className="font-medium text-white">Ball Tracking</h5>
                      <p className="text-sm text-gray-400">Track ball movement throughout the match</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-2xl mr-3">📐</span>
                    <div>
                      <h5 className="font-medium text-white">Court Detection</h5>
                      <p className="text-sm text-gray-400">Identify court boundaries and lines</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-2xl mr-3">🏃</span>
                    <div>
                      <h5 className="font-medium text-white">Player Movement</h5>
                      <p className="text-sm text-gray-400">Analyze player positioning and movement patterns</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <span className="text-2xl mr-3">📊</span>
                    <div>
                      <h5 className="font-medium text-white">Match Statistics</h5>
                      <p className="text-sm text-gray-400">Generate comprehensive match analytics</p>
                    </div>
                  </div>
                </div>
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
            <h1 className="ml-2 text-xl">Hey Coach Jackson!</h1>
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
              <img src={user.image} alt="User" className="h-10 w-10 rounded-full object-cover object-top" />
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