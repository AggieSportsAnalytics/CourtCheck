import React, { useState, Suspense } from 'react';
import { 
  HiVideoCamera, 
  HiUpload, 
  HiChartBar, 
  HiLocationMarker, 
  HiClock,
  HiLightningBolt,
  HiExclamation,
  HiUser
} from 'react-icons/hi';

// Import our enhanced components
import HeatMaps from './HeatMaps';
import PlayerStats from './PlayerStats';
import ShotPercentages from './ShotPercentages';
import PlayerSummary from './PlayerSummary';

// Fallback component
const LoadingFallback = () => (
  <div className="flex-1 h-full bg-gray-900 flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
      <p className="mt-4 text-white">Loading dashboard...</p>
    </div>
  </div>
);

const Dashboard = ({ matchData, currentMatch, setCurrentMatch }) => {
  const [activeSection, setActiveSection] = useState('recent'); // 'recent', 'insights', 'upload'
  
  // Select most recent match for the initial display if no match is selected
  const selectedMatch = currentMatch || (matchData?.length > 0 ? matchData[0] : null);

  const handleUploadClick = () => {
    // Navigate to the upload page programmatically
    window.location.href = '/upload';
  };

  // If no match data is provided, show a fallback component
  if (!matchData || matchData.length === 0) {
    return (
      <div className="flex-1 h-full bg-gray-900 flex items-center justify-center">
        <div className="text-center max-w-md p-6 bg-gray-800 rounded-xl shadow-lg">
          <HiExclamation className="h-12 w-12 text-amber-500 mx-auto" />
          <h2 className="text-xl font-bold text-white mt-4">No Match Data</h2>
          <p className="text-gray-400 mt-2">
            There appears to be no match data available. Please make sure the data is properly loaded.
          </p>
        </div>
      </div>
    );
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <div className="flex-1 h-full bg-gray-900 overflow-y-auto">
        {/* Header with profile */}
        <header className="sticky top-0 z-30 bg-gray-900 border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-white">CourtCheck Dashboard</h1>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="h-3 w-3 absolute top-0 right-0 bg-green-500 rounded-full border-2 border-gray-900"></div>
                <span className="bg-green-500/10 text-green-400 px-2 py-1 rounded-full text-xs font-medium flex items-center">
                  <span className="mr-1">●</span> LIVE
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Dashboard Navigation Tabs */}
        <div className="border-b border-gray-800">
          <nav className="flex px-6" aria-label="Tabs">
            <button
              onClick={() => setActiveSection('recent')}
              className={`py-3 px-4 text-sm font-medium border-b-2 ${
                activeSection === 'recent'
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              Recent Matches
            </button>
            <button
              onClick={() => setActiveSection('insights')}
              className={`py-3 px-4 text-sm font-medium border-b-2 ${
                activeSection === 'insights'
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              Insights
            </button>
            <button
              onClick={() => setActiveSection('upload')}
              className={`py-3 px-4 text-sm font-medium border-b-2 ${
                activeSection === 'upload'
                  ? 'border-blue-500 text-blue-500'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              Upload New
            </button>
          </nav>
        </div>

        {/* Main Content */}
        <main className="px-6 py-4">
          {activeSection === 'recent' && (
            <>
              {/* Recent Matches Section */}
              <div className="mb-6">
                <h2 className="text-xl font-bold text-white mb-4">Recent Matches</h2>

                {/* Match Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {matchData.map((match) => (
                    <div 
                      key={match.id}
                      className={`bg-gray-800 rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-shadow cursor-pointer ${
                        selectedMatch?.id === match.id ? 'ring-2 ring-blue-500' : ''
                      }`}
                      onClick={() => setCurrentMatch(match)}
                    >
                      <div className="p-5">
                        <div className="flex justify-between items-start">
                          <h3 className="text-white font-bold text-lg">{match.title}</h3>
                          <span className="text-xs text-gray-400">{match.date}</span>
                        </div>
                        <div className="text-sm text-gray-400 mt-1">{match.location}</div>
                        
                        <div className="mt-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Ball Tracking</span>
                            <span className="text-blue-400 font-medium">{match.stats.ballTrackingFrames} frames</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Bounce Detection</span>
                            <span className="text-blue-400 font-medium">{match.stats.bounceDetections} bounces</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-400">Rallies</span>
                            <span className="text-blue-400 font-medium">{match.stats.rallies} total</span>
                          </div>
                        </div>
                        
                        <div className="mt-4 pt-4 border-t border-gray-700">
                          <div className="flex items-center justify-between text-sm">
                            <div className="flex items-center text-gray-400">
                              <HiChartBar className="w-4 h-4 mr-1" />
                              <span>View Analysis</span>
                            </div>
                            <span className="text-blue-500 font-medium">→</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Selected Match Analysis */}
              {selectedMatch && (
                <div className="mt-8">
                  <h2 className="text-xl font-bold text-white mb-4">Match Analysis: {selectedMatch.title}</h2>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Left Column: Match Info, Key Metrics, and Player Summary */}
                    <div className="space-y-6">
                      {/* Match Info */}
                      <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                        <h3 className="text-white font-bold mb-4">Match Details</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Date</span>
                            <span className="text-white">{selectedMatch.date}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Location</span>
                            <span className="text-white">{selectedMatch.location}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Players</span>
                            <span className="text-white">{`${selectedMatch.players.team1.name} vs ${selectedMatch.players.team2.name}`}</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Key Metrics */}
                      <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                        <h3 className="text-white font-bold mb-4">Key Metrics</h3>
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <div className="flex items-center">
                                <HiUser className="w-4 h-4 mr-2 text-blue-400" />
                                <span className="text-gray-400">{selectedMatch.players.team1.name} Score</span>
                              </div>
                              <span className="text-blue-400 font-medium">{selectedMatch.stats.player1Score}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full" 
                                style={{ width: `${selectedMatch.stats.player1Score}%` }}
                              ></div>
                            </div>
                          </div>
                          
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <div className="flex items-center">
                                <HiUser className="w-4 h-4 mr-2 text-red-400" />
                                <span className="text-gray-400">{selectedMatch.players.team2.name} Score</span>
                              </div>
                              <span className="text-red-400 font-medium">{selectedMatch.stats.player2Score}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-red-500 h-2 rounded-full" 
                                style={{ width: `${selectedMatch.stats.player2Score}%` }}
                              ></div>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4 mt-4">
                            <div className="bg-gray-700 rounded-lg p-3">
                              <div className="text-gray-400 text-sm">Rallies</div>
                              <div className="text-xl font-bold text-white">{selectedMatch.stats.rallies}</div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-3">
                              <div className="text-gray-400 text-sm">Longest Rally</div>
                              <div className="text-xl font-bold text-white">{selectedMatch.stats.longestRally} hits</div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-3">
                              <div className="text-gray-400 text-sm">Bounces</div>
                              <div className="text-xl font-bold text-white">{selectedMatch.stats.bounceDetections}</div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-3">
                              <div className="text-gray-400 text-sm">Serves</div>
                              <div className="text-xl font-bold text-white">{selectedMatch.stats.servesDetected}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Player Summary */}
                      <PlayerSummary match={selectedMatch} />
                    </div>
                    
                    {/* Right Column: Heat Maps, Shot Percentages, Player Stats */}
                    <div className="space-y-6">
                      {/* Heat Maps */}
                      <HeatMaps match={selectedMatch} />
                      
                      {/* Player Statistics Section */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <ShotPercentages match={selectedMatch} />
                        <PlayerStats match={selectedMatch} />
                      </div>
                      
                      {/* Match Insights */}
                      <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                        <h3 className="text-white font-bold mb-4">AI Match Insights</h3>
                        <p className="text-gray-300">{selectedMatch.analysis.matchSummary}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {activeSection === 'insights' && (
            <div className="text-center py-12">
              <h2 className="text-xl font-bold text-white mb-4">Performance Insights</h2>
              <p className="text-gray-400 mb-8">Coming soon: Advanced analytics and trend analysis across multiple matches</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto text-left">
                <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                  <div className="flex items-start">
                    <div className="rounded-lg bg-blue-500/20 p-3">
                      <HiChartBar className="h-6 w-6 text-blue-500" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-white font-bold">Pattern Recognition</h3>
                      <p className="text-gray-400 text-sm mt-1">AI-powered analysis of play patterns across matches</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                  <div className="flex items-start">
                    <div className="rounded-lg bg-green-500/20 p-3">
                      <HiLightningBolt className="h-6 w-6 text-green-500" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-white font-bold">Performance Tracking</h3>
                      <p className="text-gray-400 text-sm mt-1">Track progress and improvements over time</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-800 rounded-xl p-5 shadow-md">
                  <div className="flex items-start">
                    <div className="rounded-lg bg-red-500/20 p-3">
                      <HiExclamation className="h-6 w-6 text-red-500" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-white font-bold">Weakness Identification</h3>
                      <p className="text-gray-400 text-sm mt-1">Pinpoint areas for improvement</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'upload' && (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="max-w-md w-full bg-gray-800 rounded-xl p-6 shadow-md">
                <div className="text-center">
                  <HiUpload className="mx-auto h-12 w-12 text-blue-500" />
                  <h2 className="mt-3 text-xl font-bold text-white">Upload Tennis Match</h2>
                  <p className="mt-2 text-gray-400">Upload your tennis match video for automated analysis</p>
                </div>
                
                <div className="mt-6">
                  <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
                    <p className="text-gray-400 text-sm mb-4">Supported formats: MP4, MOV (720p or higher recommended)</p>
                    <button
                      onClick={handleUploadClick}
                      className="mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                    >
                      Select Video File
                    </button>
                  </div>
                  
                  <div className="mt-6 bg-gray-700 rounded-lg p-4">
                    <h3 className="text-white font-medium mb-2">Automatic Processing</h3>
                    <ul className="text-sm text-gray-400 space-y-2">
                      <li className="flex items-start">
                        <div className="text-green-500 mr-2">✓</div>
                        <div>Court detection and tracking</div>
                      </li>
                      <li className="flex items-start">
                        <div className="text-green-500 mr-2">✓</div>
                        <div>Ball tracking and bounce detection</div>
                      </li>
                      <li className="flex items-start">
                        <div className="text-green-500 mr-2">✓</div>
                        <div>Player tracking with unique IDs</div>
                      </li>
                      <li className="flex items-start">
                        <div className="text-green-500 mr-2">✓</div>
                        <div>Automated heatmap generation</div>
                      </li>
                      <li className="flex items-start">
                        <div className="text-green-500 mr-2">✓</div>
                        <div>Video exports with analysis overlay</div>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </Suspense>
  );
};

export default Dashboard; 