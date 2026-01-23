import React from 'react';
// Safely import icons with a fallback
let HiChartBar, HiCamera, HiBadgeCheck, HiLightningBolt, HiVideoCamera;
try {
  const icons = require('react-icons/hi');
  HiChartBar = icons.HiChartBar;
  HiCamera = icons.HiCamera;
  HiBadgeCheck = icons.HiBadgeCheck;
  HiLightningBolt = icons.HiLightningBolt;
  HiVideoCamera = icons.HiVideoCamera;
} catch (e) {
  // Provide fallback components if icons fail to load
  const FallbackIcon = () => <span className="w-6 h-6 bg-blue-500 rounded-full inline-block"></span>;
  HiChartBar = HiCamera = HiBadgeCheck = HiLightningBolt = HiVideoCamera = FallbackIcon;
}

const Dashboard = () => {
  // Mock data for demonstration
  const matchStats = {
    totalBounces: 342,
    totalShots: 187,
    accurateShots: 152,
    accuracy: 81.3,
    rally: 8.4
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Match Analysis Dashboard</h1>
        <p className="text-gray-400">
          Automated insights for UC Davis Tennis vs. Cal Poly | March 12, 2023
        </p>
      </div>

      {/* Main content area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Court visualization */}
        <div className="bg-gray-800 p-5 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-white mb-4">Court Visualization</h2>
          <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Tennis court visualization appears here</p>
          </div>
        </div>

        {/* Stats panel */}
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-gray-800 p-5 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-white mb-4">Match Statistics</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Total Shots</span>
                <span className="text-white font-semibold">187</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Shot Accuracy</span>
                <span className="text-white font-semibold">81.3%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Avg. Rally Length</span>
                <span className="text-white font-semibold">8.4 shots</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-5 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-white mb-4">Ball Tracking</h2>
            <div className="bg-gray-900 h-32 rounded-lg flex items-center justify-center">
              <p className="text-gray-400">Ball tracking visualization appears here</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 