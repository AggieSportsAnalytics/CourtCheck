import React, { useState, useEffect, useRef } from 'react';

const ShotPercentages = () => {
  const [activeSet, setActiveSet] = useState('overall');
  const chartRef = useRef(null);

  // Mock data for shot percentages
  const shotData = {
    forehand: 22,
    backhand: 28,
    slices: 9,
    lob: 16,
    drop: 25
  };

  // Colors for the shot types
  const shotColors = {
    forehand: '#a4de02',
    backhand: '#436b22',
    slices: '#acc260',
    lob: '#225a28',
    drop: '#d4eb8f'
  };

  // This would normally use Chart.js or similar library
  // For this demo, we'll create a visual representation with divs

  return (
    <div className="bg-white rounded-xl p-4">
      <h3 className="text-xl font-bold mb-4 text-gray-800">Shot Percentages</h3>
      
      {/* Set Selection */}
      <div className="grid grid-cols-4 gap-1 mb-4">
        <button
          className={`py-2 rounded-md ${activeSet === 'set1' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActiveSet('set1')}
        >
          Set 1
        </button>
        <button
          className={`py-2 rounded-md ${activeSet === 'set2' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActiveSet('set2')}
        >
          Set 2
        </button>
        <button
          className={`py-2 rounded-md ${activeSet === 'set3' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActiveSet('set3')}
        >
          Set 3
        </button>
        <button
          className={`py-2 rounded-md ${activeSet === 'overall' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-600'}`}
          onClick={() => setActiveSet('overall')}
        >
          Overall
        </button>
      </div>
      
      {/* Pie Chart */}
      <div className="flex items-center justify-center mt-4">
        <div className="w-36 h-36 rounded-full border-4 border-gray-700 relative" ref={chartRef}>
          {/* This is a simple CSS pie chart - in a real app you'd use a chart library */}
          <div className="absolute inset-0 bg-[#a4de02] rounded-full" style={{ clipPath: 'polygon(50% 50%, 50% 0, 100% 0, 100% 22%, 50% 50%)' }}></div>
          <div className="absolute inset-0 bg-[#436b22] rounded-full" style={{ clipPath: 'polygon(50% 50%, 100% 22%, 100% 50%, 78% 100%, 50% 50%)' }}></div>
          <div className="absolute inset-0 bg-[#acc260] rounded-full" style={{ clipPath: 'polygon(50% 50%, 78% 100%, 50% 100%, 22% 100%, 50% 50%)' }}></div>
          <div className="absolute inset-0 bg-[#225a28] rounded-full" style={{ clipPath: 'polygon(50% 50%, 22% 100%, 0 78%, 0 50%, 50% 50%)' }}></div>
          <div className="absolute inset-0 bg-[#d4eb8f] rounded-full" style={{ clipPath: 'polygon(50% 50%, 0 50%, 0 22%, 0 0, 50% 0, 50% 50%)' }}></div>
          
          {/* Center circle */}
          <div className="absolute w-16 h-16 bg-white rounded-full top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
            <span className="text-gray-800 font-bold">100%</span>
          </div>
        </div>
      </div>
      
      {/* Legend */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-6">
        {Object.entries(shotData).map(([type, percentage]) => (
          <div key={type} className="flex items-center">
            <div 
              className="w-3 h-3 rounded-full mr-2" 
              style={{ backgroundColor: shotColors[type] }}
            ></div>
            <span className="text-sm capitalize text-gray-800">{type}</span>
            <span className="text-sm ml-auto text-gray-600">{percentage}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ShotPercentages; 