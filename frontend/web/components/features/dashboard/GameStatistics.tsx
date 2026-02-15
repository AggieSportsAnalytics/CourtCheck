'use client';

interface ShotType {
  type: string;
  color: string;
  value: number;
}

const GameStatistics = () => {
  // Mock data for shot types
  const shotTypes: ShotType[] = [
    { type: 'Serves', color: 'bg-green-800', value: 62 },
    { type: 'Forehand', color: 'bg-gray-600', value: 85 },
    { type: 'Backhand', color: 'bg-green-300', value: 78 },
    { type: 'Slice', color: 'bg-green-800', value: 72 }
  ];

  return (
    <div className="bg-white rounded-xl p-4">
      <h3 className="text-xl font-bold mb-4 text-gray-800">Points-by-Shots</h3>

      <div className="flex flex-col">
        {/* Error count */}
        <div className="mb-6">
          <span className="text-sm text-gray-600">Total number of errors:</span>
          <span className="ml-2 text-xl font-bold text-gray-800">306</span>
        </div>

        {/* Chart and Legend */}
        <div className="flex flex-wrap">
          {/* Chart Bars */}
          <div className="w-full lg:w-3/4 h-44 flex items-end justify-around space-x-6 mb-6" aria-label="Points by shot type bar chart">
            {shotTypes.map((shot, index) => (
              <div key={index} className="flex flex-col items-center">
                <div className={`w-14 ${shot.color} rounded-t-md`} style={{ height: `${shot.value * 0.4}px` }} aria-hidden="true"></div>
                <div className={`w-6 h-6 rounded-full ${shot.color} -mt-3 flex items-center justify-center text-xs font-bold text-white`} aria-label={shot.type}>
                  {shot.type.charAt(0).toLowerCase()}
                </div>
              </div>
            ))}
          </div>

          {/* Legend */}
          <div className="w-full lg:w-1/4">
            <ul className="space-y-3">
              {shotTypes.map((shot, index) => (
                <li key={index} className="flex items-center">
                  <div className={`w-4 h-4 rounded-full ${shot.color} mr-2`} aria-hidden="true"></div>
                  <span className="text-sm text-gray-800">{shot.type}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameStatistics;
