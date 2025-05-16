import React, { useState } from 'react';

const PlayerSummary = ({ match }) => {
  const [activePlayer, setActivePlayer] = useState('team1');
  
  // Hardcoded summaries for each player
  const playerSummaries = {
    team1: {
      tactical: "Shows strong baseline play with effective cross-court shots. Serves primarily to the opponent's backhand with 68% accuracy. Forehand is dominant with 73% of winners coming from that side.",
      technical: "Excellent footwork around the court with strong recovery after wide shots. Backhand technique could be improved, particularly on high balls. First serve percentage at 62% with room for improvement.",
      physical: "Covers the court well with 1872m total distance. Shows good sprint ability with 34 recorded sprints. Court coverage at 68.4% indicates good spatial awareness and positioning.",
      areas: ["Improve backhand technique", "Work on first serve percentage", "Develop net approach strategy"]
    },
    team2: {
      tactical: "Strong defensive player with consistent returns. Prefers to play from the baseline and rarely approaches the net. Targets opponent's backhand effectively.",
      technical: "Excellent two-handed backhand with good directional control. Serve has lower velocity but high placement accuracy. Forehand spin could be improved for greater depth.",
      physical: "High endurance player covering 1924m during the match. Sprint count of 38 shows good recovery ability. Court coverage at 73.1% demonstrates excellent court awareness.",
      areas: ["Develop net play", "Increase serve velocity", "Work on inside-out forehand"]
    }
  };

  const summary = playerSummaries[activePlayer];
  const player = match?.players[activePlayer];

  return (
    <div className="bg-gray-800 rounded-xl p-5 shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-bold">Player Summary</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setActivePlayer('team1')}
            className={`px-3 py-1 text-sm rounded-md ${
              activePlayer === 'team1'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {match?.players.team1.name || 'Player 1'}
          </button>
          <button
            onClick={() => setActivePlayer('team2')}
            className={`px-3 py-1 text-sm rounded-md ${
              activePlayer === 'team2'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {match?.players.team2.name || 'Player 2'}
          </button>
        </div>
      </div>
      
      {summary && (
        <div className="space-y-4">
          <div>
            <h4 className="text-blue-400 font-medium mb-1">Tactical Analysis</h4>
            <p className="text-gray-300 text-sm">{summary.tactical}</p>
          </div>
          
          <div>
            <h4 className="text-blue-400 font-medium mb-1">Technical Analysis</h4>
            <p className="text-gray-300 text-sm">{summary.technical}</p>
          </div>
          
          <div>
            <h4 className="text-blue-400 font-medium mb-1">Physical Performance</h4>
            <p className="text-gray-300 text-sm">{summary.physical}</p>
          </div>
          
          <div>
            <h4 className="text-blue-400 font-medium mb-1">Areas for Improvement</h4>
            <ul className="text-gray-300 text-sm list-disc pl-5">
              {summary.areas.map((area, index) => (
                <li key={index}>{area}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlayerSummary; 