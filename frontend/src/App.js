import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Components
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import VideoUpload from './components/VideoUpload';
import UserProfile from './components/UserProfile';

// Demo Data
import { matchData } from './data/demoData';

const App = () => {
  // Always authenticated for demo purposes
  const [currentMatch, setCurrentMatch] = useState(null);
  const [user, setUser] = useState({
    name: 'Cory_Pham_22',
    id: '64838263',
    role: 'Coach',
    team: 'UC Davis',
    image: 'https://i.pravatar.cc/150?img=3'
  });
  
  return (
    <Router>
      <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
        <Sidebar 
          user={user} 
          matchData={matchData}
          currentMatch={currentMatch}
          setCurrentMatch={setCurrentMatch}
        />
        
        <div className="flex-1 flex flex-col overflow-hidden">
          <Routes>
            <Route 
              path="/"
              element={
                <Dashboard 
                  matchData={matchData} 
                  currentMatch={currentMatch}
                  setCurrentMatch={setCurrentMatch}
                />
              }
            />
            <Route 
              path="/upload"
              element={<VideoUpload />}
            />
            <Route 
              path="/profile"
              element={<UserProfile user={user} />}
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App; 