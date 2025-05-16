import React, { useState, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Components
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import VideoUpload from './components/VideoUpload';
import UserProfile from './components/UserProfile';

// Demo Data
import { matchData } from './data/demoData';

// Fallback component for suspense
const LoadingFallback = () => (
  <div className="flex-1 h-full bg-gray-900 flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
      <p className="mt-4 text-white">Loading CourtCheck...</p>
    </div>
  </div>
);

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
  
  try {
    return (
      <Suspense fallback={<LoadingFallback />}>
        <Router>
          <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
            <Sidebar 
              key="sidebar"
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
                      key="dashboard"
                      matchData={matchData} 
                      currentMatch={currentMatch}
                      setCurrentMatch={setCurrentMatch}
                    />
                  }
                />
                <Route 
                  path="/upload"
                  element={<VideoUpload key="upload" />}
                />
                <Route 
                  path="/profile"
                  element={<UserProfile key="profile" user={user} />}
                />
              </Routes>
            </div>
          </div>
        </Router>
      </Suspense>
    );
  } catch (err) {
    console.error("Critical rendering error:", err);
    return (
      <div className="flex h-screen bg-gray-900 text-white items-center justify-center p-6">
        <div className="bg-gray-800 p-8 rounded-xl max-w-xl">
          <h1 className="text-2xl font-bold mb-4">CourtCheck</h1>
          <p className="mb-4">We're experiencing technical difficulties rendering the application.</p>
          <div className="bg-red-900/30 border border-red-700 p-4 rounded-lg text-sm">
            <p>Error: {err?.message || 'Unknown error occurred'}</p>
          </div>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-6 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg"
          >
            Try Reloading
          </button>
        </div>
      </div>
    );
  }
};

export default App; 