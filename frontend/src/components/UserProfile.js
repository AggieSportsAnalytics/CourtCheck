import React, { useState } from 'react';
import { 
  HiUser, 
  HiMail, 
  HiUserGroup, 
  HiOfficeBuilding, 
  HiCalendar,
  HiPhotograph,
  HiChartBar,
  HiClock
} from 'react-icons/hi';

const UserProfile = ({ user }) => {
  const [activeTab, setActiveTab] = useState('profile'); // 'profile', 'activity', 'settings'
  
  const userStats = {
    matchesAnalyzed: 24,
    totalPlayTime: '32h 15m',
    averageMatchLength: '1h 21m',
    lastActive: 'Today, 2:45 PM',
    teamMembers: 8,
    joinDate: 'March 15, 2023'
  };
  
  const recentActivity = [
    { 
      type: 'analysis_complete', 
      match: 'UC Davis vs. Stanford',
      date: '2 hours ago',
      icon: <HiChartBar className="w-5 h-5 text-green-500" />,
      description: 'Match analysis completed with 98.2% accuracy'
    },
    { 
      type: 'video_upload', 
      match: 'UC Davis vs. Hawaii',
      date: 'Yesterday',
      icon: <HiPhotograph className="w-5 h-5 text-blue-500" />,
      description: 'Uploaded a new match video (24:15)'
    },
    { 
      type: 'player_identified', 
      match: 'UC Davis vs. Hawaii',
      date: 'Yesterday',
      icon: <HiUser className="w-5 h-5 text-purple-500" />,
      description: 'Identified players in 10 frames'
    },
    { 
      type: 'session_ended', 
      match: 'UC Davis vs. Berkeley',
      date: '3 days ago',
      icon: <HiClock className="w-5 h-5 text-red-500" />,
      description: 'Viewed match analysis for 45 minutes'
    }
  ];

  return (
    <div className="flex-1 h-full bg-gray-900 overflow-y-auto">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">Profile</h1>
        </div>
      </header>
      
      {/* Profile Content */}
      <div className="max-w-6xl mx-auto p-6">
        {/* Profile Header */}
        <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
          <div className="flex flex-col md:flex-row items-center">
            <div className="flex-shrink-0 mb-4 md:mb-0">
              <img 
                src={user.image}
                alt={user.name}
                className="h-28 w-28 rounded-full object-cover border-4 border-blue-600"
              />
            </div>
            <div className="md:ml-8 flex-1 text-center md:text-left">
              <h2 className="text-2xl font-bold text-white">{user.name}</h2>
              <p className="text-blue-400 text-lg">{user.role} • {user.team}</p>
              <p className="text-gray-400 mt-1">ID: {user.id}</p>
              
              <div className="flex flex-wrap gap-4 mt-4 justify-center md:justify-start">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                  Edit Profile
                </button>
                <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors">
                  View Team
                </button>
              </div>
            </div>
            
            <div className="hidden md:grid grid-cols-2 gap-4 md:ml-auto mt-6 md:mt-0">
              <div className="bg-gray-700 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-white">{userStats.matchesAnalyzed}</div>
                <div className="text-sm text-gray-400">Matches Analyzed</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-white">{userStats.totalPlayTime}</div>
                <div className="text-sm text-gray-400">Total Play Time</div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Tab Navigation */}
        <div className="mt-8">
          <div className="border-b border-gray-800">
            <nav className="-mb-px flex space-x-6" aria-label="Tabs">
              <button
                onClick={() => setActiveTab('profile')}
                className={`py-3 border-b-2 font-medium text-sm ${
                  activeTab === 'profile'
                    ? 'border-blue-500 text-blue-500'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-700'
                }`}
              >
                Profile Details
              </button>
              <button
                onClick={() => setActiveTab('activity')}
                className={`py-3 border-b-2 font-medium text-sm ${
                  activeTab === 'activity'
                    ? 'border-blue-500 text-blue-500'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-700'
                }`}
              >
                Recent Activity
              </button>
              <button
                onClick={() => setActiveTab('settings')}
                className={`py-3 border-b-2 font-medium text-sm ${
                  activeTab === 'settings'
                    ? 'border-blue-500 text-blue-500'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-700'
                }`}
              >
                Settings
              </button>
            </nav>
          </div>
        </div>
        
        {/* Tab Content */}
        <div className="mt-6">
          {activeTab === 'profile' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* User Information */}
              <div className="lg:col-span-2">
                <div className="bg-gray-800 rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-white mb-4">User Information</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Full Name</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiUser className="text-gray-400 mr-3" />
                          <span className="text-white">{user.name}</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">User ID</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiUser className="text-gray-400 mr-3" />
                          <span className="text-white">{user.id}</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Email</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiMail className="text-gray-400 mr-3" />
                          <span className="text-white">{user.name.toLowerCase().replace('_', '.')}@ucdavis.edu</span>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Team</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiUserGroup className="text-gray-400 mr-3" />
                          <span className="text-white">{user.team}</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Role</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiOfficeBuilding className="text-gray-400 mr-3" />
                          <span className="text-white">{user.role}</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-400 mb-1">Joined</label>
                        <div className="flex items-center bg-gray-700 rounded-lg p-3">
                          <HiCalendar className="text-gray-400 mr-3" />
                          <span className="text-white">{userStats.joinDate}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Stats Card */}
              <div>
                <div className="bg-gray-800 rounded-xl shadow-lg p-6">
                  <h3 className="text-xl font-bold text-white mb-4">Usage Statistics</h3>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-700 rounded-lg p-3 text-center">
                        <div className="text-xl font-bold text-white">{userStats.matchesAnalyzed}</div>
                        <div className="text-xs text-gray-400">Matches</div>
                      </div>
                      <div className="bg-gray-700 rounded-lg p-3 text-center">
                        <div className="text-xl font-bold text-white">{userStats.teamMembers}</div>
                        <div className="text-xs text-gray-400">Team Members</div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700 rounded-lg p-4">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-400">Last Active</span>
                        <span className="text-sm text-blue-400">{userStats.lastActive}</span>
                      </div>
                      <div className="h-1 w-full bg-gray-600 rounded-full">
                        <div className="h-1 bg-blue-500 rounded-full" style={{ width: '75%' }}></div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700 rounded-lg p-4">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-400">Total Play Time</span>
                        <span className="text-sm text-green-400">{userStats.totalPlayTime}</span>
                      </div>
                      <div className="h-1 w-full bg-gray-600 rounded-full">
                        <div className="h-1 bg-green-500 rounded-full" style={{ width: '60%' }}></div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700 rounded-lg p-4">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-400">Avg. Match Length</span>
                        <span className="text-sm text-purple-400">{userStats.averageMatchLength}</span>
                      </div>
                      <div className="h-1 w-full bg-gray-600 rounded-full">
                        <div className="h-1 bg-purple-500 rounded-full" style={{ width: '45%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'activity' && (
            <div className="bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-white mb-6">Recent Activity</h3>
              <div className="space-y-6">
                {recentActivity.map((activity, index) => (
                  <div key={index} className="flex">
                    <div className="flex-shrink-0">
                      <div className="bg-gray-700 rounded-full p-3 mr-4">
                        {activity.icon}
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-white">{activity.match}</p>
                        <p className="text-xs text-gray-400">{activity.date}</p>
                      </div>
                      <p className="text-sm text-gray-400 mt-1">{activity.description}</p>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-8 text-center">
                <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors">
                  View All Activity
                </button>
              </div>
            </div>
          )}
          
          {activeTab === 'settings' && (
            <div className="bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-white mb-6">Account Settings</h3>
              <div className="space-y-6">
                <div>
                  <h4 className="text-lg font-medium text-white mb-3">Preferences</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                      <span className="text-gray-300">Email Notifications</span>
                      <button className="relative inline-flex items-center h-6 rounded-full w-12 bg-blue-600">
                        <span className="inline-block h-4 w-4 transform translate-x-7 rounded-full bg-white"></span>
                      </button>
                    </div>
                    <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                      <span className="text-gray-300">Auto-process Uploads</span>
                      <button className="relative inline-flex items-center h-6 rounded-full w-12 bg-blue-600">
                        <span className="inline-block h-4 w-4 transform translate-x-7 rounded-full bg-white"></span>
                      </button>
                    </div>
                    <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                      <span className="text-gray-300">Dark Mode</span>
                      <button className="relative inline-flex items-center h-6 rounded-full w-12 bg-blue-600">
                        <span className="inline-block h-4 w-4 transform translate-x-7 rounded-full bg-white"></span>
                      </button>
                    </div>
                    <div className="flex items-center justify-between bg-gray-700 rounded-lg p-4">
                      <span className="text-gray-300">Share Analytics</span>
                      <button className="relative inline-flex items-center h-6 rounded-full w-12 bg-gray-600">
                        <span className="inline-block h-4 w-4 transform translate-x-1 rounded-full bg-white"></span>
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-700">
                  <h4 className="text-lg font-medium text-white mb-3">Account Management</h4>
                  <div className="space-y-4">
                    <button className="w-full text-left px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center">
                      <HiMail className="mr-3" />
                      Change Email Address
                    </button>
                    <button className="w-full text-left px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center">
                      <HiUser className="mr-3" />
                      Update Profile Information
                    </button>
                    <button className="w-full text-left px-4 py-3 bg-red-900/30 hover:bg-red-900/40 text-red-400 rounded-lg transition-colors flex items-center">
                      <HiX className="mr-3" />
                      Delete Account
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile; 