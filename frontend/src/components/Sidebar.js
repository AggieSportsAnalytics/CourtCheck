import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { 
  HiHome, 
  HiVideoCamera, 
  HiChartBar, 
  HiUser, 
  HiMenuAlt2,
  HiX,
  HiChevronDown,
  HiChevronUp
} from 'react-icons/hi';
import Logo from './Logo';

const Sidebar = ({ user, matchData, currentMatch, setCurrentMatch }) => {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMatchesOpen, setIsMatchesOpen] = useState(false);

  const navItems = [
    { name: 'Dashboard', path: '/', icon: <HiHome className="w-6 h-6" /> },
    { name: 'Upload Video', path: '/upload', icon: <HiVideoCamera className="w-6 h-6" /> },
    { name: 'Profile', path: '/profile', icon: <HiUser className="w-6 h-6" /> },
  ];

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <>
      {/* Mobile Menu Button */}
      <button 
        className="md:hidden fixed top-4 left-4 z-50 bg-gray-800 p-2 rounded-lg text-white"
        onClick={toggleSidebar}
      >
        {isCollapsed ? <HiMenuAlt2 className="w-6 h-6" /> : <HiX className="w-6 h-6" />}
      </button>

      {/* Sidebar */}
      <div 
        className={`fixed inset-y-0 left-0 z-40 flex flex-col bg-gray-900 text-white border-r border-gray-700 transition-all duration-300 ease-in-out md:relative ${
          isCollapsed ? '-translate-x-full md:translate-x-0 md:w-20' : 'w-64'
        }`}
      >
        {/* Logo */}
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center">
            <Logo size={isCollapsed ? "md" : "md"} />
            {!isCollapsed && <span className="ml-2 text-xl font-bold">CourtCheck</span>}
          </div>
          <button 
            className="hidden md:block text-gray-400 hover:text-white"
            onClick={toggleSidebar}
          >
            {isCollapsed ? <HiMenuAlt2 className="w-6 h-6" /> : <HiX className="w-6 h-6" />}
          </button>
        </div>

        {/* Navigation Links */}
        <div className="flex-1 overflow-y-auto py-4">
          <nav className="px-2 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) => 
                  `flex items-center px-2 py-3 rounded-lg transition-colors ${
                    isActive 
                      ? 'bg-blue-700 text-white' 
                      : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                  } ${isCollapsed ? 'justify-center' : 'justify-start'}`
                }
              >
                <span className="text-xl">{item.icon}</span>
                {!isCollapsed && <span className="ml-3">{item.name}</span>}
              </NavLink>
            ))}

            {/* Match Analysis Dropdown */}
            <div>
              <button
                onClick={() => setIsMatchesOpen(!isMatchesOpen)}
                className={`w-full flex items-center px-2 py-3 rounded-lg transition-colors text-gray-300 hover:bg-gray-800 hover:text-white ${
                  location.pathname.includes('/match') && 'bg-gray-800 text-white'
                } ${isCollapsed ? 'justify-center' : 'justify-between'}`}
              >
                <div className="flex items-center">
                  <span className="text-xl"><HiChartBar className="w-6 h-6" /></span>
                  {!isCollapsed && <span className="ml-3">Match Analysis</span>}
                </div>
                {!isCollapsed && (
                  isMatchesOpen ? <HiChevronUp className="w-5 h-5" /> : <HiChevronDown className="w-5 h-5" />
                )}
              </button>

              {isMatchesOpen && !isCollapsed && (
                <div className="pl-10 mt-1 space-y-1">
                  {matchData.map((match) => (
                    <button
                      key={match.id}
                      onClick={() => setCurrentMatch(match)}
                      className={`w-full text-left py-2 px-3 rounded-md text-sm ${
                        currentMatch?.id === match.id
                          ? 'bg-gray-700 text-white'
                          : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                      }`}
                    >
                      {match.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </nav>
        </div>

        {/* User Profile */}
        {!isCollapsed ? (
          <div className="p-4 border-t border-gray-700">
            <div className="flex items-center">
              <img 
                src={user.image}
                alt={user.name} 
                className="h-10 w-10 rounded-full"
              />
              <div className="ml-3">
                <p className="text-sm font-medium text-white">{user.name}</p>
                <p className="text-xs text-gray-400">{user.role} • {user.team}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-4 border-t border-gray-700 flex flex-col items-center">
            <img 
              src={user.image}
              alt={user.name} 
              className="h-10 w-10 rounded-full"
            />
          </div>
        )}
      </div>
    </>
  );
};

export default Sidebar; 