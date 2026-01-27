import React, { useState } from 'react';
import Logo from './Logo';

const Sidebar = ({ username, onGameSelect, onNavSelect }) => {
  const [activeNav, setActiveNav] = useState('Match Stats');
  
  const navItems = [
    { 
      name: 'Dashboard', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
      )
    },
    { 
      name: 'Overall Stats', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      )
    },
    { 
      name: 'Opponents', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      )
    },
    { 
      name: 'Recordings', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      )
    },
    { 
      name: 'Friends', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      )
    },
    { 
      name: 'Match Stats', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
  ];

  const games = [
    { 
      name: 'Game_01', 
      isActive: true,
      subItems: [
        { name: 'Heat Maps', isActive: true },
        { name: 'Player-Movement', isActive: false },
        { name: 'Shot Percentages', isActive: false }
      ]
    },
    { 
      name: 'Game_02', 
      isActive: false,
      subItems: []
    }
  ];

  const utilities = [
    { 
      name: 'Payment plans', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
        </svg>
      )
    },
    { 
      name: 'Referrals', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
        </svg>
      )
    },
    { 
      name: 'Settings', 
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )
    },
  ];

  const handleNavClick = (itemName) => {
    setActiveNav(itemName);
    onNavSelect(itemName);
  };

  const handleGameClick = (gameName) => {
    onGameSelect(gameName);
  };

  return (
    <aside className="w-56 bg-primary text-white flex flex-col">
      {/* Logo */}
      <div className="p-4">
        <Logo />
      </div>

      {/* Main Navigation */}
      <nav className="flex-grow p-4">
        <ul className="space-y-2">
          {navItems.map((item, idx) => (
            <li key={idx}>
              <button 
                onClick={() => handleNavClick(item.name)}
                className={`flex items-center p-2 rounded-lg w-full ${
                  activeNav === item.name ? 'bg-primary text-accent' : 'hover:bg-primary hover:bg-opacity-60'
                }`}
              >
                {item.icon}
                <span className="ml-3">{item.name}</span>
                {activeNav === item.name && item.name === 'Match Stats' && <span className="ml-auto">▲</span>}
              </button>
              
              {/* Render game submenu if this is the active Match Stats item */}
              {item.name === 'Match Stats' && activeNav === item.name && (
                <ul className="pl-6 mt-2 space-y-1">
                  {games.map((game, gameIdx) => (
                    <li key={gameIdx}>
                      <button 
                        onClick={() => handleGameClick(game.name)}
                        className={`flex items-center p-2 rounded-lg w-full text-left ${
                          game.isActive ? 'text-accent' : 'hover:text-accent'
                        }`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                        <span className="ml-2">{game.name}</span>
                      </button>
                      
                      {/* Sub-items for Game_01 */}
                      {game.isActive && game.subItems.length > 0 && (
                        <ul className="pl-4 mt-1 space-y-1">
                          {game.subItems.map((subItem, subIdx) => (
                            <li key={subIdx}>
                              <button 
                                onClick={() => handleGameClick(`${game.name}_${subItem.name}`)}
                                className={`flex items-center p-2 text-sm rounded-lg w-full text-left ${
                                  subItem.isActive ? 'text-accent' : 'hover:text-accent'
                                }`}
                              >
                                <span className="ml-2">{subItem.name}</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>

        {/* Divider */}
        <div className="my-4 border-t border-gray-700"></div>

        {/* Utilities */}
        <ul className="space-y-2">
          {utilities.map((item, idx) => (
            <li key={idx}>
              <button 
                onClick={() => handleNavClick(item.name)}
                className={`flex items-center p-2 rounded-lg w-full ${
                  activeNav === item.name ? 'bg-primary text-accent' : 'hover:bg-primary hover:bg-opacity-60'
                }`}
              >
                {item.icon}
                <span className="ml-3">{item.name}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar; 