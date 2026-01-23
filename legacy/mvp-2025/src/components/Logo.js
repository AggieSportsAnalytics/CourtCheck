import React from 'react';

const Logo = () => {
  const ballLogoUrl = "https://raw.githubusercontent.com/AggieSportsAnalytics/CourtCheck/cory/images/courtcheck_ball_logo.png";
  
  return (
    <div className="flex items-center">
      {/* Tennis Ball Logo */}
      <div className="relative w-[48px] h-[48px] flex-shrink-0 mr-2">
        <img 
          src={ballLogoUrl} 
          alt="Tennis Ball" 
          className="w-full h-full object-contain"
          style={{ filter: 'brightness(1.1)' }}
        />
      </div>
      
      {/* CourtCheck Text */}
      <span className="text-2xl font-bold">CourtCheck</span>
    </div>
  );
};

export default Logo; 