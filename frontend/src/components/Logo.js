import React from 'react';

const Logo = ({ size = 'md', className = '' }) => {
  const sizeClass = {
    'sm': 'h-6 w-6',
    'md': 'h-8 w-8',
    'lg': 'h-12 w-12',
    'xl': 'h-16 w-16',
    '2xl': 'h-20 w-20'
  }[size] || 'h-8 w-8';
  
  return (
    <img 
      src="/assets/courtcheck_ball_logo.png"
      alt="CourtCheck" 
      className={`${sizeClass} ${className}`}
    />
  );
};

export default Logo; 