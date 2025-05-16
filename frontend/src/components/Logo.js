import React from 'react';

const Logo = ({ size = 'md', className = '' }) => {
  const sizeClass = {
    'sm': 'h-6',
    'md': 'h-8',
    'lg': 'h-12',
    'xl': 'h-16',
    '2xl': 'h-20'
  }[size] || 'h-8';
  
  return (
    <img 
      src="/assets/courtcheck_ball_logo.png"
      alt="CourtCheck" 
      className={`${sizeClass} object-contain ${className}`}
    />
  );
};

export default Logo; 