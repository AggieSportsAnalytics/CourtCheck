import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Create root for React 18
const container = document.getElementById('root');
const root = createRoot(container);

// Render the app
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
); 