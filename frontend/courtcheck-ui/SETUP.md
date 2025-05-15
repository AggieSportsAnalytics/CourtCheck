# CourtCheck UI - Setup Guide

This guide will help you set up the CourtCheck UI development environment.

## Prerequisites

Make sure you have the following installed on your system:

- [Node.js](https://nodejs.org/) (v14 or later)
- npm (comes with Node.js) or [Yarn](https://yarnpkg.com/)

## Quick Setup

### Option 1: Using the setup script

1. Make sure you're in the CourtCheck project root
2. Run the setup script:
   ```bash
   ./frontend/courtcheck-ui/setup-dependencies.sh
   ```
3. Start the development server:
   ```bash
   cd frontend/courtcheck-ui
   npm start
   ```

### Option 2: Manual setup

1. Navigate to the project directory:
   ```bash
   cd frontend/courtcheck-ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```
   or if using Yarn:
   ```bash
   yarn install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   or with Yarn:
   ```bash
   yarn start
   ```

4. Open your browser and visit: http://localhost:3000

## Project Structure

- `public/` - Static assets and HTML template
- `src/` - React source code
  - `components/` - Reusable UI components
  - `App.js` - Main application component
  - `index.js` - Application entry point

## Build for Production

To build the application for production:

```bash
npm run build
```

This will create a `dist` folder with optimized production files.

## Technology Stack

- React 18
- Tailwind CSS
- Webpack
- Babel
- Chart.js for data visualization 