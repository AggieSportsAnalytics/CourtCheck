# CourtCheck Frontend

This is the frontend application for CourtCheck, a tennis match analysis tool that provides visual and statistical insights from match footage.

## Overview

The CourtCheck frontend is built using React and Tailwind CSS, providing a modern and responsive user interface for visualizing tennis match data including:

- Heat maps of court usage and player movement
- Shot analysis and statistics
- Match history and game play duration
- Player-specific analytics

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory:
```
cd frontend/courtcheck-ui
```
3. Install dependencies:
```
npm install
```
or
```
yarn
```

### Development

To run the development server:

```
npm start
```
or
```
yarn start
```

The application will be available at `http://localhost:3000`

## Project Structure

- `src/components/` - Reusable UI components
- `src/pages/` - Page components for different views
- `src/assets/` - Static assets like images and icons

## Features

- **Dashboard**: Overview of tennis match statistics and recent activity
- **Heat Maps**: Visualizations of court usage and player movement
- **Shot Analysis**: Breakdown of shot types, success rates, and tendencies
- **Game Statistics**: Detailed match statistics and performance metrics
- **Player Tracking**: Movement patterns and position analysis

## Integration with Backend

The frontend communicates with the CourtCheck backend which processes tennis match videos using computer vision and machine learning to extract insights. The UI visualizes this data to provide coaches and players with actionable insights.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 