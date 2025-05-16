# CourtCheck Frontend

This is the frontend React application for CourtCheck, a tennis analytics platform.

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

```bash
# Install dependencies
npm install

# Start the development server
npm start
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

## Image Assets

For the tennis court heat maps to display properly, you need to add the following image files to the `public/assets` directory:

1. `tennis_heatmap_bounces.jpg` - A heat map showing ball bounce locations
2. `tennis_heatmap_player1.jpg` - A heat map showing Player 1's movement patterns
3. `tennis_heatmap_player2.jpg` - A heat map showing Player 2's movement patterns

You can use the sample images provided in the project screenshots or source your own tennis court heat map images. The images should be of a tennis court from a top-down view with heat map overlays showing intensity of activity.

## Features

- Interactive dashboard for tennis match analysis
- Heat maps for shot bounces and player movement
- Shot percentage analysis
- Player summary and statistics
- Match insights and metrics

## Integration with Backend

The frontend communicates with the CourtCheck backend which processes tennis match videos using computer vision and machine learning to extract insights. The UI visualizes this data to provide coaches and players with actionable insights.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
