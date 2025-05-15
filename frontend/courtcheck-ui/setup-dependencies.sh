#!/bin/bash

# Make sure we're in the right directory
cd "$(dirname "$0")"

echo "Installing dependencies for CourtCheck UI..."

# Check if npm is installed
if ! command -v npm &> /dev/null
then
    echo "npm could not be found. Please install Node.js and npm first."
    echo "Visit https://nodejs.org/ for installation instructions."
    exit 1
fi

# Install dependencies
npm install

# Success message
echo ""
echo "Dependencies installed successfully!"
echo ""
echo "To start the development server, run:"
echo "  cd frontend/courtcheck-ui"
echo "  npm start"
echo ""
echo "The application will be available at http://localhost:3000" 