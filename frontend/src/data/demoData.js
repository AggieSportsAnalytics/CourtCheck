// Demo Match Data
export const matchData = [
  {
    id: 'match_001',
    title: 'UC Davis vs. Hawaii',
    date: '2023-04-15',
    location: 'UC Davis Tennis Courts',
    players: {
      team1: {
        name: 'Me',
        team: 'UC Davis',
        tempId: 3
      },
      team2: {
        name: 'Opponent',
        team: 'Hawaii',
        tempId: 4
      }
    },
    stats: {
      // Core metrics from README
      bounceDetections: 187,
      bounceAccuracy: 94.2, // percentage
      ballTrackingFrames: 2341,
      servesDetected: 68,
      rallies: 42,
      avgRallyLength: 6.3, // hits
      longestRally: 21, // hits
      player1Score: 86.5, // New metric for player 1
      player2Score: 78.3, // New metric for player 2
      playerMovement: {
        team1: {
          totalDistance: 1872, // meters
          sprintCount: 34,
          courtCoverage: 68.4 // percentage
        },
        team2: {
          totalDistance: 1924, // meters
          sprintCount: 38,
          courtCoverage: 73.1 // percentage
        }
      }
    },
    analysis: {
      // Generated insights mentioned in README
      heatmaps: {
        bounces: `./assets/tennis_heatmap_bounces.jpg`,
        team1Movement: `./assets/tennis_heatmap_player1.jpg`,
        team2Movement: `./assets/tennis_heatmap_player2.jpg`
      },
      videos: {
        withOverlay: '/assets/videos/match_001_overlay.mp4',
        minimap: '/assets/videos/match_001_minimap.mp4'
      },
      courtAlignment: {
        accuracy: 98.6, // percentage
        framesCovered: 3142,
        referencePoints: 14
      },
      matchSummary: "Player showed strong coverage on baseline, but struggled with cross-court returns. Bounce analysis shows most points scored on opponent's backhand side. Movement patterns suggest opportunity to improve net approach positioning."
    }
  },
  {
    id: 'match_002',
    title: 'UC Davis vs. Stanford',
    date: '2023-03-28',
    location: 'Stanford Tennis Center',
    players: {
      team1: {
        name: 'Me',
        team: 'UC Davis',
        tempId: 2
      },
      team2: {
        name: 'Opponent',
        team: 'Stanford',
        tempId: 7
      }
    },
    stats: {
      bounceDetections: 142,
      bounceAccuracy: 96.1,
      ballTrackingFrames: 1845,
      servesDetected: 52,
      rallies: 38,
      avgRallyLength: 5.7,
      longestRally: 18,
      player1Score: 82.7, // New metric for player 1
      player2Score: 74.9, // New metric for player 2
      playerMovement: {
        team1: {
          totalDistance: 1654,
          sprintCount: 29,
          courtCoverage: 65.2
        },
        team2: {
          totalDistance: 1721,
          sprintCount: 33,
          courtCoverage: 69.8
        }
      }
    },
    analysis: {
      heatmaps: {
        bounces: `./assets/tennis_heatmap_bounces.jpg`,
        team1Movement: `./assets/tennis_heatmap_player1.jpg`,
        team2Movement: `./assets/tennis_heatmap_player2.jpg`
      },
      videos: {
        withOverlay: '/assets/videos/match_002_overlay.mp4',
        minimap: '/assets/videos/match_002_minimap.mp4'
      },
      courtAlignment: {
        accuracy: 97.9,
        framesCovered: 2857,
        referencePoints: 14
      },
      matchSummary: "Strong serving performance with effective placement. Opponent was able to exploit lateral movement weaknesses. Ball trajectory analysis shows success with deep baseline shots forcing opponent out of position."
    }
  },
  {
    id: 'match_003',
    title: 'UC Davis vs. UC Berkeley',
    date: '2023-02-17',
    location: 'UC Davis Tennis Courts',
    players: {
      team1: {
        name: 'Me',
        team: 'UC Davis',
        tempId: 3
      },
      team2: {
        name: 'Opponent',
        team: 'UC Berkeley',
        tempId: 8
      }
    },
    stats: {
      bounceDetections: 165,
      bounceAccuracy: 93.8,
      ballTrackingFrames: 2103,
      servesDetected: 62,
      rallies: 41,
      avgRallyLength: 6.8,
      longestRally: 23,
      player1Score: 89.2, // New metric for player 1
      player2Score: 81.5, // New metric for player 2
      playerMovement: {
        team1: {
          totalDistance: 1742,
          sprintCount: 31,
          courtCoverage: 67.3
        },
        team2: {
          totalDistance: 1876,
          sprintCount: 36,
          courtCoverage: 72.5
        }
      }
    },
    analysis: {
      heatmaps: {
        bounces: `./assets/tennis_heatmap_bounces.jpg`,
        team1Movement: `./assets/tennis_heatmap_player1.jpg`,
        team2Movement: `./assets/tennis_heatmap_player2.jpg`
      },
      videos: {
        withOverlay: '/assets/videos/match_003_overlay.mp4',
        minimap: '/assets/videos/match_003_minimap.mp4'
      },
      courtAlignment: {
        accuracy: 98.2,
        framesCovered: 2984,
        referencePoints: 14
      },
      matchSummary: "Effective positioning at the net led to multiple winning points. Ball tracking shows consistent depth on groundstrokes. Opponent struggled with serves to the backhand side, presenting a tactical advantage."
    }
  }
]; 