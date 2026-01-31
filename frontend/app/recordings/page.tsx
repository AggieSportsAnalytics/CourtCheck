"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, Download, Play } from "lucide-react";
import { useState } from "react";
import { TennisCourt } from "@/components/tennis-court";

interface Recording {
  id: string;
  uploadDate: string;
  duration: string;
}

interface GameRecordings {
  gameId: string;
  gameName: string;
  recordings: Recording[];
}

const gameRecordings: GameRecordings[] = [
  {
    gameId: "game-01",
    gameName: "Game_01",
    recordings: [
      { id: "1", uploadDate: "03/15/2024", duration: "15 mins" },
      { id: "2", uploadDate: "03/12/2024", duration: "12 mins" },
    ],
  },
  {
    gameId: "game-02",
    gameName: "Game_02",
    recordings: [
      { id: "3", uploadDate: "03/05/2024", duration: "18 mins" },
      { id: "4", uploadDate: "03/03/2024", duration: "12 mins" },
    ],
  },
];

export default function RecordingsPage() {
  const [expandedGames, setExpandedGames] = useState<string[]>(["game-01", "game-02"]);

  const toggleGame = (gameId: string) => {
    setExpandedGames((prev) =>
      prev.includes(gameId) ? prev.filter((id) => id !== gameId) : [...prev, gameId]
    );
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-foreground flex items-center gap-2">
              <span className="text-yellow-500">*</span> Hey Cory!
            </h1>
          </div>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-foreground mb-4">Game Recordings</h2>
        </div>

        <div className="space-y-4">
          {gameRecordings.map((game) => (
            <div key={game.gameId} className="space-y-3">
              <button
                onClick={() => toggleGame(game.gameId)}
                className="flex items-center gap-2 text-left w-full"
              >
                <ChevronDown
                  className={`w-4 h-4 transition-transform ${
                    expandedGames.includes(game.gameId) ? "rotate-0" : "-rotate-90"
                  }`}
                />
                <span className="font-semibold text-foreground">{game.gameName}</span>
              </button>

              {expandedGames.includes(game.gameId) && (
                <div className="space-y-3 ml-6">
                  {game.recordings.map((recording) => (
                    <Card key={recording.id} className="overflow-hidden">
                      <CardContent className="p-0">
                        <div className="flex items-center gap-4">
                          <div className="relative w-40 h-24 bg-court-green flex items-center justify-center">
                            <TennisCourt className="w-full h-full p-2" />
                            <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                              <div className="w-10 h-10 rounded-full bg-white/90 flex items-center justify-center">
                                <Play className="w-5 h-5 text-navy ml-0.5" />
                              </div>
                            </div>
                          </div>
                          <div className="flex-1 py-3">
                            <p className="text-sm text-muted-foreground">
                              Uploaded: {recording.uploadDate}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              Duration: {recording.duration}
                            </p>
                          </div>
                          <div className="pr-4">
                            <Button
                              variant="outline"
                              size="sm"
                              className="text-court-green border-court-green hover:bg-court-green hover:text-navy bg-transparent"
                            >
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
