"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

const players = [
  { id: 1, name: "Player 1", lastMatch: "Game_02", winRate: 82, avatar: "P1" },
  { id: 2, name: "Player 2", lastMatch: "Game_02", winRate: 62, avatar: "P2" },
  { id: 3, name: "Player 3", lastMatch: "Game_02", winRate: 82, avatar: "P3" },
  { id: 4, name: "Player 4", lastMatch: "Game_02", winRate: 82, avatar: "P4" },
  { id: 5, name: "Player 5", lastMatch: "Game_02", winRate: 68, avatar: "P5" },
  { id: 6, name: "Player 6", lastMatch: "Game_02", winRate: 82, avatar: "P6" },
];

export default function MatchStatsPage() {
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

        <div className="grid grid-cols-3 gap-6">
          {players.map((player) => (
            <Link key={player.id} href={`/opponents/brian-le`}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer">
                <CardContent className="p-6">
                  <div className="flex flex-col items-center text-center">
                    <Avatar className="w-16 h-16 mb-3">
                      <AvatarImage src={`/placeholder-player-${player.id}.jpg`} alt={player.name} />
                      <AvatarFallback className="bg-secondary text-muted-foreground text-lg">
                        {player.avatar}
                      </AvatarFallback>
                    </Avatar>
                    <h3 className="font-semibold text-foreground">{player.name}</h3>
                    <Badge variant="secondary" className="mt-2 bg-court-green text-navy">
                      Winner Recent
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-3">
                      Last Match: <span className="text-foreground">{player.lastMatch}</span>
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Win Rate: <span className="text-foreground">{player.winRate}%</span>
                    </p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
