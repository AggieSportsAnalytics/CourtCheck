"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from "recharts";

const gameHistoryData = [
  { name: "Mar 1-7", value: 2 },
  { name: "Mar 8-14", value: 1.2 },
  { name: "Mar 15-21", value: 0.8 },
  { name: "Mar 22-28", value: 1.5 },
  { name: "First Week", value: 2 },
];

const winRateData = [
  { name: "Wins", value: 62, color: "#9acd32" },
  { name: "Losses", value: 38, color: "#e5e7eb" },
];

const pointsByShotsData = [
  { name: "Serve", value: 120, color: "#9acd32" },
  { name: "Forehand", value: 95, color: "#22c55e" },
  { name: "Backhand", value: 80, color: "#ef4444" },
  { name: "Volley", value: 45, color: "#f59e0b" },
];

const historyPlayers = [
  { name: "Brian Le", matches: 3, avatar: "BL" },
  { name: "Gerald F.", matches: 2, avatar: "GF" },
  { name: "Sarah M.", matches: 5, avatar: "SM" },
  { name: "Tom K.", matches: 1, avatar: "TK" },
];

export default function Dashboard() {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-foreground flex items-center gap-2">
              <span className="text-yellow-500">*</span> Hey Cory!
            </h1>
            <p className="text-muted-foreground mt-1">
              You Recorded <span className="font-semibold text-foreground">5.89 Hrs</span> of
              Game-Play This Month!
            </p>
          </div>
          <Badge variant="secondary" className="bg-court-green/20 text-court-green border-0">
            Last 30 Days
          </Badge>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Main content area */}
          <div className="col-span-9 space-y-6">
            {/* Game History Chart */}
            <Card>
              <CardContent className="p-6">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={gameHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#6b7280" />
                      <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#9acd32"
                        strokeWidth={2}
                        dot={{ fill: "#9acd32", strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Win Rate and Points by Shots */}
            <div className="grid grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base font-medium">Win Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center">
                    <div className="relative">
                      <ResponsiveContainer width={180} height={180}>
                        <PieChart>
                          <Pie
                            data={winRateData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            dataKey="value"
                            startAngle={90}
                            endAngle={-270}
                          >
                            {winRateData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-3xl font-bold text-foreground">62%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-center gap-8 mt-4">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-court-green" />
                      <span className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">8</span> Wins
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <span className="text-sm text-muted-foreground">
                        <span className="font-semibold text-foreground">13</span> Games Lost
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base font-medium">Points by Shots</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={pointsByShotsData} layout="vertical">
                        <XAxis type="number" hide />
                        <YAxis
                          type="category"
                          dataKey="name"
                          tick={{ fontSize: 12 }}
                          width={70}
                          stroke="#6b7280"
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {pointsByShotsData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-sm">
                    <p className="text-muted-foreground">
                      Total number of shots: <span className="font-semibold text-foreground">306</span>
                    </p>
                    <div className="flex flex-wrap gap-4 mt-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-court-green" />
                        <span className="text-xs text-muted-foreground">Serve</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-green-500" />
                        <span className="text-xs text-muted-foreground">Forehand</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-red-500" />
                        <span className="text-xs text-muted-foreground">Backhand</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-yellow-500" />
                        <span className="text-xs text-muted-foreground">Volley</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Sidebar - History */}
          <div className="col-span-3">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="text-base font-medium">History</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {historyPlayers.map((player, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-3 p-2 rounded-lg hover:bg-secondary transition-colors cursor-pointer"
                  >
                    <Avatar className="w-10 h-10">
                      <AvatarImage src={`/placeholder-${index}.jpg`} alt={player.name} />
                      <AvatarFallback className="bg-navy text-white text-sm">
                        {player.avatar}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="text-sm font-medium text-foreground">{player.name}</p>
                      <p className="text-xs text-muted-foreground">{player.matches} matches</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
