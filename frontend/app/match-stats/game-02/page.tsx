"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TennisCourt } from "@/components/tennis-court";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

const shotPercentagesData = [
  { name: "Baseline", value: 45, color: "#9acd32" },
  { name: "Net Shots", value: 20, color: "#3b82f6" },
  { name: "Lob Shots", value: 15, color: "#f59e0b" },
  { name: "Drop Shots", value: 20, color: "#ef4444" },
];

export default function GameStatsPage() {
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
          <h2 className="text-xl font-semibold text-foreground mb-4">Game_02 Statistics</h2>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Court Visualization */}
          <Card>
            <CardContent className="p-6">
              <TennisCourt showMovement showShots className="h-64" />
            </CardContent>
          </Card>

          {/* Player Movement */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-medium">Player Movement</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline" className="text-xs">
                    Game_01 Opponent
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="aspect-video bg-secondary rounded-lg flex items-center justify-center">
                    <div className="w-12 h-16 bg-muted-foreground/20 rounded-lg" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Heat Maps */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-medium">Heat Maps</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="secondary" className="bg-court-green text-navy">
                    Winner
                  </Badge>
                  <Badge variant="outline">Player</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4 mb-4">
                <Badge variant="outline" className="cursor-pointer">Set 1</Badge>
                <Badge variant="outline" className="cursor-pointer">Set 2</Badge>
                <Badge variant="outline" className="cursor-pointer">Set 3</Badge>
                <Badge variant="outline" className="cursor-pointer">Overall</Badge>
              </div>
              <TennisCourt showHeatmap className="h-48" />
            </CardContent>
          </Card>

          {/* Shot Percentages */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-medium">Shot Percentages</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline" className="cursor-pointer">Set 1</Badge>
                  <Badge variant="outline" className="cursor-pointer">Set 2</Badge>
                  <Badge variant="secondary" className="bg-court-green text-navy">Overall</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-8">
                <ResponsiveContainer width={200} height={200}>
                  <PieChart>
                    <Pie
                      data={shotPercentagesData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      dataKey="value"
                      startAngle={90}
                      endAngle={-270}
                    >
                      {shotPercentagesData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-3">
                  {shotPercentagesData.map((item, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm text-muted-foreground">{item.name}</span>
                      <span className="text-sm font-medium text-foreground ml-auto">
                        {item.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
