"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { TennisCourt } from "@/components/tennis-court";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
} from "recharts";
import { Target, Zap, AlertTriangle } from "lucide-react";

const strokeUsageData = [
  { name: "Serve", value: 30, color: "#1a1a2e" },
  { name: "Forehand", value: 35, color: "#9acd32" },
  { name: "Backhand", value: 25, color: "#3b82f6" },
  { name: "Other", value: 10, color: "#e5e7eb" },
];

const strokeStats = [
  { name: "Forehand", value: 116 },
  { name: "Backhand", value: 116 },
];

const shotsSummaryData = [
  { name: "Aces", value: 12, color: "#9acd32" },
  { name: "Double Faults", value: 4, color: "#ef4444" },
  { name: "Unforced Errors", value: 18, color: "#f59e0b" },
  { name: "1st Serve %", value: 65, color: "#3b82f6" },
];

const recentPerformance = [
  { label: "Aces / Match", value: "1.2", icon: "tennis" },
  { label: "Double Faults / Match", value: "4", icon: "x" },
  { label: "Unforced Errors / Match", value: "2", icon: "alert" },
  { label: "1st Serve %", value: "73%", icon: "percent" },
];

const suggestedStrategies = [
  { icon: Target, text: "Target the backhand side" },
  { icon: Zap, text: "Maintain deep baseline rallies" },
  { icon: AlertTriangle, text: "Limit double faults" },
];

export default function BrianLeStatsPage() {
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

        <h2 className="text-xl font-semibold text-foreground">{"Brian Le's Stats Overview"}</h2>

        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Player Info and Court */}
          <div className="col-span-5 space-y-6">
            {/* Player Card */}
            <Card>
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <Avatar className="w-20 h-20">
                    <AvatarImage src="/brian-le.jpg" alt="Brian Le" />
                    <AvatarFallback className="bg-court-green text-navy text-xl">BL</AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-foreground">Brian Le</h3>
                    <p className="text-sm text-muted-foreground">ID: 125664</p>
                    <div className="flex gap-2 mt-2">
                      <Badge className="bg-court-green text-navy">Active Player</Badge>
                      <Badge variant="secondary">56%</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Stroke Usage */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base font-medium">Stroke Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-6">
                  <ResponsiveContainer width={150} height={150}>
                    <PieChart>
                      <Pie
                        data={strokeUsageData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={60}
                        dataKey="value"
                        startAngle={90}
                        endAngle={-270}
                      >
                        {strokeUsageData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex-1 space-y-2">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-foreground">182</p>
                      <p className="text-xs text-muted-foreground">shots</p>
                    </div>
                    <div className="flex justify-between text-sm">
                      <div className="text-center">
                        <p className="font-semibold text-foreground">116</p>
                        <p className="text-xs text-muted-foreground">Forehand</p>
                      </div>
                      <div className="text-center">
                        <p className="font-semibold text-foreground">116</p>
                        <p className="text-xs text-muted-foreground">Backhand</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Court Visualization */}
            <Card>
              <CardContent className="p-4">
                <TennisCourt showShots className="h-48" />
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Performance Stats */}
          <div className="col-span-7 space-y-6">
            <div className="grid grid-cols-2 gap-6">
              {/* Time Stats */}
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-2xl font-bold text-foreground">3.25 Hours</p>
                      <p className="text-sm text-muted-foreground">Total Play Time</p>
                    </div>
                    <div className="w-16 h-16">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                              { value: 56, color: "#9acd32" },
                              { value: 44, color: "#e5e7eb" },
                            ]}
                            cx="50%"
                            cy="50%"
                            innerRadius={20}
                            outerRadius={30}
                            dataKey="value"
                          >
                            <Cell fill="#9acd32" />
                            <Cell fill="#e5e7eb" />
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Recent Match Performance */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Recent Match Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {recentPerformance.map((item, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-court-green" />
                        <span className="text-muted-foreground">{item.label}</span>
                        <span className="ml-auto font-medium text-foreground">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Suggested Strategy */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base font-medium">Suggested Strategy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {suggestedStrategies.map((strategy, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-3 rounded-lg bg-secondary"
                    >
                      <div className="w-6 h-6 rounded-full bg-court-green flex items-center justify-center">
                        <strategy.icon className="w-3 h-3 text-navy" />
                      </div>
                      <span className="text-sm text-foreground">{strategy.text}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Shots Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base font-medium">Shots Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shotsSummaryData}>
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#6b7280" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {shotsSummaryData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
