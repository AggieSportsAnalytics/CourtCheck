"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const monthlyData = [
  { month: "Jan", wins: 4, losses: 2 },
  { month: "Feb", wins: 6, losses: 3 },
  { month: "Mar", wins: 5, losses: 4 },
  { month: "Apr", wins: 8, losses: 2 },
  { month: "May", wins: 7, losses: 3 },
  { month: "Jun", wins: 9, losses: 1 },
];

const performanceData = [
  { name: "Aces", value: 45, color: "#9acd32" },
  { name: "Double Faults", value: 12, color: "#ef4444" },
  { name: "Winners", value: 78, color: "#3b82f6" },
  { name: "Unforced Errors", value: 34, color: "#f59e0b" },
];

export default function OverallStatsPage() {
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

        <h2 className="text-xl font-semibold text-foreground">Overall Statistics</h2>

        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground">Total Matches</p>
              <p className="text-3xl font-bold text-foreground">47</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground">Win Rate</p>
              <p className="text-3xl font-bold text-court-green">68%</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground">Hours Played</p>
              <p className="text-3xl font-bold text-foreground">32.5</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <p className="text-sm text-muted-foreground">Current Streak</p>
              <p className="text-3xl font-bold text-court-green">5W</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">Monthly Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={monthlyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="month" tick={{ fontSize: 12 }} stroke="#6b7280" />
                    <YAxis tick={{ fontSize: 12 }} stroke="#6b7280" />
                    <Line
                      type="monotone"
                      dataKey="wins"
                      stroke="#9acd32"
                      strokeWidth={2}
                      dot={{ fill: "#9acd32" }}
                      name="Wins"
                    />
                    <Line
                      type="monotone"
                      dataKey="losses"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={{ fill: "#ef4444" }}
                      name="Losses"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-6 mt-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-court-green" />
                  <span className="text-sm text-muted-foreground">Wins</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <span className="text-sm text-muted-foreground">Losses</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">Shot Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceData}>
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#6b7280" />
                    <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {performanceData.map((entry, index) => (
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
    </DashboardLayout>
  );
}
