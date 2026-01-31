"use client";

import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChevronRight, ExternalLink, Target, Zap, AlertTriangle, TrendingUp } from "lucide-react";
import Link from "next/link";

const coachNotes = [
  { icon: Target, text: "Strong baseline rallies" },
  { icon: AlertTriangle, text: "Weak backhand under pressure" },
  { icon: Zap, text: "Aggressive net play after serve" },
  { icon: TrendingUp, text: "Tends to struggle in long rallies" },
];

const externalLinks = [
  {
    title: "USTA Match History",
    description: "View official match records",
    icon: "USTA",
  },
  {
    title: "ITF Player Profile",
    description: "International Tennis Federation profile",
    icon: "ITF",
  },
  {
    title: "Match Footage Archive",
    description: "Access recorded matches",
    icon: "MFA",
  },
];

const suggestedStrategies = [
  "Target backhand",
  "Extend rallies",
  "Avoid net exchanges early",
];

export default function OpponentsPage() {
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

        <div className="grid grid-cols-2 gap-6">
          {/* Coach Notes */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">Coach Notes on Opponent Team</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {coachNotes.map((note, index) => (
                  <li key={index} className="flex items-center gap-3">
                    <note.icon className="w-4 h-4 text-court-green" />
                    <span className="text-sm text-muted-foreground">{note.text}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          {/* External Links */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">External Links to Opponent Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {externalLinks.map((link, index) => (
                <Link
                  key={index}
                  href="#"
                  className="flex items-center justify-between p-3 rounded-lg border border-border hover:bg-secondary transition-colors group"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-navy flex items-center justify-center">
                      <span className="text-xs text-white font-medium">{link.icon}</span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-foreground">{link.title}</p>
                      <p className="text-xs text-muted-foreground">{link.description}</p>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                </Link>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Suggested Match Strategy */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base font-medium">Suggested Match Strategy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                {suggestedStrategies.map((strategy, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-3 p-3 rounded-lg bg-secondary"
                  >
                    <div className="w-6 h-6 rounded-full bg-court-green flex items-center justify-center">
                      <span className="text-xs text-navy font-bold">{index + 1}</span>
                    </div>
                    <span className="text-sm text-foreground">{strategy}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-3">
                <Link
                  href="#"
                  className="flex items-center gap-3 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>Info</span>
                </Link>
                {externalLinks.map((link, index) => (
                  <Link
                    key={index}
                    href="#"
                    className="flex items-center justify-between p-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <div className="w-5 h-5 rounded bg-navy flex items-center justify-center">
                        <span className="text-[10px] text-white">{link.icon}</span>
                      </div>
                      <span>{link.title}</span>
                    </div>
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
