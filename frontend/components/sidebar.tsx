"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  BarChart3,
  Video,
  Users,
  Activity,
  Settings,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

const navItems = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Overall Stats", href: "/overall-stats", icon: BarChart3 },
  { name: "Recordings", href: "/recordings", icon: Video },
  {
    name: "Opponents",
    href: "/opponents",
    icon: Users,
    children: [
      { name: "Brian Le", href: "/opponents/brian-le" },
      { name: "Gerald F.", href: "/opponents/gerald-f" },
    ],
  },
  {
    name: "Match Stats",
    href: "/match-stats",
    icon: Activity,
    children: [
      { name: "Game_01", href: "/match-stats/game-01" },
      { name: "Game_02", href: "/match-stats/game-02" },
    ],
  },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [expandedItems, setExpandedItems] = useState<string[]>(["Opponents", "Match Stats"]);

  const toggleExpand = (name: string) => {
    setExpandedItems((prev) =>
      prev.includes(name) ? prev.filter((item) => item !== name) : [...prev, name]
    );
  };

  return (
    <aside className="w-56 bg-navy min-h-screen flex flex-col">
      <div className="p-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-court-green rounded-lg flex items-center justify-center">
            <span className="text-navy font-bold text-sm">CC</span>
          </div>
          <span className="text-white font-semibold text-lg">CourtCheck</span>
          <span className="text-court-green text-lg">*</span>
        </Link>
      </div>

      <nav className="flex-1 px-3 py-4">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            const isExpanded = expandedItems.includes(item.name);
            const hasChildren = item.children && item.children.length > 0;

            return (
              <li key={item.name}>
                <div
                  className={cn(
                    "flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer",
                    isActive
                      ? "bg-navy-light text-white"
                      : "text-gray-400 hover:text-white hover:bg-navy-light"
                  )}
                  onClick={() => hasChildren && toggleExpand(item.name)}
                >
                  <Link
                    href={item.href}
                    className="flex items-center gap-3 flex-1"
                    onClick={(e) => hasChildren && e.preventDefault()}
                  >
                    <item.icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </Link>
                  {hasChildren && (
                    <span className="text-gray-500">
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4" />
                      ) : (
                        <ChevronRight className="w-4 h-4" />
                      )}
                    </span>
                  )}
                </div>

                {hasChildren && isExpanded && (
                  <ul className="ml-7 mt-1 space-y-1">
                    {item.children.map((child) => {
                      const isChildActive = pathname === child.href;
                      return (
                        <li key={child.name}>
                          <Link
                            href={child.href}
                            className={cn(
                              "block px-3 py-1.5 text-sm rounded-lg transition-colors",
                              isChildActive
                                ? "text-court-green"
                                : "text-gray-500 hover:text-gray-300"
                            )}
                          >
                            {child.name}
                          </Link>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
