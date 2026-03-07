import DashboardStats from '@/components/features/dashboard/DashboardStats';
import HeatMaps from '@/components/features/dashboard/HeatMaps';
import PlayerStats from '@/components/features/dashboard/PlayerStats';
import ShotPercentages from '@/components/features/dashboard/ShotPercentages';
import GamePlay from '@/components/features/dashboard/GamePlay';
import GameStatistics from '@/components/features/dashboard/GameStatistics';

export default function DashboardPage() {
  return (
    <div className="px-6 py-8 max-w-7xl">
      {/* Page header */}
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
          Overview
        </p>
        <h1 className="text-3xl font-black tracking-tight text-white">Dashboard</h1>
        <p className="text-sm mt-1" style={{ color: '#5A5A66' }}>
          Your tennis analytics at a glance
        </p>
      </div>

      {/* Top stats strip */}
      <DashboardStats />

      <div className="grid gap-4 pb-10">
        {/* Row 1: Heatmaps + Stroke Breakdown + Shot Quality */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 flex">
            <HeatMaps />
          </div>
          <div className="lg:col-span-2 grid gap-4">
            <PlayerStats />
            <ShotPercentages />
          </div>
        </div>

        {/* Row 2: Recent Sessions + Shot History chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <GamePlay />
          <GameStatistics />
        </div>
      </div>
    </div>
  );
}
