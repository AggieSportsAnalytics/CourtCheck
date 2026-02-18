import DashboardStats from '@/components/features/dashboard/DashboardStats';
import HeatMaps from '@/components/features/dashboard/HeatMaps';
import PlayerStats from '@/components/features/dashboard/PlayerStats';
import ShotPercentages from '@/components/features/dashboard/ShotPercentages';
import GamePlay from '@/components/features/dashboard/GamePlay';
import GameStatistics from '@/components/features/dashboard/GameStatistics';

export default function DashboardPage() {
  return (
    <div className="px-5 py-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Dashboard</h2>
        <p className="text-sm text-gray-400 mt-1">Your tennis analytics overview</p>
      </div>

      {/* Top stats strip: shots / bounces / rallies / court accuracy */}
      <DashboardStats />

      <div className="grid gap-4 pb-8">
        {/* Row 1: Heatmaps + Stroke Breakdown + Shot Quality */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3">
            <HeatMaps />
          </div>
          <div className="lg:col-span-2 grid gap-4 content-start">
            {/* Stroke Breakdown (forehand / backhand / serve) */}
            <PlayerStats />
            {/* Shot Quality (in-bounds vs out-of-bounds) */}
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
