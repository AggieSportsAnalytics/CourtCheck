import HeatMaps from '@/components/features/dashboard/HeatMaps';
import PlayerStats from '@/components/features/dashboard/PlayerStats';
import ShotPercentages from '@/components/features/dashboard/ShotPercentages';
import GamePlay from '@/components/features/dashboard/GamePlay';
import GameStatistics from '@/components/features/dashboard/GameStatistics';

export default function DashboardPage() {
  return (
    <div className="px-4 py-6">
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>

      <div className="grid gap-4 pb-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <HeatMaps />
          <div className="grid gap-4">
            <PlayerStats />
            <ShotPercentages />
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <GamePlay />
          <div className="col-span-2">
            <GameStatistics />
          </div>
        </div>
      </div>
    </div>
  );
}
