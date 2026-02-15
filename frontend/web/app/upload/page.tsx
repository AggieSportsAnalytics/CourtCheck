'use client';

import { useRouter } from 'next/navigation';
import VideoUpload from '@/components/features/upload/VideoUpload';

export default function UploadPage() {
  const router = useRouter();

  const handleUploadComplete = () => {
    // Redirect to recordings page after upload completes
    router.push('/recordings');
  };

  return (
    <div className="px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold mb-2">Upload Your Tennis Match</h2>
          <p className="text-gray-400">
            Upload a video of your tennis match to analyze ball tracking, court detection, player movement, and more.
          </p>
        </div>

        <VideoUpload onUploadComplete={handleUploadComplete} />

        <div className="mt-8 p-6 bg-secondary rounded-lg">
          <h4 className="text-lg font-semibold text-white mb-4">What We Analyze:</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start">
              <span className="text-2xl mr-3">🎾</span>
              <div>
                <h5 className="font-medium text-white">Ball Tracking</h5>
                <p className="text-sm text-gray-400">Track ball movement throughout the match</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-2xl mr-3">📐</span>
              <div>
                <h5 className="font-medium text-white">Court Detection</h5>
                <p className="text-sm text-gray-400">Identify court boundaries and lines</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-2xl mr-3">🏃</span>
              <div>
                <h5 className="font-medium text-white">Player Movement</h5>
                <p className="text-sm text-gray-400">Analyze player positioning and movement patterns</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-2xl mr-3">📊</span>
              <div>
                <h5 className="font-medium text-white">Match Statistics</h5>
                <p className="text-sm text-gray-400">Generate comprehensive match analytics</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
