'use client';

import { useRouter } from 'next/navigation';
import VideoUpload from '@/components/features/upload/VideoUpload';

export default function UploadPage() {
  const router = useRouter();

  const handleUploadComplete = () => {
    // Redirect to recordings page after upload completes
    router.push('/recordings');
  };

  const features = [
    {
      title: 'Ball Tracking',
      description: 'Track ball movement and bounce detection throughout the match',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="9" strokeWidth={1.5} />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3a9 9 0 019 9" />
        </svg>
      ),
    },
    {
      title: 'Court Detection',
      description: 'Identify court boundaries, lines, and perspective mapping',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v18" />
        </svg>
      ),
    },
    {
      title: 'Player Movement',
      description: 'Analyze player positioning and generate position heatmaps',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      ),
    },
    {
      title: 'Match Heatmaps',
      description: 'Generate ball bounce and player position heatmaps for review',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="px-5 py-8">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white">Upload Tennis Match</h2>
          <p className="text-gray-400 mt-1">
            Upload a video to analyze ball tracking, court detection, player movement, and generate heatmaps.
          </p>
        </div>

        <VideoUpload onUploadComplete={handleUploadComplete} />

        <div className="mt-8 p-5 bg-secondary rounded-xl border border-gray-700/40">
          <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            What We Analyze
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {features.map((feat) => (
              <div key={feat.title} className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0 mt-0.5">
                  {feat.icon}
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-white">{feat.title}</h5>
                  <p className="text-xs text-gray-400 mt-0.5">{feat.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
