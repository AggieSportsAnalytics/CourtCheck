'use client';

import { useRouter } from 'next/navigation';
import VideoUpload from '@/components/features/upload/VideoUpload';

export default function UploadPage() {
  const router = useRouter();

  const handleUploadComplete = () => {
    router.push('/recordings');
  };

  const features = [
    {
      title: 'Ball Tracking',
      description: 'YOLO-based detection tracks every shot and bounce throughout the match',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <circle cx="12" cy="12" r="9" strokeWidth={1.5} />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3a9 9 0 019 9" />
        </svg>
      ),
    },
    {
      title: 'Court Detection',
      description: 'Keypoint detection maps court lines and perspective in real time',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16M12 3v18" />
        </svg>
      ),
    },
    {
      title: 'Player Movement',
      description: 'Track positioning, court coverage, and generate player heatmaps',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      ),
    },
    {
      title: 'AI Scouting Report',
      description: 'GPT-4o mini generates a personalised coaching report from your data',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="px-6 py-8">
      <div className="max-w-2xl mx-auto">
        {/* Page header */}
        <div className="mb-8">
          <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: '#B4F000' }}>
            Analyze
          </p>
          <h1 className="text-3xl font-black tracking-tight text-white">Upload a Match</h1>
          <p className="text-sm mt-1" style={{ color: '#5A5A66' }}>
            Drop in your video and get full AI analysis of your match.
          </p>
        </div>

        <VideoUpload onUploadComplete={handleUploadComplete} />

        {/* Feature list */}
        <div
          className="mt-8 rounded-2xl p-6"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
        >
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-5" style={{ color: '#5A5A66' }}>
            What we analyze
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {features.map((feat) => (
              <div key={feat.title} className="flex items-start gap-3">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                  style={{ background: 'rgba(180,240,0,0.1)', color: '#B4F000', border: '1px solid rgba(180,240,0,0.15)' }}
                >
                  {feat.icon}
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-white">{feat.title}</h5>
                  <p className="text-xs mt-0.5 leading-relaxed" style={{ color: '#5A5A66' }}>{feat.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
