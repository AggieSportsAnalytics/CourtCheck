import React from 'react';
import VideoUpload from './components/VideoUpload';
import AnalyticsDisplay from './components/AnalyticsDisplay';

function AppSimple() {
  const [uploadedVideo, setUploadedVideo] = React.useState(null);
  const [showAnalytics, setShowAnalytics] = React.useState(false);

  const handleUploadComplete = (videoData) => {
    console.log('Upload complete:', videoData);
    setUploadedVideo(videoData);
    setShowAnalytics(true);
  };

  const handleReset = () => {
    setUploadedVideo(null);
    setShowAnalytics(false);
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6' }}>
      <header style={{ backgroundColor: 'white', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '24px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1 style={{ fontSize: '30px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
            CourtCheck - Tennis Match Analysis
          </h1>
          {showAnalytics && (
            <button
              onClick={handleReset}
              style={{
                padding: '8px 16px',
                backgroundColor: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = '#1d4ed8'}
              onMouseOut={(e) => e.target.style.backgroundColor = '#2563eb'}
            >
              Upload New Video
            </button>
          )}
        </div>
      </header>

      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 16px' }}>
        {!showAnalytics ? (
          <VideoUpload onUploadComplete={handleUploadComplete} />
        ) : (
          <AnalyticsDisplay
            analytics={uploadedVideo?.analytics}
            downloadUrl={uploadedVideo?.downloadUrl}
            videoId={uploadedVideo?.videoId}
          />
        )}
      </main>

      <footer style={{ backgroundColor: 'white', marginTop: '48px', borderTop: '1px solid #e5e7eb' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '24px 16px' }}>
          <p style={{ textAlign: 'center', fontSize: '14px', color: '#6b7280', margin: 0 }}>
            CourtCheck - AI-Powered Tennis Match Analysis
          </p>
        </div>
      </footer>
    </div>
  );
}

export default AppSimple;
