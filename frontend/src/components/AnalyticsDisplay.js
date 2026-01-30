import React from 'react';

const AnalyticsDisplay = ({ analytics, downloadUrl, videoId }) => {
  if (!analytics) return null;

  const {
    total_frames,
    duration_seconds,
    fps,
    ball_detected_frames,
    ball_detection_rate,
    total_bounces,
    stroke_statistics,
  } = analytics;

  return (
    <div style={{ 
      backgroundColor: 'white', 
      borderRadius: '8px', 
      boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)', 
      padding: '24px', 
      marginTop: '24px' 
    }}>
      <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', color: '#1f2937' }}>
        Video Analysis Complete!
      </h2>

      {/* Download Button */}
      <div style={{ marginBottom: '24px' }}>
        <a
          href={downloadUrl}
          download={`processed_${videoId}.mp4`}
          style={{
            display: 'inline-block',
            backgroundColor: '#16a34a',
            color: 'white',
            fontWeight: 'bold',
            padding: '12px 24px',
            borderRadius: '8px',
            textDecoration: 'none',
            transition: 'background-color 0.3s'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#15803d'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#16a34a'}
        >
          Download Processed Video
        </a>
      </div>

      {/* Video Stats */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '16px', 
        marginBottom: '24px' 
      }}>
        <div style={{ backgroundColor: '#eff6ff', padding: '16px', borderRadius: '8px' }}>
          <div style={{ fontSize: '14px', color: '#4b5563' }}>Duration</div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2563eb' }}>
            {duration_seconds?.toFixed(1)}s
          </div>
        </div>

        <div style={{ backgroundColor: '#faf5ff', padding: '16px', borderRadius: '8px' }}>
          <div style={{ fontSize: '14px', color: '#4b5563' }}>Total Frames</div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#9333ea' }}>
            {total_frames?.toLocaleString()}
          </div>
        </div>

        <div style={{ backgroundColor: '#f0fdf4', padding: '16px', borderRadius: '8px' }}>
          <div style={{ fontSize: '14px', color: '#4b5563' }}>Frame Rate</div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#16a34a' }}>
            {fps?.toFixed(1)} fps
          </div>
        </div>
      </div>

      {/* Ball Tracking Stats */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#374151' }}>
          Ball Tracking
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
          <div style={{ backgroundColor: '#fefce8', padding: '16px', borderRadius: '8px' }}>
            <div style={{ fontSize: '14px', color: '#4b5563' }}>Frames with Ball Detected</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ca8a04' }}>
              {ball_detected_frames?.toLocaleString()} / {total_frames?.toLocaleString()}
            </div>
          </div>

          <div style={{ backgroundColor: '#fff7ed', padding: '16px', borderRadius: '8px' }}>
            <div style={{ fontSize: '14px', color: '#4b5563' }}>Detection Rate</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ea580c' }}>
              {((ball_detection_rate || 0) * 100).toFixed(1)}%
            </div>
            <div style={{ width: '100%', backgroundColor: '#e5e7eb', borderRadius: '9999px', height: '8px', marginTop: '8px' }}>
              <div
                style={{ 
                  backgroundColor: '#ea580c', 
                  height: '8px', 
                  borderRadius: '9999px',
                  width: `${(ball_detection_rate || 0) * 100}%`
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Bounce Detection */}
      {total_bounces !== undefined && (
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#374151' }}>
            Bounce Detection
          </h3>
          <div style={{ backgroundColor: '#ecfeff', padding: '16px', borderRadius: '8px' }}>
            <div style={{ fontSize: '14px', color: '#4b5563' }}>Total Bounces Detected</div>
            <div style={{ fontSize: '30px', fontWeight: 'bold', color: '#0891b2' }}>
              {total_bounces}
            </div>
          </div>
        </div>
      )}

      {/* Stroke Classification */}
      {stroke_statistics && Object.keys(stroke_statistics).length > 0 && (
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#374151' }}>
            Stroke Statistics
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
            {Object.entries(stroke_statistics).map(([stroke, count]) => (
              <div key={stroke} style={{ backgroundColor: '#fdf2f8', padding: '12px', borderRadius: '8px' }}>
                <div style={{ fontSize: '14px', color: '#4b5563' }}>{stroke}</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#db2777' }}>{count}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <div style={{ 
        marginTop: '24px', 
        padding: '16px', 
        backgroundColor: '#f9fafb', 
        borderRadius: '8px', 
        borderLeft: '4px solid #3b82f6' 
      }}>
        <p style={{ fontSize: '14px', color: '#374151' }}>
          <strong>Summary:</strong> Processed {total_frames?.toLocaleString()} frames
          ({duration_seconds?.toFixed(1)}s) with{' '}
          {((ball_detection_rate || 0) * 100).toFixed(1)}% ball detection rate.
          {total_bounces > 0 && ` Detected ${total_bounces} bounce${total_bounces !== 1 ? 's' : ''}.`}
          {stroke_statistics && Object.keys(stroke_statistics).length > 0 && 
            ` Classified ${Object.values(stroke_statistics).reduce((a, b) => a + b, 0)} strokes.`
          }
        </p>
      </div>
    </div>
  );
};

export default AnalyticsDisplay;
