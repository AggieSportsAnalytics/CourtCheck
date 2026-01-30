import React, { useState, useCallback } from 'react';

const VideoUpload = ({ onUploadComplete, apiUrl }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoId, setVideoId] = useState(null);
  const [error, setError] = useState(null);

  // Local backend API URL
  const API_BASE_URL = apiUrl || 'http://localhost:8000';

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleUpload(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleUpload(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleUpload = async (file) => {
    setSelectedFile(file);
    setIsUploading(true);
    setError(null);
    setProcessingStage('Uploading video...');
    setUploadProgress(0);

    try {
      // Upload video
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const uploadData = await uploadResponse.json();
      setVideoId(uploadData.video_id);
      setUploadProgress(50);
      setProcessingStage('Starting processing...');

      // Start processing
      const processResponse = await fetch(
        `${API_BASE_URL}/api/process/${uploadData.video_id}?filename=${encodeURIComponent(file.name)}`,
        {
          method: 'POST',
        }
      );

      if (!processResponse.ok) {
        throw new Error('Processing failed to start');
      }

      setUploadProgress(60);
      setProcessingStage('Processing video...');

      // Poll for status
      await pollStatus(uploadData.video_id);

    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.message || 'Upload failed. Please try again.');
      setIsUploading(false);
    }
  };

  const pollStatus = async (videoId) => {
    const maxAttempts = 120; // 10 minutes (5 second intervals)
    let attempts = 0;

    const checkStatus = async () => {
      try {
        const statusResponse = await fetch(`${API_BASE_URL}/api/status/${videoId}`);
        const statusData = await statusResponse.json();

        if (statusData.status === 'completed') {
          setProcessingStage('Processing complete!');
          setUploadProgress(100);
          setTimeout(() => {
            setIsUploading(false);
            onUploadComplete && onUploadComplete({
              videoId: videoId,
              downloadUrl: `${API_BASE_URL}/api/download/${videoId}`,
              analytics: statusData.result || {}
            });
          }, 1000);
        } else if (statusData.status === 'processing') {
          attempts++;
          if (attempts < maxAttempts) {
            // Update progress based on time
            const progress = 60 + (attempts / maxAttempts) * 35;
            setUploadProgress(Math.min(95, progress));
            setTimeout(checkStatus, 5000); // Check every 5 seconds
          } else {
            throw new Error('Processing timeout');
          }
        } else {
          throw new Error('Unknown status');
        }
      } catch (err) {
        setError('Failed to check processing status. Please try again.');
        setIsUploading(false);
      }
    };

    checkStatus();
  };

  return (
    <div style={{ width: '100%', maxWidth: '672px', margin: '0 auto', padding: '24px' }}>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: '2px dashed #d1d5db',
          borderRadius: '8px',
          padding: '32px',
          textAlign: 'center',
          cursor: isUploading ? 'default' : 'pointer',
          backgroundColor: isUploading ? '#f9fafb' : 'white',
          transition: 'all 0.3s'
        }}
      >
        <input
          type="file"
          id="video-upload"
          accept="video/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          disabled={isUploading}
        />
        
        {!isUploading ? (
          <label htmlFor="video-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <div style={{ fontSize: '60px', marginBottom: '16px' }}>🎾</div>
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#374151', marginBottom: '12px' }}>
              Drag & drop your tennis match video
            </h3>
            <p style={{ color: '#6b7280', marginBottom: '8px' }}>or click to select a file</p>
            <p style={{ fontSize: '14px', color: '#9ca3af' }}>Supported formats: MP4, MOV, AVI</p>
            
            {error && (
              <div style={{ 
                marginTop: '16px', 
                padding: '16px', 
                backgroundColor: '#fef2f2', 
                border: '1px solid #fecaca', 
                borderRadius: '8px' 
              }}>
                <p style={{ fontSize: '14px', color: '#dc2626' }}>{error}</p>
              </div>
            )}
          </label>
        ) : (
          <div>
            <div style={{ fontSize: '24px', fontWeight: '600', color: '#374151', marginBottom: '16px' }}>
              {processingStage}
            </div>
            <div style={{ 
              width: '100%', 
              backgroundColor: '#e5e7eb', 
              borderRadius: '9999px', 
              height: '10px',
              marginBottom: '16px'
            }}>
              <div
                style={{ 
                  backgroundColor: '#2563eb', 
                  height: '10px', 
                  borderRadius: '9999px', 
                  transition: 'width 0.3s',
                  width: `${uploadProgress}%`
                }}
              ></div>
            </div>
            <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '8px' }}>
              {selectedFile && `Processing ${selectedFile.name}...`}
            </p>
            <p style={{ fontSize: '12px', color: '#9ca3af' }}>
              This may take several minutes depending on video length.
            </p>
          </div>
        )}
      </div>

      {videoId && !isUploading && (
        <div style={{ 
          marginTop: '16px', 
          padding: '16px', 
          backgroundColor: '#f0fdf4', 
          border: '1px solid #bbf7d0', 
          borderRadius: '8px' 
        }}>
          <p style={{ fontSize: '14px', color: '#15803d' }}>
            Video processed successfully! Video ID: {videoId}
          </p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
