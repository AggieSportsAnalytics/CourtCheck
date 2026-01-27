import React, { useState, useCallback } from 'react';

const VideoUpload = ({ onUploadComplete, apiUrl }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoId, setVideoId] = useState(null);
  const [error, setError] = useState(null);

  // Use environment variable or prop for API URL
  const API_BASE_URL = apiUrl || process.env.REACT_APP_API_URL || 'https://your-modal-app-url.modal.run';

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
              downloadUrl: `${API_BASE_URL}/api/download/${videoId}`
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
    <div className="w-full max-w-2xl mx-auto p-6">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isUploading ? 'border-gray-300 bg-gray-50' : 'border-gray-300 hover:border-blue-400'}`}
      >
        <input
          type="file"
          id="video-upload"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
          disabled={isUploading}
        />
        
        {!isUploading ? (
          <label htmlFor="video-upload" className="cursor-pointer block space-y-4">
            <div className="text-6xl mb-4">🎾</div>
            <h3 className="text-xl font-semibold text-gray-700">
              Drag & drop your tennis match video
            </h3>
            <p className="text-gray-500">or click to select a file</p>
            <p className="text-sm text-gray-400">Supported formats: MP4, MOV, AVI</p>
            
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </label>
        ) : (
          <div className="space-y-4">
            <div className="text-2xl font-semibold text-gray-700">{processingStage}</div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-500">
              {selectedFile && `Processing ${selectedFile.name}...`}
            </p>
            <p className="text-xs text-gray-400">This may take several minutes depending on video length.</p>
          </div>
        )}
      </div>

      {videoId && !isUploading && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-sm text-green-700">
            Video processed successfully! Video ID: {videoId}
          </p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
