import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

// Backend API configuration
// TODO: Backend endpoint will be integrated here
const API_BASE_URL = 'http://localhost:8000/api';
const USE_MOCK_MODE = true; // Set to false when backend is ready

// Mock data for demonstration
const MOCK_VIDEO = {
  name: 'tennis_match.mp4',
  size: '128MB',
  duration: '2:15:30'
};

const VideoUpload = ({ onUploadComplete }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);

  // Real upload function (ready for backend integration)
  const uploadToBackend = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Upload video
      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const uploadData = await uploadResponse.json();
      const videoId = uploadData.video_id;

      // Start processing
      const processResponse = await fetch(`${API_BASE_URL}/process/${videoId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: file.name }),
      });

      if (!processResponse.ok) {
        throw new Error('Processing failed to start');
      }

      // Poll for status
      return videoId;
    } catch (error) {
      throw error;
    }
  };

  // Mock upload function for demonstration
  const mockUpload = async () => {
    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 10;
      });
    }, 300);

    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setProcessingStage('Processing video...');
    // Simulate processing stages with realistic timing
    const stages = [
      { stage: 'Detecting court...', duration: 2000 },
      { stage: 'Tracking players...', duration: 3000 },
      { stage: 'Analyzing ball movement...', duration: 2500 },
      { stage: 'Generating analytics...', duration: 2000 }
    ];
    
    for (const { stage, duration } of stages) {
      setProcessingStage(stage);
      await new Promise(resolve => setTimeout(resolve, duration));
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setSelectedFile(file);
    setIsUploading(true);
    setError(null);
    setProcessingStage('Uploading video...');
    
    try {
      if (USE_MOCK_MODE) {
        // Use mock upload for demonstration
        await mockUpload();
      } else {
        // Use real backend (when ready)
        await uploadToBackend(file);
      }

      onUploadComplete && onUploadComplete();
    } catch (error) {
      console.error('Upload failed:', error);
      setError(error.message || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      setProcessingStage('');
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi']
    },
    maxFiles: 1
  });

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-900 bg-opacity-20' : 'border-gray-600 hover:border-blue-400'}`}
      >
        <input {...getInputProps()} />
        
        {!isUploading ? (
          <div className="space-y-4">
            <div className="text-6xl mb-4">🎾</div>
            <h3 className="text-xl font-semibold text-white">
              {isDragActive ? 'Drop your tennis match video here' : 'Drag & drop your tennis match video'}
            </h3>
            <p className="text-gray-400">or click to select a file</p>
            <p className="text-sm text-gray-500">Supported formats: MP4, MOV, AVI (Max 500MB)</p>
            
            {selectedFile && (
              <div className="mt-4 p-4 bg-secondary rounded-lg">
                <p className="text-sm text-gray-400">Selected file:</p>
                <p className="text-sm font-medium text-white">{selectedFile.name}</p>
                <p className="text-xs text-gray-500">
                  Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-2xl font-semibold text-white">{processingStage}</div>
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-400">Please wait while we process your video...</p>
          </div>
        )}
      </div>
      
      {error && (
        <div className="mt-4 p-4 bg-red-900 bg-opacity-30 border border-red-600 rounded-lg">
          <p className="text-sm text-red-200">❌ {error}</p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload; 