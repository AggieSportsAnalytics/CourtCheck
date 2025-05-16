import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

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

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setSelectedFile(file);
    setIsUploading(true);
    setProcessingStage('Uploading video...');
    
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

    try {
      // Simulate API call
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

      onUploadComplete && onUploadComplete();
    } catch (error) {
      console.error('Upload failed:', error);
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
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}
      >
        <input {...getInputProps()} />
        
        {!isUploading ? (
          <div className="space-y-4">
            <div className="text-6xl mb-4">🎾</div>
            <h3 className="text-xl font-semibold text-gray-700">
              {isDragActive ? 'Drop your tennis match video here' : 'Drag & drop your tennis match video'}
            </h3>
            <p className="text-gray-500">or click to select a file</p>
            <p className="text-sm text-gray-400">Supported formats: MP4, MOV, AVI</p>
            
            {/* Demo video info */}
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Demo video available:</p>
              <p className="text-sm font-medium text-gray-700">{MOCK_VIDEO.name}</p>
              <p className="text-xs text-gray-500">Size: {MOCK_VIDEO.size} • Duration: {MOCK_VIDEO.duration}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-2xl font-semibold text-gray-700">{processingStage}</div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-500">Please wait while we process your video...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoUpload; 