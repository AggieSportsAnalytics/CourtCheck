'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useVideoUpload } from '@/hooks';
import { APP_CONFIG } from '@/constants';

interface VideoUploadProps {
  onUploadComplete?: () => void;
}

const VideoUpload = ({ onUploadComplete }: VideoUploadProps) => {
  const { match_id, status, progress, error, handleFile, getStatusMessage } = useVideoUpload(onUploadComplete);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setSelectedFile(file);
    await handleFile(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': APP_CONFIG.SUPPORTED_VIDEO_FORMATS
    },
    maxFiles: 1,
    disabled: status !== 'idle',
    maxSize: APP_CONFIG.MAX_VIDEO_SIZE
  });

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${status !== 'idle' ? 'cursor-not-allowed opacity-60' : ''}
          ${isDragActive ? 'border-blue-500 bg-blue-900 bg-opacity-20' : 'border-gray-600 hover:border-blue-400'}`}
        role="button"
        aria-label="Upload tennis match video"
      >
        <input {...getInputProps()} aria-label="Video file input" />

        {status === 'idle' ? (
          <div className="space-y-4">
            <div className="text-6xl mb-4">🎾</div>
            <h3 className="text-xl font-semibold text-white">
              {isDragActive ? 'Drop your tennis match video here' : 'Drag & drop your tennis match video'}
            </h3>
            <p className="text-gray-400">or click to select a file</p>
            <p className="text-sm text-gray-500">Supported formats: MP4, MOV, AVI (Max 500MB)</p>

            {selectedFile && status === 'idle' && (
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
            <div className="text-2xl font-semibold text-white">{getStatusMessage()}</div>
            <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-out"
                style={{
                  width: (status === 'processing' || status === 'pending') ? `${progress * 100}%` :
                         status === 'done' ? '100%' : '0%'
                }}
              ></div>
            </div>
            {(status === 'processing' || status === 'pending') && (
              <p className="text-sm text-gray-400">
                {Math.round(progress * 100)}%
              </p>
            )}
            {match_id && (
              <p className="text-xs text-gray-500">Match ID: {match_id}</p>
            )}
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
