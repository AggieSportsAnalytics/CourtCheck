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
  }, [handleFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': APP_CONFIG.SUPPORTED_VIDEO_FORMATS },
    maxFiles: 1,
    disabled: status !== 'idle',
    maxSize: APP_CONFIG.MAX_VIDEO_SIZE,
  });

  const isActive = status !== 'idle';
  const isDone = status === 'done';
  const progressPct = isDone ? 100 : (status === 'processing' || status === 'pending') ? Math.round(progress * 100) : 0;

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className="relative rounded-2xl transition-all duration-200"
        role="button"
        aria-label="Upload tennis match video"
        style={{
          background: isDragActive
            ? 'rgba(180,240,0,0.04)'
            : 'rgba(255,255,255,0.02)',
          border: isDragActive
            ? '1px solid rgba(180,240,0,0.4)'
            : isActive
            ? '1px solid rgba(255,255,255,0.08)'
            : '1px dashed rgba(255,255,255,0.12)',
          cursor: isActive ? 'default' : 'pointer',
        }}
      >
        <input {...getInputProps()} aria-label="Video file input" />

        {!isActive ? (
          <div className="px-8 py-14 text-center">
            {/* Upload icon */}
            <div
              className="w-14 h-14 rounded-2xl mx-auto mb-5 flex items-center justify-center"
              style={{ background: 'rgba(180,240,0,0.08)', border: '1px solid rgba(180,240,0,0.15)' }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: '#B4F000' }}>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>

            <h3 className="text-base font-semibold text-white mb-1">
              {isDragActive ? 'Drop to upload' : 'Drop your video here'}
            </h3>
            <p className="text-sm mb-4" style={{ color: '#5A5A66' }}>
              or <span style={{ color: '#B4F000' }}>browse files</span>
            </p>
            <p className="text-xs" style={{ color: '#3A3A44' }}>
              MP4 · MOV · AVI &nbsp;·&nbsp; max 500 MB
            </p>

            {selectedFile && (
              <div
                className="mt-5 mx-auto max-w-xs rounded-xl px-4 py-3 text-left"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
              >
                <p className="text-xs text-white font-medium truncate">{selectedFile.name}</p>
                <p className="text-[11px] mt-0.5" style={{ color: '#5A5A66' }}>
                  {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="px-8 py-14 text-center">
            {isDone ? (
              <div
                className="w-14 h-14 rounded-2xl mx-auto mb-5 flex items-center justify-center"
                style={{ background: 'rgba(180,240,0,0.1)', border: '1px solid rgba(180,240,0,0.2)' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: '#B4F000' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            ) : (
              <div className="w-14 h-14 mx-auto mb-5 flex items-center justify-center">
                <div
                  className="w-10 h-10 rounded-full animate-spin"
                  style={{ border: '2px solid rgba(180,240,0,0.15)', borderTopColor: '#B4F000' }}
                />
              </div>
            )}

            <p className="text-base font-semibold text-white mb-1">{getStatusMessage()}</p>

            <div
              className="w-full max-w-xs mx-auto mt-4 rounded-full overflow-hidden h-1.5"
              style={{ background: 'rgba(255,255,255,0.08)' }}
            >
              <div
                className="h-full rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progressPct}%`, background: '#B4F000' }}
              />
            </div>

            {progressPct > 0 && (
              <p className="text-xs mt-2" style={{ color: '#5A5A66' }}>{progressPct}%</p>
            )}

            {match_id && (
              <p className="text-[10px] mt-3 font-mono" style={{ color: '#2A2A33' }}>
                {match_id.slice(0, 8)}…
              </p>
            )}
          </div>
        )}
      </div>

      {error && (
        <div
          className="mt-3 flex items-start gap-3 px-4 py-3 rounded-xl"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 shrink-0 mt-0.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
