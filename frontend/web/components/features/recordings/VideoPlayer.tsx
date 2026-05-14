'use client';

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Minimize,
  Gauge,
  PictureInPicture2,
} from 'lucide-react';

interface VideoPlayerProps {
  src: string;
  title?: string;
  /** Optional poster shown before metadata loads. */
  poster?: string;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

const SPEED_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2] as const;

/**
 * Branded custom video player. Replaces the native `<video controls>` so the
 * browser's "Download / Playback speed / Picture in Picture" right-click menu
 * doesn't show up — controls + context menu are owned by us.
 *
 * Exposes the underlying HTMLVideoElement via forwardRef so callers (notes
 * panel, keypoint timeline) can seek/observe currentTime without prop drilling.
 */
const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(function VideoPlayer(
  { src, title, poster },
  externalRef,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  useImperativeHandle(externalRef, () => videoRef.current as HTMLVideoElement, []);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [speedOpen, setSpeedOpen] = useState(false);
  const [isPip, setIsPip] = useState(false);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const supportsPip = useMemo(() => {
    if (typeof document === 'undefined') return false;
    return Boolean(document.pictureInPictureEnabled);
  }, []);

  const resetHideTimer = useCallback(() => {
    setShowControls(true);
    if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
    if (isPlaying && !speedOpen) {
      hideTimeoutRef.current = setTimeout(() => setShowControls(false), 3000);
    }
  }, [isPlaying, speedOpen]);

  useEffect(() => {
    if (!isPlaying || speedOpen) {
      setShowControls(true);
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
    } else {
      resetHideTimer();
    }
  }, [isPlaying, speedOpen, resetHideTimer]);

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onEnter = () => setIsPip(true);
    const onLeave = () => setIsPip(false);
    video.addEventListener('enterpictureinpicture', onEnter);
    video.addEventListener('leavepictureinpicture', onLeave);
    return () => {
      video.removeEventListener('enterpictureinpicture', onEnter);
      video.removeEventListener('leavepictureinpicture', onLeave);
    };
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, []);

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Number(e.target.value);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    const val = Number(e.target.value);
    video.volume = val;
    setVolume(val);
    setIsMuted(val === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setIsMuted(video.muted);
  };

  const toggleFullscreen = () => {
    const container = containerRef.current;
    if (!container) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void container.requestFullscreen();
    }
  };

  const togglePip = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !supportsPip) return;
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else {
        await video.requestPictureInPicture();
      }
    } catch {
      /* user-gesture or fullscreen conflict — ignore */
    }
  }, [supportsPip]);

  const applySpeed = (rate: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = rate;
    setSpeed(rate);
    setSpeedOpen(false);
  };

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const video = videoRef.current;
      if (!video) return;

      switch (e.key) {
        case ' ':
        case 'k':
          e.preventDefault();
          togglePlay();
          break;
        case 'ArrowRight':
          e.preventDefault();
          video.currentTime = Math.min(video.currentTime + 5, video.duration);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          video.currentTime = Math.max(video.currentTime - 5, 0);
          break;
        case 'f':
          e.preventDefault();
          toggleFullscreen();
          break;
        case 'm':
          e.preventDefault();
          toggleMute();
          break;
        case ',': {
          e.preventDefault();
          const idx = SPEED_OPTIONS.indexOf(speed as (typeof SPEED_OPTIONS)[number]);
          if (idx > 0) applySpeed(SPEED_OPTIONS[idx - 1]);
          break;
        }
        case '.': {
          e.preventDefault();
          const idx = SPEED_OPTIONS.indexOf(speed as (typeof SPEED_OPTIONS)[number]);
          if (idx >= 0 && idx < SPEED_OPTIONS.length - 1) applySpeed(SPEED_OPTIONS[idx + 1]);
          break;
        }
      }
    },
    [togglePlay, speed],
  );

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;
  const speedLabel = speed === 1 ? '1×' : `${speed}×`;

  return (
    <div
      ref={containerRef}
      className="relative bg-black rounded-[10px] overflow-hidden aspect-video focus:outline-none group"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseMove={resetHideTimer}
      onMouseLeave={() => isPlaying && !speedOpen && setShowControls(false)}
      onContextMenu={(e) => e.preventDefault()}
    >
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        className="w-full h-full cursor-pointer"
        controlsList="nodownload noremoteplayback noplaybackrate"
        disablePictureInPicture={false}
        x-webkit-airplay="deny"
        onClick={togglePlay}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onRateChange={() => {
          const r = videoRef.current?.playbackRate;
          if (typeof r === 'number' && Number.isFinite(r)) setSpeed(r);
        }}
        onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
        onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
        onContextMenu={(e) => e.preventDefault()}
        playsInline
      />

      {/* Big play button overlay when paused */}
      {!isPlaying && (
        <button
          type="button"
          onClick={togglePlay}
          aria-label="Play"
          className="absolute inset-0 flex items-center justify-center bg-black/30 transition-opacity"
        >
          <div className="w-16 h-16 rounded-full bg-paper/95 flex items-center justify-center shadow-lg">
            <Play className="w-7 h-7 text-ink ml-[3px]" />
          </div>
        </button>
      )}

      {/* Controls overlay */}
      <div
        className={`absolute bottom-0 inset-x-0 pt-12 pb-3 px-4 transition-opacity duration-200 ${
          showControls ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        style={{
          background:
            'linear-gradient(to top, color-mix(in srgb, black 78%, transparent) 0%, color-mix(in srgb, black 32%, transparent) 60%, transparent 100%)',
        }}
      >
        {/* Seek bar */}
        <div className="relative w-full h-[3px] bg-white/25 rounded-full cursor-pointer mb-3 group/seek">
          <div
            className="absolute top-0 left-0 h-full rounded-full pointer-events-none"
            style={{
              width: `${progressPercent}%`,
              background: 'var(--color-court)',
            }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full opacity-0 group-hover/seek:opacity-100 transition-opacity pointer-events-none"
            style={{
              left: `calc(${progressPercent}% - 6px)`,
              background: 'var(--color-court)',
              boxShadow: '0 0 0 2px color-mix(in srgb, white 90%, transparent)',
            }}
          />
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.05}
            value={currentTime}
            onChange={handleSeek}
            aria-label="Seek"
            className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>

        {/* Bottom controls row */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={togglePlay}
              aria-label={isPlaying ? 'Pause' : 'Play'}
              className="text-white/90 hover:text-white transition-colors"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button
              type="button"
              onClick={toggleMute}
              aria-label={isMuted ? 'Unmute' : 'Mute'}
              className="text-white/90 hover:text-white transition-colors"
            >
              {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              aria-label="Volume"
              className="w-20 h-1 hidden sm:block"
              style={{ accentColor: 'var(--color-court)' }}
            />
            <span
              className="text-white/85 text-[0.78rem] font-mono tabular-nums whitespace-nowrap"
              style={{ fontFeatureSettings: '"tnum"' }}
            >
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {title && (
              <span className="text-white/65 text-sm truncate max-w-[200px] hidden md:inline">
                {title}
              </span>
            )}

            {/* Speed control */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setSpeedOpen((v) => !v)}
                aria-label="Playback speed"
                aria-expanded={speedOpen}
                className="text-white/90 hover:text-white transition-colors inline-flex items-center gap-1 font-mono text-[0.72rem] tabular-nums"
              >
                <Gauge className="w-[18px] h-[18px]" />
                <span>{speedLabel}</span>
              </button>
              {speedOpen && (
                <div
                  role="menu"
                  className="absolute bottom-full right-0 mb-2 min-w-[112px] rounded-[8px] py-1 shadow-lg z-10"
                  style={{
                    background: 'color-mix(in srgb, black 88%, transparent)',
                    border: '1px solid color-mix(in srgb, white 14%, transparent)',
                    backdropFilter: 'blur(8px)',
                  }}
                  onMouseLeave={() => setSpeedOpen(false)}
                >
                  {SPEED_OPTIONS.map((rate) => {
                    const active = rate === speed;
                    return (
                      <button
                        key={rate}
                        type="button"
                        role="menuitemradio"
                        aria-checked={active}
                        onClick={() => applySpeed(rate)}
                        className={`w-full text-left px-3 py-1.5 text-[0.82rem] font-mono tabular-nums transition-colors ${
                          active ? 'text-white' : 'text-white/70 hover:text-white'
                        }`}
                        style={{
                          background: active
                            ? 'color-mix(in srgb, var(--color-court) 30%, transparent)'
                            : 'transparent',
                        }}
                      >
                        {rate === 1 ? 'Normal' : `${rate}×`}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {supportsPip && (
              <button
                type="button"
                onClick={togglePip}
                aria-label={isPip ? 'Exit picture in picture' : 'Picture in picture'}
                className="text-white/90 hover:text-white transition-colors"
              >
                <PictureInPicture2 className="w-[18px] h-[18px]" />
              </button>
            )}

            <button
              type="button"
              onClick={toggleFullscreen}
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              className="text-white/90 hover:text-white transition-colors"
            >
              {isFullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

export default VideoPlayer;
