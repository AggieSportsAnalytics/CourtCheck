"use client";

interface TennisCourtProps {
  showHeatmap?: boolean;
  showMovement?: boolean;
  showShots?: boolean;
  className?: string;
}

export function TennisCourt({
  showHeatmap = false,
  showMovement = false,
  showShots = false,
  className = "",
}: TennisCourtProps) {
  return (
    <div className={`relative ${className}`}>
      <svg viewBox="0 0 240 140" className="w-full h-full">
        {/* Court outline */}
        <rect
          x="10"
          y="10"
          width="220"
          height="120"
          fill="#9acd32"
          stroke="#1a1a2e"
          strokeWidth="2"
        />

        {/* Net line (center vertical) */}
        <line x1="120" y1="10" x2="120" y2="130" stroke="#1a1a2e" strokeWidth="2" />

        {/* Service boxes */}
        <line x1="60" y1="10" x2="60" y2="130" stroke="#1a1a2e" strokeWidth="1" />
        <line x1="180" y1="10" x2="180" y2="130" stroke="#1a1a2e" strokeWidth="1" />

        {/* Center service line */}
        <line x1="60" y1="70" x2="180" y2="70" stroke="#1a1a2e" strokeWidth="1" />

        {/* Baseline center marks */}
        <line x1="10" y1="70" x2="20" y2="70" stroke="#1a1a2e" strokeWidth="1" />
        <line x1="220" y1="70" x2="230" y2="70" stroke="#1a1a2e" strokeWidth="1" />

        {/* Heat map overlay */}
        {showHeatmap && (
          <>
            <ellipse cx="90" cy="90" rx="25" ry="20" fill="rgba(239, 68, 68, 0.6)" />
            <ellipse cx="75" cy="50" rx="15" ry="12" fill="rgba(239, 68, 68, 0.4)" />
            <ellipse cx="150" cy="70" rx="20" ry="15" fill="rgba(239, 68, 68, 0.5)" />
          </>
        )}

        {/* Shot markers */}
        {showShots && (
          <>
            <circle cx="45" cy="45" r="4" fill="#ef4444" />
            <circle cx="95" cy="35" r="4" fill="#ef4444" />
            <circle cx="75" cy="85" r="4" fill="#ef4444" />
            <circle cx="55" cy="105" r="4" fill="#ef4444" />
            <circle cx="165" cy="55" r="4" fill="#ef4444" />
            <circle cx="185" cy="95" r="4" fill="#ef4444" />
            <circle cx="145" cy="75" r="4" fill="#ef4444" />
          </>
        )}

        {/* Movement tracking lines */}
        {showMovement && (
          <>
            <path
              d="M 30 100 Q 50 80, 70 60 T 90 40"
              fill="none"
              stroke="#1a1a2e"
              strokeWidth="2"
              strokeDasharray="4,2"
            />
            <circle cx="30" cy="100" r="6" fill="#1a1a2e" />
            <circle cx="90" cy="40" r="6" fill="#ef4444" />
          </>
        )}
      </svg>
    </div>
  );
}
