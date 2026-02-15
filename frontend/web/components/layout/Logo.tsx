import Image from 'next/image';

const Logo = () => {
  const ballLogoUrl = "https://raw.githubusercontent.com/AggieSportsAnalytics/CourtCheck/cory/images/courtcheck_ball_logo.png";

  return (
    <div className="flex items-center">
      {/* Tennis Ball Logo */}
      <div className="relative w-12 h-12 shrink-0 mr-2">
        <Image
          src={ballLogoUrl}
          alt="CourtCheck tennis ball logo"
          width={48}
          height={48}
          className="object-contain"
          style={{ filter: 'brightness(1.1)' }}
        />
      </div>

      {/* CourtCheck Text */}
      <span className="text-2xl font-bold">CourtCheck</span>
    </div>
  );
};

export default Logo;
