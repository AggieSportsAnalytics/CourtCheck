import Image from 'next/image';

interface LogoProps {
  size?: 'sm' | 'md';
}

const Logo = ({ size = 'md' }: LogoProps) => {
  const ballLogoUrl = "/courtcheck_ball_logo.png";
  const imgSize = size === 'sm' ? 28 : 40;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex items-center gap-3">
        <Image
          src={ballLogoUrl}
          alt="CourtCheck tennis ball logo"
          width={imgSize}
          height={imgSize}
          className="object-contain shrink-0"
          style={{ filter: 'brightness(1.1)' }}
        />
        <span className={`font-bold leading-none tracking-tight text-white ${size === 'sm' ? 'text-[15px]' : 'text-2xl'}`}>
          CourtCheck
        </span>
      </div>
      {size === 'md' && (
        <div>
          <span className="text-[10px] tracking-[0.18em] uppercase text-gray-500">
            Tennis Analytics
          </span>
        </div>
      )}
    </div>
  );
};

export default Logo;
