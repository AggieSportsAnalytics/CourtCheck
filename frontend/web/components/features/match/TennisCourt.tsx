import Image from 'next/image';

const TennisCourt = () => {
  return (
    <Image
      src="/tennis-court.png"
      alt="Tennis court diagram showing court boundaries and lines"
      width={800}
      height={600}
      className="w-full h-auto rounded-xl"
    />
  );
};

export default TennisCourt;
