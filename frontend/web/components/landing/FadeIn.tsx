'use client';

import { useEffect, useRef, useState, CSSProperties, ReactNode } from 'react';

interface FadeInProps {
  children: ReactNode;
  delay?: number;
  className?: string;
  y?: number;
}

export function FadeIn({ children, delay = 0, className = '', y = 20 }: FadeInProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const style: CSSProperties = {
    opacity: visible ? 1 : 0,
    transform: visible ? 'translateY(0px)' : `translateY(${y}px)`,
    transition: `opacity 0.65s ease ${delay}ms, transform 0.65s ease ${delay}ms`,
  };

  return (
    <div ref={ref} style={style} className={className}>
      {children}
    </div>
  );
}
