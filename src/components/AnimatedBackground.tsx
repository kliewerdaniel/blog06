'use client';

import { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';

type AnimatedBackgroundProps = {
  type?: 'gradient' | 'particles' | 'waves' | 'geometric';
  intensity?: 'subtle' | 'moderate' | 'intense';
  colors?: string[];
  isBlackAndWhite?: boolean;
};

// A simpler, more stable AnimatedBackground component
export default function AnimatedBackground({
  type = 'geometric',
  intensity = 'moderate',
  colors = ['#000000', '#ffffff'],
  isBlackAndWhite = true,
}: AnimatedBackgroundProps) {
  // Static settings based on intensity
  const speed = {
    subtle: 30,
    moderate: 20,
    intense: 10,
  };

  // Use client-side only rendering to avoid hydration issues
  const [isClient, setIsClient] = useState(false);
  
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Don't render anything during SSR to avoid hydration mismatches
  if (!isClient) {
    return <div className="fixed top-0 left-0 w-full h-full -z-10 bg-background"></div>;
  }

  // Gradient background
  if (type === 'gradient') {
    return (
      <motion.div
        className="fixed top-0 left-0 w-full h-full -z-10 opacity-20"
        style={{
          background: `linear-gradient(45deg, ${colors.join(', ')})`,
          backgroundSize: '400% 400%',
        }}
        animate={{
          backgroundPosition: ['0% 0%', '100% 100%'],
        }}
        transition={{
          duration: speed[intensity],
          ease: 'easeInOut',
          repeat: Infinity,
          repeatType: 'mirror',
        }}
      />
    );
  }

  // Particles background - simplified and deterministic
  if (type === 'particles') {
    const particleCount = intensity === 'subtle' ? 20 : intensity === 'moderate' ? 40 : 60;
    const particles = [];
    
    for (let i = 0; i < particleCount; i++) {
      const xPos = ((i * 13) % 100);
      const yPos = ((i * 17) % 100);
      const size = 1 + ((i * 7) % 4);
      const color = colors[i % colors.length];
      
      particles.push(
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            left: `${xPos}%`,
            top: `${yPos}%`,
            width: `${size}px`,
            height: `${size}px`,
            backgroundColor: color,
            opacity: 0.2,
          }}
          animate={{
            x: [((i * 7) % 50) - 25, ((i * 11) % 50) - 25, ((i * 13) % 50) - 25],
            y: [((i * 17) % 50) - 25, ((i * 19) % 50) - 25, ((i * 23) % 50) - 25],
            scale: [1, 1.2, 1],
            opacity: [0.1, 0.2, 0.1],
          }}
          transition={{
            duration: speed[intensity],
            ease: 'easeInOut',
            repeat: Infinity,
            repeatType: 'mirror',
          }}
        />
      );
    }
    
    return (
      <div className="fixed top-0 left-0 w-full h-full -z-10 overflow-hidden bg-background">
        {particles}
      </div>
    );
  }

  // Waves background
  if (type === 'waves') {
    return (
      <div className="fixed top-0 left-0 w-full h-full -z-10 overflow-hidden bg-background">
        {colors.map((color, index) => (
          <motion.div
            key={index}
            className="absolute w-[200%] h-[50vh] rounded-[100%]"
            style={{
              bottom: `${-30 - index * 5}%`,
              left: '-50%',
              backgroundColor: color,
              opacity: 0.05,
            }}
            animate={{
              translateX: ['-10%', '10%', '-10%'],
            }}
            transition={{
              duration: speed[intensity] + index * 2,
              ease: 'easeInOut',
              repeat: Infinity,
              repeatType: 'mirror',
            }}
          />
        ))}
      </div>
    );
  }

  // Geometric patterns - simplified and deterministic
  if (type === 'geometric') {
    const patternCount = intensity === 'subtle' ? 20 : intensity === 'moderate' ? 40 : 60;
    const shapes = [];
    
    for (let i = 0; i < patternCount; i++) {
      const isEven = i % 2 === 0;
      const shapeType = i % 3; // 0: circle, 1: square, 2: triangle
      const size = 20 + ((i * 7) % 80);
      const posX = ((i * 13) % 100);
      const posY = ((i * 17) % 100);
      const rotationStart = ((i * 23) % 360);
      const rotationEnd = rotationStart + ((i * 11) % 180) - 90;
      const opacity = 0.05 + ((i * 3) % 30) / 1000;
      const duration = speed[intensity] + ((i * 7) % 50) / 10;
      const delay = ((i * 13) % 20) / 10;
      
      shapes.push(
        <motion.div
          key={i}
          className={`absolute ${shapeType === 0 ? 'rounded-full' : ''}`}
          style={{
            left: `${posX}%`,
            top: `${posY}%`,
            width: `${size}px`,
            height: `${size}px`,
            backgroundColor: isEven ? '#000000' : '#ffffff',
            opacity: opacity,
            border: `1px solid ${isEven ? '#ffffff' : '#000000'}`,
            clipPath: shapeType === 2 ? 'polygon(50% 0%, 0% 100%, 100% 100%)' : 'none',
          }}
          initial={{
            rotate: rotationStart,
          }}
          animate={{
            opacity: [0.04, 0.07, 0.04],
            scale: [1, 1.05, 1],
            rotate: rotationEnd,
          }}
          transition={{
            duration: duration,
            ease: "easeInOut",
            repeat: Infinity,
            repeatType: "mirror",
            delay: delay,
          }}
        />
      );
    }
    
    return (
      <div className="fixed top-0 left-0 w-full h-full -z-10 overflow-hidden bg-background">
        {shapes}
      </div>
    );
  }

  // Fallback
  return <div className="fixed top-0 left-0 w-full h-full -z-10 bg-background"></div>;
}
