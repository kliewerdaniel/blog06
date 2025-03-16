'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';

type PageTransitionProps = {
  children: ReactNode;
  transitionType?: 'fade' | 'slide' | 'scale' | 'none';
};

export default function PageTransition({
  children,
  transitionType = 'fade',
}: PageTransitionProps) {
  // No animation if animation type is none
  if (transitionType === 'none') {
    return <>{children}</>;
  }

  // Define variants for different animation types
  const variants = {
    fade: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
    },
    slide: {
      initial: { opacity: 0, x: 20 },
      animate: { opacity: 1, x: 0 },
      exit: { opacity: 0, x: -20 },
    },
    scale: {
      initial: { opacity: 0, scale: 0.95 },
      animate: { opacity: 1, scale: 1 },
      exit: { opacity: 0, scale: 0.95 },
    },
  };

  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={variants[transitionType]}
      transition={{
        duration: 0.3,
        ease: 'easeInOut',
      }}
    >
      {children}
    </motion.div>
  );
}
