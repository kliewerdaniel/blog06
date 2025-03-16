'use client';

import { ReactNode } from 'react';
import { AnimatePresence } from 'framer-motion';
import { usePathname } from 'next/navigation';

// This component acts as a provider for AnimatePresence
// It needs to be a client component since it uses the usePathname hook
export default function MotionProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  
  return (
    <AnimatePresence mode="wait">
      {/* We use the pathname as key to trigger animations on route change */}
      <div key={pathname}>
        {children}
      </div>
    </AnimatePresence>
  );
}
