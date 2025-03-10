'use client';

import { useSearchParams as useNextSearchParams } from 'next/navigation';
import { ReactNode, useState, useEffect } from 'react';

interface SearchParamsProviderProps {
  children: (searchParams: ReturnType<typeof useNextSearchParams>) => ReactNode;
}

/**
 * A wrapper component that provides search parameters to its children
 * This component uses useSearchParams() which requires it to be:
 * 1. Used only in Client Components (marked with 'use client')
 * 2. Wrapped in a Suspense boundary when used in a page
 */
export function SearchParamsProvider({ children }: SearchParamsProviderProps) {
  const searchParams = useNextSearchParams();
  const [mounted, setMounted] = useState(false);
  
  // Only render on client-side to avoid hydration issues
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // During SSR or before hydration, render a placeholder
  if (!mounted) {
    return null;
  }
  
  // Once mounted on client, render with actual searchParams
  return <>{children(searchParams)}</>;
}
