import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';
import { trackPageView } from '@/utils/analytics';

/**
 * A custom hook to track page views in Google Analytics
 * This hook listens for route changes and triggers analytics tracking 
 * Only tracks if consent has been given and GA is available
 */
export function usePageTracking() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    // Track page view when the path or search params change
    if (pathname) {
      // Small delay to ensure page has fully loaded
      // This ensures accurate page titles and URLs
      setTimeout(() => {
        trackPageView();
      }, 100);
    }
  }, [pathname, searchParams]);

  return null;
}
