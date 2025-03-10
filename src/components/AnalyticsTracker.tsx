'use client';

import { usePageTracking } from '@/hooks/usePageTracking';

/**
 * A client component that handles page view tracking for analytics
 * This component should be included in the root layout to track all page views
 */
export default function AnalyticsTracker() {
  // Use the custom hook to track page views
  usePageTracking();
  
  // This component doesn't render anything to the DOM
  return null;
}
