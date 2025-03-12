'use client';

import { Suspense } from 'react';
import { usePageTracking } from '@/hooks/usePageTracking';

/**
 * A client component that handles page view tracking for Google Tag Manager
 * This component should be included in the root layout to track all page views
 * It's wrapped in a Suspense boundary to handle useSearchParams correctly
 */
function AnalyticsTrackerInner() {
  // Use the custom hook to track page views with GTM
  usePageTracking();
  
  // This component doesn't render anything to the DOM
  return null;
}

export default function AnalyticsTracker() {
  return (
    <Suspense fallback={null}>
      <AnalyticsTrackerInner />
    </Suspense>
  );
}
