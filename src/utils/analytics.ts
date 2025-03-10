/**
 * Utility functions for Google Analytics
 */

// Track page views when routes change in Next.js App Router
export const trackPageView = () => {
  if (typeof window !== 'undefined' && window.nextGAPageView) {
    window.nextGAPageView();
  }
};

// Track custom events
export const trackEvent = (
  action: string, 
  category: string, 
  label?: string, 
  value?: number,
  nonInteraction: boolean = false
) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
      non_interaction: nonInteraction
    });
  }
};

// Set user properties - use sparingly and never with PII
export const setUserProperty = (name: string, value: string) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('set', 'user_properties', {
      [name]: value
    });
  }
};
