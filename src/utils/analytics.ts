/**
 * Utility functions for Google Tag Manager
 */

// Initialize dataLayer if it doesn't exist
const initDataLayer = () => {
  if (typeof window !== 'undefined') {
    window.dataLayer = window.dataLayer || [];
  }
};

// Track page views when routes change in Next.js App Router
export const trackPageView = () => {
  if (typeof window !== 'undefined') {
    initDataLayer();
    
    // Push page view event to dataLayer
    window.dataLayer.push({
      event: 'page_view',
      page_title: document.title,
      page_location: window.location.href,
      page_path: window.location.pathname
    });
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
  if (typeof window !== 'undefined') {
    initDataLayer();
    
    // Push custom event to dataLayer
    window.dataLayer.push({
      event: action,
      event_category: category,
      event_label: label,
      value: value,
      non_interaction: nonInteraction
    });
  }
};

// Set user properties - use sparingly and never with PII
export const setUserProperty = (name: string, value: string) => {
  if (typeof window !== 'undefined') {
    initDataLayer();
    
    // Push user property to dataLayer
    window.dataLayer.push({
      event: 'set_user_property',
      user_property_name: name,
      user_property_value: value
    });
  }
};

// Declare global dataLayer type
declare global {
  interface Window {
    dataLayer: any[];
  }
}
