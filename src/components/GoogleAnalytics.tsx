'use client';

import { useEffect, useState } from 'react';
import Script from 'next/script';

interface GoogleAnalyticsProps {
  measurementId: string; // Your GA4 measurement ID (G-XXXXXXXXXX)
}

const GoogleAnalytics = ({ measurementId }: GoogleAnalyticsProps) => {
  const [consentGiven, setConsentGiven] = useState<boolean | null>(null);
  const [cookieBannerVisible, setCookieBannerVisible] = useState(false);

  useEffect(() => {
    // Check for Do Not Track setting
    const doNotTrack = 
      navigator.doNotTrack === '1' || 
      navigator.doNotTrack === 'yes' ||
      window.doNotTrack === '1';
    
    // Check for existing consent in localStorage
    const storedConsent = localStorage.getItem('analytics-consent');
    
    if (doNotTrack) {
      // Respect Do Not Track browser setting
      setConsentGiven(false);
    } else if (storedConsent !== null) {
      // Use stored preference if available
      setConsentGiven(storedConsent === 'true');
    } else {
      // Show cookie banner if no preference is stored
      setCookieBannerVisible(true);
    }
  }, []);

  const handleAccept = () => {
    localStorage.setItem('analytics-consent', 'true');
    setConsentGiven(true);
    setCookieBannerVisible(false);
  };

  const handleDecline = () => {
    localStorage.setItem('analytics-consent', 'false');
    setConsentGiven(false);
    setCookieBannerVisible(false);
  };

  return (
    <>
      {/* Show cookie consent banner if needed */}
      {cookieBannerVisible && (
        <div className="fixed bottom-0 left-0 right-0 bg-black/95 text-white p-4 z-50 shadow-lg border-t border-gray-700">
          <div className="container mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <p className="text-sm">
                This site uses cookies to analyze traffic and improve your experience. 
                We respect your privacy and only collect anonymized data with your consent.
              </p>
              <p className="text-xs mt-1 text-gray-400">
                <a href="/blog/privacy-policy" className="underline">View our Privacy Policy</a>
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleDecline}
                className="px-4 py-2 text-sm border border-gray-600 rounded hover:bg-gray-800 transition-colors"
              >
                Decline
              </button>
              <button
                onClick={handleAccept}
                className="px-4 py-2 text-sm bg-primary border border-primary rounded hover:bg-primary/90 transition-colors"
              >
                Accept
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Only load GA scripts if consent is given */}
      {consentGiven && (
        <>
          {/* Google Analytics initialization script */}
          <Script id="google-analytics-init" strategy="afterInteractive">
            {`
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('consent', 'default', {
                'analytics_storage': 'granted'
              });
              
              // Configure with privacy-focused settings
              gtag('config', '${measurementId}', {
                send_page_view: false,  // Disable automatic page views to control timing
                anonymize_ip: true,     // Anonymize IP addresses
                allow_google_signals: false, // Disable Google signals
                allow_ad_personalization_signals: false, // Disable ad personalization
                restricted_data_processing: true // Enable restricted data processing
              });
            `}
          </Script>

          {/* Google Analytics base script */}
          <Script 
            src={`https://www.googletagmanager.com/gtag/js?id=${measurementId}`}
            strategy="afterInteractive"
            onLoad={() => {
              // Send initial pageview after script loads
              window.gtag('js', new Date());
              window.gtag('event', 'page_view', {
                page_title: document.title,
                page_location: window.location.href,
                page_path: window.location.pathname
              });
            }}
          />

          {/* Script to track page changes in Next.js */}
          <Script id="google-analytics-nextjs-router" strategy="afterInteractive">
            {`
              if (typeof window !== 'undefined') {
                window.nextGAPageView = function() {
                  window.gtag('event', 'page_view', {
                    page_title: document.title,
                    page_location: window.location.href,
                    page_path: window.location.pathname
                  });
                }
              }
            `}
          </Script>
        </>
      )}
    </>
  );
};

export default GoogleAnalytics;

// Add global type declaration for gtag
declare global {
  interface Window {
    gtag: (
      command: string,
      action: string | Date,
      config?: Record<string, any>
    ) => void;
    nextGAPageView: () => void;
    dataLayer: any[];
    doNotTrack: string;
  }
}
