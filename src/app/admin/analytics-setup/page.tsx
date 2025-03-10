import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Google Analytics 4 Setup Guide",
  description: "Admin documentation for setting up and customizing Google Analytics 4 on this Next.js blog",
  robots: {
    index: false,
    follow: false,
  }
};

export default function AnalyticsSetupGuide() {
  return (
    <div className="container max-w-4xl py-12">
      <h1 className="text-3xl font-bold mb-8">Google Analytics 4 Setup Guide</h1>
      
      <div className="prose prose-invert max-w-none">
        <div className="bg-amber-950/30 border border-amber-800/50 rounded-lg p-4 mb-8">
          <p className="text-amber-200 font-medium">
            This is an admin page with implementation details for the website owner. This page should not be linked from public navigation.
          </p>
        </div>

        <h2 className="text-2xl font-semibold mt-8 mb-4">Getting Started</h2>
        <ol className="list-decimal pl-6 mb-6 space-y-4">
          <li>
            <strong>Create a GA4 Property:</strong> Go to <a href="https://analytics.google.com/" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Google Analytics</a> and create a new GA4 property for your website.
          </li>
          <li>
            <strong>Get Your Measurement ID:</strong> In your GA4 property, go to Admin → Data Streams → Web → [Your Stream] and copy the Measurement ID (format: G-XXXXXXXXXX).
          </li>
          <li>
            <strong>Update Your Code:</strong> Open <code>src/app/layout.tsx</code> and replace the placeholder <code>G-XXXXXXXXXX</code> with your actual Measurement ID in the GoogleAnalytics component:
            <pre className="bg-gray-800 p-4 rounded-md overflow-x-auto">
              {`<GoogleAnalytics measurementId="G-XXXXXXXXXX" />`}
            </pre>
          </li>
        </ol>

        <h2 className="text-2xl font-semibold mt-10 mb-4">Configuration Overview</h2>
        <p>
          This implementation includes the following privacy-focused features:
        </p>

        <ul className="list-disc pl-6 mb-6 space-y-3">
          <li>
            <strong>Cookie Consent Banner</strong> - Analytics tracking only occurs after explicit user consent
          </li>
          <li>
            <strong>Do Not Track Support</strong> - Automatically respects browser DNT settings
          </li>
          <li>
            <strong>IP Anonymization</strong> - All IP addresses are anonymized
          </li>
          <li>
            <strong>Enhanced Privacy Settings</strong> - Google signals, ad personalization, and data sharing have been disabled
          </li>
          <li>
            <strong>Restricted Data Processing</strong> - Enabled to comply with privacy regulations
          </li>
          <li>
            <strong>SPA Tracking</strong> - Custom implementation to track page views in Next.js App Router
          </li>
        </ul>

        <h2 className="text-2xl font-semibold mt-10 mb-4">Implementation Details</h2>
        <p>The GA4 implementation consists of several files:</p>

        <div className="mt-6 space-y-4">
          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">GoogleAnalytics.tsx</h3>
            <p className="text-gray-300 mb-2">Client component that handles loading Google Analytics scripts and manages cookie consent.</p>
            <p className="text-gray-400">Location: <code>src/components/GoogleAnalytics.tsx</code></p>
          </div>

          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">AnalyticsTracker.tsx</h3>
            <p className="text-gray-300 mb-2">Client component that tracks page views as users navigate through the site.</p>
            <p className="text-gray-400">Location: <code>src/components/AnalyticsTracker.tsx</code></p>
          </div>

          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">usePageTracking.tsx</h3>
            <p className="text-gray-300 mb-2">Custom hook that detects route changes and triggers analytics events.</p>
            <p className="text-gray-400">Location: <code>src/hooks/usePageTracking.tsx</code></p>
          </div>

          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">analytics.ts</h3>
            <p className="text-gray-300 mb-2">Utility functions for tracking page views and custom events.</p>
            <p className="text-gray-400">Location: <code>src/utils/analytics.ts</code></p>
          </div>

          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">middleware.ts</h3>
            <p className="text-gray-300 mb-2">Edge middleware to help handle page transitions for analytics.</p>
            <p className="text-gray-400">Location: <code>src/middleware.ts</code></p>
          </div>

          <div className="border border-gray-700 rounded-md p-4">
            <h3 className="text-lg font-medium mb-2">privacy-policy/page.tsx</h3>
            <p className="text-gray-300 mb-2">Privacy policy page that explains data collection practices to users.</p>
            <p className="text-gray-400">Location: <code>src/app/privacy-policy/page.tsx</code></p>
          </div>
        </div>

        <h2 className="text-2xl font-semibold mt-10 mb-4">Using Analytics Events</h2>
        <p>
          You can track custom events in your components using the utility functions from <code>src/utils/analytics.ts</code>:
        </p>

        <div className="bg-gray-800 p-4 rounded-md overflow-x-auto mt-4 mb-6">
          <pre>{`import { trackEvent } from '@/utils/analytics';

// In your component
function handleButtonClick() {
  // Do something...
  
  // Track the event
  trackEvent(
    'button_click',       // Event action
    'user_engagement',    // Event category
    'signup_button',      // Event label (optional)
    1,                    // Event value (optional)
    false                 // Non-interaction event (optional, default: false)
  );
}

// Set user properties (use sparingly and never with PII)
import { setUserProperty } from '@/utils/analytics';

setUserProperty('preferred_theme', 'dark');`}</pre>
        </div>

        <h2 className="text-2xl font-semibold mt-10 mb-4">Debugging</h2>
        <p>
          To verify your GA4 implementation is working correctly:
        </p>

        <ol className="list-decimal pl-6 mb-6 space-y-3">
          <li>
            Install the <a href="https://chrome.google.com/webstore/detail/google-analytics-debugger/jnkmfdileelhofjcijamephohjechhna" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Google Analytics Debugger</a> Chrome extension
          </li>
          <li>
            Use the GA4 DebugView in your Google Analytics Admin panel
          </li>
          <li>
            Check the browser console for any errors related to analytics
          </li>
          <li>
            Verify events are appearing in your GA4 Real-Time reports after accepting cookies
          </li>
        </ol>

        <h2 className="text-2xl font-semibold mt-10 mb-4">Compliance Considerations</h2>
        <p>
          This implementation focuses on privacy by:
        </p>

        <ul className="list-disc pl-6 mb-6 space-y-2">
          <li>Only loading GA after explicit user consent</li>
          <li>Respecting Do Not Track browser settings</li>
          <li>Anonymizing all IP addresses</li>
          <li>Using restricted data processing mode</li>
          <li>Providing a detailed privacy policy</li>
          <li>Allowing users to easily withdraw consent</li>
        </ul>

        <p>
          Depending on your region, you may need additional compliance measures for regulations like GDPR, CCPA, or other privacy laws.
        </p>

        <div className="mt-12 border-t border-gray-700 pt-8">
          <p className="text-sm text-gray-400">
            Last updated: March 10, 2025
          </p>
        </div>
      </div>
    </div>
  );
}
