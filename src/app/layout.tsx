import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import "../styles/codeHighlight.css";
import "../styles/animations.css";
import AnalyticsTracker from "@/components/AnalyticsTracker";
import AnimatedBackground from "@/components/AnimatedBackground";
import MotionProvider from "@/components/MotionProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Separate viewport export as recommended by Next.js
export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  minimumScale: 1,
  userScalable: true,
  themeColor: "#0a0a0a",
};

export const metadata: Metadata = {
  metadataBase: new URL("https://danielkliewer.com"),
  title: {
    template: "%s | Daniel Kliewer",
    default: "Daniel Kliewer | Software Engineer & AI Developer",
  },
  description: "Portfolio and blog of Daniel Kliewer, showcasing projects in AI, web development, and software engineering. Specializing in large language models and privacy-focused solutions.",
  applicationName: "Daniel Kliewer Portfolio",
  authors: [{ name: "Daniel Kliewer", url: "https://danielkliewer.com" }],
  creator: "Daniel Kliewer",
  publisher: "Daniel Kliewer",
  formatDetection: {
    email: true,
    address: true,
    telephone: true,
  },
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://danielkliewer.com",
    siteName: "Daniel Kliewer | Software Engineer & AI Developer",
    title: "Daniel Kliewer | Software Engineer & AI Developer",
    description: "Portfolio and blog of Daniel Kliewer, showcasing projects in AI, web development, and software engineering.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Daniel Kliewer | Software Engineer & AI Developer",
    description: "Portfolio and blog of Daniel Kliewer, showcasing projects in AI, web development, and software engineering.",
    creator: "@danielkliewer",
  },
  alternates: {
    canonical: "https://danielkliewer.com",
    languages: {
      'en-US': "https://danielkliewer.com",
    },
  },
  category: "technology",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Create a stable className string that won't change between server and client renders
  const bodyClasses = [geistSans.variable, geistMono.variable, 'antialiased'].join(' ');
  
  return (
    <html lang="en">
      <head>
        {/* JavaScript Error Fix - Prevents "eP[i] is not a function" error */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
            /**
             * Enhanced fix for "eP[i] is not a function" error with animation frame handling
             */
            (function() {
                console.log("[Animation Fix] Initializing enhanced error prevention script");
                
                // Store original methods
                const originalRequestAnimationFrame = window.requestAnimationFrame;
                
                // Function to safely check if an item is a valid function
                function ensureFunctionSafety() {
                    console.log("[Animation Fix] Setting up function safety protections");
                    
                    // Protect animation frame handling with improved error prevention
                    window.requestAnimationFrame = function(callback) {
                        // Wrap the callback in a try-catch to prevent errors from bubbling up
                        const safeCallback = function(timestamp) {
                            try {
                                return callback(timestamp);
                            } catch (error) {
                                // More specific error detection
                                if (error.message && (
                                    error.message.includes('is not a function') ||
                                    error.message.includes('eP[i]') ||
                                    error.message.includes('undefined is not a function')
                                )) {
                                    console.warn("[Animation Fix] Caught error in animation frame:", error.message);
                                    return null;
                                }
                                throw error; // Re-throw other errors
                            }
                        };
                        
                        return originalRequestAnimationFrame.call(window, safeCallback);
                    };
                    
                    // Create a protective wrapper for problematic objects
                    function createFunctionSafetyProxy(obj, name) {
                        if (!obj || typeof obj !== 'object') return obj;
                        
                        return new Proxy(obj, {
                            get: function(target, prop) {
                                const value = target[prop];
                                
                                // Handle the specific case of eP array
                                if (prop === 'eP' && Array.isArray(value)) {
                                    console.log("[Animation Fix] Found eP array in " + name + ", applying protection");
                                    return createArrayFunctionSafetyProxy(value, name + ".eP");
                                }
                                
                                // Handle 'measureInitialState' method
                                if (prop === 'measureInitialState' && typeof value === 'function') {
                                    console.log("[Animation Fix] Found measureInitialState in " + name + ", applying protection");
                                    return function() {
                                        try {
                                            return value.apply(this, arguments);
                                        } catch (error) {
                                            if (error.message && error.message.includes('is not a function')) {
                                                console.warn("[Animation Fix] Caught error in " + name + ".measureInitialState:", error.message);
                                                return null; // Return a safe value instead of failing
                                            } else {
                                                throw error; // Re-throw other errors
                                            }
                                        }
                                    };
                                }
                                
                                // Handle animation-related methods (expanded to cover more potential issues)
                                if ((prop === 'oG' || prop === 'oX' || prop === 'process' || prop === 'm' || 
                                     prop === 'start' || prop === 'scheduleResolve' || prop === 'hook' ||
                                     prop === 'overrideMethod' || prop === 'eI' || prop === 'update' ||
                                     prop === 'render' || prop === 'requestAnimationFrame') && typeof value === 'function') {
                                    return function() {
                                        try {
                                            return value.apply(this, arguments);
                                        } catch (error) {
                                            if (error.message && error.message.includes('is not a function')) {
                                                console.warn("[Animation Fix] Caught error in " + name + "." + prop + ":", error.message);
                                                return null;
                                            } else {
                                                throw error;
                                            }
                                        }
                                    };
                                }
                                
                                return value;
                            }
                        });
                    }
                    
                    // Create a protective wrapper specifically for arrays that might be used as function collections
                    function createArrayFunctionSafetyProxy(array, name) {
                        if (!Array.isArray(array)) return array;
                        
                        return new Proxy(array, {
                            get: function(target, prop) {
                                // If trying to access array element by index
                                if (!isNaN(parseInt(prop))) {
                                    const index = parseInt(prop);
                                    const item = target[index];
                                    
                                    // If the item is not a function but might be called as one
                                    if (item !== undefined && typeof item !== 'function') {
                                        console.warn("[Animation Fix] Protection: " + name + "[" + index + "] is not a function (" + (typeof item) + ")");
                                        
                                        // Return a no-op function instead of the non-function item
                                        return function() {
                                            console.warn("[Animation Fix] Called " + name + "[" + index + "] safely instead of throwing error");
                                            return null; // Safe return value
                                        };
                                    }
                                }
                                
                                return target[prop];
                            }
                        });
                    }
                    
                    // Protect the window object and scan for potential issues
                    setTimeout(function() {
                        console.log("[Animation Fix] Scanning for problematic objects");
                        
                        // Look through window objects for potential matches
                        for (const key in window) {
                            try {
                                if (key.startsWith('__') || key === 'webpackChunk') continue; // Skip special objects
                                
                                const obj = window[key];
                                if (!obj || typeof obj !== 'object') continue;
                                
                                // Look for key objects that might match our error pattern
                                if (typeof obj.measureInitialState === 'function' || 
                                    (obj.eP && Array.isArray(obj.eP)) ||
                                    typeof obj.oG === 'function' ||
                                    typeof obj.oX === 'function' ||
                                    typeof obj.process === 'function' ||
                                    typeof obj.scheduleResolve === 'function') {
                                    console.log("[Animation Fix] Found potential match: window." + key);
                                    window[key] = createFunctionSafetyProxy(obj, "window." + key);
                                }
                                
                                // Also look for objects named with minified-style keys (common in bundled code)
                                if ((/^[a-zA-Z]{1,2}$/.test(key) || /^[a-zA-Z][A-Z]$/.test(key)) && typeof obj === 'object') {
                                    window[key] = createFunctionSafetyProxy(obj, "window." + key);
                                }
                                
                                // Specifically target hook.js related objects (based on error trace)
                                if (key.includes('hook') || (typeof obj === 'object' && obj.overrideMethod)) {
                                    console.log("[Animation Fix] Found hook-related object: window." + key);
                                    window[key] = createFunctionSafetyProxy(obj, "window." + key);
                                }
                            } catch (e) {
                                // Skip objects that can't be accessed
                            }
                        }
                        
                        console.log("[Animation Fix] Finished scanning for problematic objects");
                    }, 0);
                }
                
                // Start the protection immediately
                ensureFunctionSafety();
                
                // Also run after DOM is loaded to catch any objects initialized later
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', ensureFunctionSafety);
                }
                
                // Add additional safety by running after window load
                window.addEventListener('load', ensureFunctionSafety);
                
                // Run one more time after a short delay to catch dynamically loaded scripts
                setTimeout(ensureFunctionSafety, 2000);
                
                console.log("[Animation Fix] Fix script installed successfully");
            })();
            `
          }}
        />
        {/* Google Tag Manager */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
            new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
            j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
            'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
            })(window,document,'script','dataLayer','GTM-T5J5JWX');`
          }}
        />
        {/* End Google Tag Manager */}
        
        {/* JSON-LD structured data for better SEO */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "WebSite",
                  "@id": "https://danielkliewer.com/#website",
                  "url": "https://danielkliewer.com/",
                  "name": "Daniel Kliewer | Software Engineer & AI Developer",
                  "description": "Portfolio and blog of Daniel Kliewer, showcasing projects in AI, web development, and software engineering.",
                  "publisher": {
                    "@id": "https://danielkliewer.com/#person"
                  },
                  "inLanguage": "en-US",
                },
                {
                  "@type": "Person",
                  "@id": "https://danielkliewer.com/#person",
                  "name": "Daniel Kliewer",
                  "description": "Software Engineer & AI Developer with 15+ years of experience in web development, data annotation, and AI.",
                  "jobTitle": "Software Engineer & AI Developer",
                  "email": "danielkliewer@gmail.com",
                  "sameAs": [
                    "https://github.com/kliewerdaniel"
                  ],
                  "worksFor": {
                    "@id": "https://danielkliewer.com/#organization"
                  }
                },
                {
                  "@type": "Organization",
                  "@id": "https://danielkliewer.com/#organization",
                  "name": "Daniel Kliewer Projects",
                  "url": "https://danielkliewer.com/",
                  "logo": {
                    "@type": "ImageObject",
                    "url": "https://danielkliewer.com/logo.png", // Add a logo if available
                    "width": 112,
                    "height": 112
                  },
                  "founder": {
                    "@id": "https://danielkliewer.com/#person"
                  }
                }
              ]
            }),
          }}
        />
      </head>
      <body
        className={bodyClasses}
        suppressHydrationWarning={true}
      >
        {/* Google Tag Manager (noscript) */}
        <noscript>
          <iframe 
            src="https://www.googletagmanager.com/ns.html?id=GTM-T5J5JWX"
            height="0" 
            width="0" 
            style={{ display: 'none', visibility: 'hidden' }}
            title="Google Tag Manager"
          />
        </noscript>
        {/* End Google Tag Manager (noscript) */}
        
        {/* Component to track page views across route changes */}
        <AnalyticsTracker />
        
        {/* Animated Background - applied to all pages */}
        <AnimatedBackground type="geometric" intensity="moderate" colors={['#000000', '#ffffff']} isBlackAndWhite={true} />
        
        <Header />
        <MotionProvider>
          <main>{children}</main>
        </MotionProvider>
        <Footer />
      </body>
    </html>
  );
}

function Header() {
  return (
    <header className="border-b border-border backdrop-blur-sm bg-background/80">
      <div className="container flex justify-between items-center" style={{ height: 'var(--header-height)' }}>
        <Link href="/" className="font-bold text-foreground hover:text-primary link-animate">
          Daniel Kliewer
        </Link>
        
        {/* Mobile Navigation - Hamburger Menu */}
        <nav className="mobile-nav md:hidden">
          <input type="checkbox" id="nav-toggle" className="nav-toggle sr-only" />
          <label htmlFor="nav-toggle" className="nav-toggle-label" aria-label="Toggle navigation menu">
            <span className="hamburger">
              <span className="hamburger-line"></span>
              <span className="hamburger-line"></span>
              <span className="hamburger-line"></span>
            </span>
          </label>
          <ul className="mobile-menu">
            <li>
              <Link href="/" className="mobile-link">
                Home
              </Link>
            </li>
            <li>
              <Link href="/about" className="mobile-link">
                About
              </Link>
            </li>
            <li>
              <Link href="/blog" className="mobile-link">
                Blog
              </Link>
            </li>
            <li>
              <Link href="/art" className="mobile-link">
                Art
              </Link>
            </li>
          </ul>
        </nav>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:block">
          <ul className="flex gap-6 stagger-animation">
            <li>
              <Link href="/" className="hover:text-primary text-base">
                Home
              </Link>
            </li>
            <li>
              <Link href="/about" className="hover:text-primary text-base">
                About
              </Link>
            </li>
            <li>
              <Link href="/blog" className="hover:text-primary text-base">
                Blog
              </Link>
            </li>
            <li>
              <Link href="/art" className="hover:text-primary text-base">
                Art
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  // Use a static year to avoid hydration mismatch
  const currentYear = 2025;
  
  return (
    <footer className="border-t border-border py-8 mt-12 backdrop-blur-sm bg-background/80">
      <div className="container">
        <div className="flex flex-col md:flex-row justify-between items-start gap-6">
          <div>
            <p className="text-muted-foreground text-center md:text-left">© {currentYear} Daniel Kliewer. All rights reserved.</p>
          </div>
          
          <div className="w-full md:w-auto flex justify-center md:justify-end">
            <nav className="flex flex-col md:flex-row gap-4 md:gap-8">
              <div>
                <ul className="flex flex-wrap gap-4 items-center justify-center md:justify-start">
                  <li>
                    <Link href="/" className="text-muted-foreground hover:text-primary text-sm">
                      Home
                    </Link>
                  </li>
                  <li>
                    <Link href="/about" className="text-muted-foreground hover:text-primary text-sm">
                      About
                    </Link>
                  </li>
                  <li>
                    <Link href="/blog" className="text-muted-foreground hover:text-primary text-sm">
                      Blog
                    </Link>
                  </li>
                  <li>
                    <Link href="/art" className="text-muted-foreground hover:text-primary text-sm">
                      Art
                    </Link>
                  </li>
                  <li>
                    <Link href="/privacy-policy" className="text-muted-foreground hover:text-primary text-sm">
                      Privacy Policy
                    </Link>
                  </li>
                  <li>
                    <a href="https://github.com/kliewerdaniel" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-muted-foreground hover:text-primary">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                      </svg>
                    </a>
                  </li>
                  <li>
                    <a href="mailto:danielkliewer@gmail.com" aria-label="Email" className="text-muted-foreground hover:text-primary">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                        <polyline points="22,6 12,13 2,6"></polyline>
                      </svg>
                    </a>
                  </li>
                </ul>
              </div>
            </nav>
          </div>
        </div>
      </div>
    </footer>
  );
}
