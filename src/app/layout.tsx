import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import "../styles/codeHighlight.css";

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
        <Header />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}

function Header() {
  return (
    <header className="border-b border-border">
      <div className="container flex justify-between items-center" style={{ height: 'var(--header-height)' }}>
        <Link href="/" className="font-bold text-foreground hover:text-primary">
          Daniel Kliewer
        </Link>
        <nav>
          <ul className="flex gap-6">
            <li>
              <Link href="/" className="hover:text-primary transition-colors">
                Home
              </Link>
            </li>
            <li>
              <Link href="/about" className="hover:text-primary transition-colors">
                About
              </Link>
            </li>
            <li>
              <Link href="/blog" className="hover:text-primary transition-colors">
                Blog
              </Link>
            </li>
            <li>
              <Link href="/contact" className="hover:text-primary transition-colors">
                Contact
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
    <footer className="border-t border-border py-8 mt-12">
      <div className="container">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <p className="text-muted-foreground">© {currentYear} Daniel Kliewer. All rights reserved.</p>
          </div>
          <div className="flex gap-4">
            <a href="https://github.com/kliewerdaniel" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-muted-foreground hover:text-primary transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
              </svg>
            </a>
            <a href="mailto:danielkliewer@gmail.com" aria-label="Email" className="text-muted-foreground hover:text-primary transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                <polyline points="22,6 12,13 2,6"></polyline>
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
