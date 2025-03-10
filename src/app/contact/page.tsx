import { Metadata } from "next";
import { Suspense } from "react";
import { SearchParamsProvider } from "@/components/SearchParamsProvider";

// Separate viewport export as recommended by Next.js
export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: "Contact Daniel Kliewer | Software Engineer & AI Developer",
  description: "Get in touch with Daniel Kliewer to discuss AI projects, software development, or potential collaborations. Connect for machine learning implementations, web development, and technology solutions.",
  keywords: "contact, Daniel Kliewer, software engineer, AI developer, collaboration, machine learning, web development, hire developer",
  authors: [{ name: "Daniel Kliewer" }],
  openGraph: {
    title: "Contact Daniel Kliewer | Software Engineer & AI Developer",
    description: "Get in touch with Daniel Kliewer to discuss AI projects, software development, or potential collaborations. Connect for machine learning implementations, web development, and technology solutions.",
    url: "https://danielkliewer.com/contact",
    siteName: "Daniel Kliewer | Software Engineer & AI Developer",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Contact Daniel Kliewer | Software Engineer & AI Developer",
    description: "Get in touch with Daniel Kliewer to discuss AI projects, software development, or potential collaborations. Connect for machine learning implementations, web development, and technology solutions.",
  },
  alternates: {
    canonical: "https://danielkliewer.com/contact",
  },
};

export default function ContactPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Contact</h1>
      <div className="max-w-2xl mx-auto">
        {/* Ensure SearchParamsProvider is wrapped in Suspense to handle useSearchParams safely for SSR */}
        <Suspense fallback={<div>Loading...</div>}>
          <SearchParamsProvider>
            {(searchParams) => (
              <>
                <div className="prose prose-lg dark:prose-invert mb-12">
                  <p>
                    I&apos;m passionate about leveraging technology to solve complex problems. If you&apos;re interested in discussing:
                  </p>
                  <ul>
                    <li>Machine Learning projects and implementations</li>
                    <li>Software development best practices</li>
                    <li>Web development with modern frameworks</li>
                    <li>AI integration into existing applications</li>
                    <li>Potential collaborations or opportunities</li>
                  </ul>
                  <p>
                    I&apos;d love to connect! Whether you have a specific project in mind or just want to chat about the latest in tech, 
                    feel free to reach out through the channels below.
                  </p>
                </div>
                
                {/* Contact Information */}
                <div className="border-t border-border pt-8">
                  <h2 className="text-xl font-bold mb-4">Ways to Reach Me</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="font-medium mb-2">Email</h3>
                      <a href="mailto:danielkliewer@gmail.com" className="text-primary hover:underline">
                        danielkliewer@gmail.com
                      </a>
                    </div>
                    <div>
                      <h3 className="font-medium mb-2">GitHub</h3>
                      <a href="https://github.com/kliewerdaniel" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                        github.com/kliewerdaniel
                      </a>
                    </div>
                  </div>
                </div>
              </>
            )}
          </SearchParamsProvider>
        </Suspense>
      </div>
    </div>
  );
}
