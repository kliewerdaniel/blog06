import { Metadata } from "next";
import { Suspense } from "react";
import { ClientContactContent } from './client-content';

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
        {/* Use Suspense for components that use search params */}
        <Suspense fallback={<div>Loading...</div>}>
          <ClientContactContent />
        </Suspense>
      </div>
    </div>
  );
}
