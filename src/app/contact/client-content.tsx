'use client';

import { useSearchParams } from 'next/navigation';

export function ClientContactContent() {
  // We can use useSearchParams here since this is a client component
  const searchParams = useSearchParams();
  
  return (
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
  );
}
