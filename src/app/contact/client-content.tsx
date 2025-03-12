'use client';

import { useSearchParams } from 'next/navigation';

export function ClientContactContent() {
  const searchParams = useSearchParams();

  return (
    <>
      <section className="prose prose-lg dark:prose-invert mb-12">
        <p>
          I am dedicated to leveraging technology to address complex challenges, and I welcome the opportunity to discuss:
        </p>
        <ul>
          <li>Machine Learning projects and implementations</li>
          <li>Software development best practices</li>
          <li>Web development with modern frameworks</li>
          <li>AI integration into existing applications</li>
          <li>Potential collaborations and opportunities</li>
        </ul>
        <p>
          Whether you have a specific project in mind or would simply like to explore the latest advancements in technology, please feel free to reach out using one of the contact methods below.
        </p>
      </section>
      
      <section className="border-t border-border pt-8">
        <h2 className="text-xl font-bold mb-4">Contact Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium mb-2">Email</h3>
            <a 
              href="mailto:danielkliewer@gmail.com" 
              className="text-primary hover:underline"
            >
              danielkliewer@gmail.com
            </a>
          </div>
          <div>
            <h3 className="font-medium mb-2">GitHub</h3>
            <a 
              href="https://github.com/kliewerdaniel" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-primary hover:underline"
            >
              github.com/kliewerdaniel
            </a>
          </div>
        </div>
      </section>
    </>
  );
}