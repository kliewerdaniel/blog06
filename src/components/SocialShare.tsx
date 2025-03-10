'use client';

import React from 'react';
import { usePathname } from 'next/navigation';

interface SocialShareProps {
  title: string;
  summary?: string;
  className?: string;
}

export default function SocialShare({ title, summary = '', className = '' }: SocialShareProps) {
  const pathname = usePathname();
  const url = `https://danielkliewer.com${pathname}`;
  const encodedUrl = encodeURIComponent(url);
  const encodedTitle = encodeURIComponent(title);
  const encodedSummary = encodeURIComponent(summary);
  
  const shareLinks = [
    {
      name: 'Twitter',
      href: `https://twitter.com/intent/tweet?url=${encodedUrl}&text=${encodedTitle}`,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z" />
        </svg>
      ),
      label: 'Share on Twitter',
    },
    {
      name: 'Facebook',
      href: `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
        </svg>
      ),
      label: 'Share on Facebook',
    },
    {
      name: 'LinkedIn',
      href: `https://www.linkedin.com/shareArticle?mini=true&url=${encodedUrl}&title=${encodedTitle}&summary=${encodedSummary}`,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
          <rect x="2" y="9" width="4" height="12" />
          <circle cx="4" cy="4" r="2" />
        </svg>
      ),
      label: 'Share on LinkedIn',
    },
    {
      name: 'Reddit',
      href: `https://www.reddit.com/submit?url=${encodedUrl}&title=${encodedTitle}`,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <circle cx="18" cy="6" r="2" />
          <circle cx="9" cy="9" r="1.5" />
          <circle cx="15" cy="9" r="1.5" />
          <path d="M12 16c2 0 3-1.5 3-1.5" />
          <path d="M9 16c2 0 3-1.5 3-1.5" />
        </svg>
      ),
      label: 'Share on Reddit',
    },
    {
      name: 'Email',
      href: `mailto:?subject=${encodedTitle}&body=${encodedSummary}%0A%0A${encodedUrl}`,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
          <polyline points="22,6 12,13 2,6" />
        </svg>
      ),
      label: 'Share via Email',
    },
  ];

  return (
    <div className={`flex flex-wrap items-center gap-3 ${className}`}>
      <span className="text-sm font-medium text-muted-foreground">Share:</span>
      <div className="flex flex-wrap gap-2">
        {shareLinks.map((link) => (
          <a
            key={link.name}
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={link.label}
            className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            title={link.label}
          >
            <span className="sr-only">{link.label}</span>
            {link.icon}
          </a>
        ))}
      </div>
    </div>
  );
}
