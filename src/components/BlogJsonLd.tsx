'use client';

interface BlogJsonLdProps {
  title: string;
  description: string;
  datePublished: string;
  authorName: string;
  slug: string;
  tags?: string[];
  categories?: string[];
}

export function BlogJsonLd({
  title,
  description,
  datePublished,
  authorName,
  slug,
  tags = [],
  categories = [],
}: BlogJsonLdProps) {
  const canonical = `https://danielkliewer.com/blog/${slug}`;
  
  // Format combined keywords from tags and categories
  const keywords = [...tags, ...categories].join(', ');
  
  // Create schema.org Article JSON-LD structured data
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': canonical,
    },
    headline: title,
    description: description,
    image: [
      'https://danielkliewer.com/og-image.jpg', // Default OG image
    ],
    datePublished: datePublished,
    dateModified: datePublished, // Assuming no explicit modification date is available
    author: {
      '@type': 'Person',
      name: authorName,
      url: 'https://danielkliewer.com/about',
    },
    publisher: {
      '@type': 'Organization',
      name: 'Daniel Kliewer',
      logo: {
        '@type': 'ImageObject',
        url: 'https://danielkliewer.com/logo.png', // If available
        width: 112,
        height: 112,
      },
    },
    keywords: keywords,
    inLanguage: 'en-US',
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{
        __html: JSON.stringify(jsonLd),
      }}
    />
  );
}
