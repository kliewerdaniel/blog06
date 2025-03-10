import React from 'react';

interface BlogJsonLdProps {
  title: string;
  description: string;
  datePublished: string;
  authorName: string;
  slug: string;
  tags?: string[];
  categories?: string[];
  imageUrl?: string;
}

export const BlogJsonLd: React.FC<BlogJsonLdProps> = ({
  title,
  description,
  datePublished,
  authorName,
  slug,
  tags = [],
  categories = [],
  imageUrl
}) => {
  const canonicalUrl = `https://danielkliewer.com/blog/${slug}`;
  
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: title,
    description: description,
    image: imageUrl ? [imageUrl] : undefined,
    datePublished: datePublished,
    author: {
      '@type': 'Person',
      name: authorName,
      url: 'https://danielkliewer.com/about'
    },
    publisher: {
      '@type': 'Person',
      name: authorName,
      url: 'https://danielkliewer.com'
    },
    url: canonicalUrl,
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': canonicalUrl
    },
    keywords: [...tags, ...categories].join(', ')
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
};
