import { MetadataRoute } from 'next';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: ['/api/'], // Disallow API routes if any
    },
    sitemap: 'https://danielkliewer.com/sitemap.xml',
  };
}
