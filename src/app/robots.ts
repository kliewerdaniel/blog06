import { MetadataRoute } from 'next';

// Force this route to be static for export
export const dynamic = "force-static";

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
