import fs from 'fs';
import path from 'path';
import { MetadataRoute } from 'next';

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://danielkliewer.com';
  
  // Static routes with their last modified date
  const staticRoutes = [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: 'weekly' as const,
      priority: 1.0,
    },
    {
      url: `${baseUrl}/about`,
      lastModified: new Date(),
      changeFrequency: 'monthly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/blog`,
      lastModified: new Date(),
      changeFrequency: 'weekly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/contact`,
      lastModified: new Date(),
      changeFrequency: 'monthly' as const,
      priority: 0.7,
    },
  ];

  // Get blog posts for dynamic routes
  const postsDirectory = path.join(process.cwd(), '_posts');
  const filenames = fs.readdirSync(postsDirectory);
  
  const blogPosts = filenames
    .filter(filename => filename.endsWith('.md') && !filename.startsWith('_template'))
    .map(filename => {
      const slug = filename.replace(/\.md$/, '');
      const filePath = path.join(postsDirectory, filename);
      const fileStat = fs.statSync(filePath);
      
      return {
        url: `${baseUrl}/blog/${slug}`,
        lastModified: fileStat.mtime,
        changeFrequency: 'monthly' as const,
        priority: 0.7,
      };
    });

  return [...staticRoutes, ...blogPosts];
}
