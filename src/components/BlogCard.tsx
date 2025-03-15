'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { PostMetadata } from '@/types/blog';

// Function to calculate reading time based on word count
const calculateReadingTime = (content: string): string => {
  const wordsPerMinute = 200;
  const wordCount = content.trim().split(/\s+/).length;
  const readingTime = Math.ceil(wordCount / wordsPerMinute);
  return `${readingTime} min read`;
};

interface BlogCardProps {
  post: PostMetadata;
  content?: string;
}

const BlogCard: React.FC<BlogCardProps> = ({ post, content = '' }) => {
  const readingTime = content ? calculateReadingTime(content) : '3 min read';
  
  // Default image if none provided - using a simple text placeholder if image is missing
  const imageSrc = post.image || '';
  const hasImage = !!imageSrc;
  const formattedDate = new Date(post.date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC' // Ensure consistent timezone between server and client
  });

  const excerpt = post.excerpt || (content ? 
    content.replace(/[#*`_]/g, '').split('\n\n').find(p => p.trim() && !p.includes('#')) || '' : 
    'Click to read this insightful article by Daniel Kliewer.');
  
  return (
    <Link 
      href={`/blog/${post.slug}`}
      className="card overflow-hidden bg-white dark:bg-gray-800 rounded-lg shadow-sm hover:shadow-md transition-shadow duration-300"
    >
      <div className="aspect-w-16 aspect-h-9 relative">
        {hasImage ? (
          <Image 
            src={imageSrc}
            alt={post.title || 'Blog post image'}
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
        ) : (
          <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
            <span className="text-gray-500 dark:text-gray-400 text-sm font-medium">
              {post.title?.substring(0, 30) || 'Featured Image'}
            </span>
          </div>
        )}
      </div>
      <div className="p-5">
        {post.categories && post.categories.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {post.categories.map((category: string, idx: number) => (
              <span 
                key={idx} 
                className="inline-block text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
              >
                {category}
              </span>
            ))}
          </div>
        )}
        <h3 className="text-xl font-bold mb-2 line-clamp-2">{post.title}</h3>
        <p className="text-muted-foreground mb-3 line-clamp-2 text-sm">{excerpt}</p>
        <div className="flex justify-between items-center text-xs text-muted-foreground">
          <time dateTime={new Date(post.date).toISOString()}>{formattedDate}</time>
          <span>{readingTime}</span>
        </div>
      </div>
    </Link>
  );
};

export default BlogCard;
