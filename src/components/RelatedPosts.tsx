'use client';

import React from 'react';
import Link from 'next/link';
import { RelatedPost } from '@/types/blog';

interface RelatedPostsProps {
  posts: RelatedPost[];
  title?: string;
}

const RelatedPosts: React.FC<RelatedPostsProps> = ({ 
  posts, 
  title = "Related Posts" 
}) => {
  if (!posts.length) return null;

  return (
    <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-800">
      <h2 className="text-2xl font-bold mb-6">{title}</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map((post) => (
          <Link 
            key={post.slug}
            href={`/blog/${post.slug}`}
            className="group p-4 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-all"
          >
            <h3 className="font-bold mb-2 group-hover:text-primary transition-colors line-clamp-2">
              {post.title}
            </h3>
            {post.excerpt && (
              <p className="text-muted-foreground text-sm mb-2 line-clamp-2">
                {post.excerpt}
              </p>
            )}
            <span className="text-xs text-muted-foreground">
              {new Date(post.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                timeZone: 'UTC'
              })}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default RelatedPosts;
