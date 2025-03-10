'use client';

import React from 'react';
import BlogCard from './BlogCard';
import { PostMetadata } from '@/types/blog';

interface FeaturedPostsProps {
  posts: PostMetadata[];
}

const FeaturedPosts: React.FC<FeaturedPostsProps> = ({ posts }) => {
  if (!posts || posts.length === 0) return null;

  return (
    <section className="mb-16">
      <h2 className="text-2xl md:text-3xl font-bold mb-8">Featured Posts</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map((post) => (
          <BlogCard key={post.slug} post={post} />
        ))}
      </div>
    </section>
  );
};

export default FeaturedPosts;
