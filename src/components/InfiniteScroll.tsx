'use client';

import React, { useEffect, useRef, useState } from 'react';
import { PostMetadata } from '@/types/blog';
import BlogCard from './BlogCard';

interface InfiniteScrollProps {
  initialPosts: PostMetadata[];
  postContents: Record<string, string>;
  postsPerPage?: number;
}

const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  initialPosts,
  postContents,
  postsPerPage = 9,
}) => {
  const [posts, setPosts] = useState<PostMetadata[]>(initialPosts.slice(0, postsPerPage));
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(initialPosts.length > postsPerPage);
  const observer = useRef<IntersectionObserver | null>(null);
  const loadingRef = useRef<HTMLDivElement>(null);

  // Function to load more posts
  const loadMorePosts = () => {
    if (loading || !hasMore) return;
    
    setLoading(true);
    
    // Simulate API call by delaying load
    setTimeout(() => {
      const currentLength = posts.length;
      const nextPosts = initialPosts.slice(currentLength, currentLength + postsPerPage);
      
      setPosts(prevPosts => [...prevPosts, ...nextPosts]);
      setLoading(false);
      
      if (currentLength + nextPosts.length >= initialPosts.length) {
        setHasMore(false);
      }
    }, 300);
  };

  // Setup intersection observer
  useEffect(() => {
    if (loading) return;
    
    if (observer.current) {
      observer.current.disconnect();
    }
    
    observer.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMore) {
        loadMorePosts();
      }
    }, { threshold: 0.1 });
    
    if (loadingRef.current) {
      observer.current.observe(loadingRef.current);
    }
    
    return () => observer.current?.disconnect();
  }, [loading, hasMore]);

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map(post => (
          <BlogCard 
            key={post.slug} 
            post={post} 
            content={postContents[post.slug] || ''} 
          />
        ))}
      </div>
      
      {/* Loading indicator */}
      {hasMore && (
        <div ref={loadingRef} className="flex justify-center items-center py-8">
          {loading ? (
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-8 w-8 bg-primary/20 rounded-full"></div>
              <span className="mt-2 text-sm text-muted-foreground">Loading more posts...</span>
            </div>
          ) : (
            <div className="h-8 w-full"></div> // Invisible element for observer
          )}
        </div>
      )}
      
      {/* End message */}
      {!hasMore && posts.length > 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <p>You've reached the end of the posts</p>
        </div>
      )}
    </>
  );
};

export default InfiniteScroll;
