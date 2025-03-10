'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { PostMetadata } from '@/types/blog';

interface ContentSliderProps {
  posts: PostMetadata[];
}

const ContentSlider: React.FC<ContentSliderProps> = ({ posts }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  // Handle client-side mounting to prevent hydration errors
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Auto-advance the slider
  useEffect(() => {
    if (!isMounted || posts.length <= 1) return;
    
    const timer = setInterval(() => {
      goToNext();
    }, 5000); // Advance every 5 seconds
    
    return () => clearInterval(timer);
  }, [currentIndex, posts.length, isMounted]);

  const goToSlide = useCallback((index: number) => {
    if (isTransitioning || !isMounted) return;
    setIsTransitioning(true);
    setCurrentIndex(index);
    setTimeout(() => setIsTransitioning(false), 500); // Match transition duration
  }, [isTransitioning, isMounted]);

  const goToNext = useCallback(() => {
    const newIndex = (currentIndex + 1) % posts.length;
    goToSlide(newIndex);
  }, [currentIndex, posts.length, goToSlide]);

  const goToPrevious = useCallback(() => {
    const newIndex = (currentIndex - 1 + posts.length) % posts.length;
    goToSlide(newIndex);
  }, [currentIndex, posts.length, goToSlide]);

  if (!posts || posts.length === 0) return null;

  return (
    <section className="relative overflow-hidden mb-16 rounded-xl shadow-lg">
      <div
        ref={sliderRef}
        className="relative w-full h-[400px] md:h-[500px] overflow-hidden"
      >
        <div 
          className="flex h-full transition-transform duration-500 ease-in-out"
          style={isMounted ? { transform: `translateX(-${currentIndex * 100}%)` } : {}}
        >
          {posts.map((post, index) => (
            <div key={post.slug} className="min-w-full relative h-full flex-shrink-0">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/70 z-10" />
              
              {post.image ? (
                <Image 
                  src={post.image} 
                  alt={post.title || 'Featured post'} 
                  fill
                  className="object-cover"
                  priority={index === 0}
                />
              ) : (
                <div className="w-full h-full bg-gradient-to-r from-primary/30 to-primary/10 flex items-center justify-center">
                  <span className="text-white text-2xl font-medium">
                    {post.title}
                  </span>
                </div>
              )}
              
              <div className="absolute bottom-0 left-0 right-0 p-6 md:p-8 text-white z-20">
                <div className="flex flex-wrap gap-2 mb-3">
                  {post.categories && post.categories.map((category, idx) => (
                    <span 
                      key={idx} 
                      className="inline-block text-xs px-2 py-1 bg-primary text-white rounded-full"
                    >
                      {category}
                    </span>
                  ))}
                </div>
                <h2 className="text-2xl md:text-3xl font-bold mb-2">{post.title}</h2>
                <p className="mb-4 line-clamp-2 text-gray-200">
                  {post.excerpt || 'Click to read this featured article.'}
                </p>
                <Link 
                  href={`/blog/${post.slug}`}
                  className="inline-block px-5 py-2 bg-primary text-white rounded-md font-medium hover:bg-primary/90 transition-colors"
                >
                  Read More
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Navigation arrows */}
      <button 
        onClick={goToPrevious}
        className="absolute left-4 top-1/2 -translate-y-1/2 bg-black/30 hover:bg-black/50 text-white rounded-full p-2 z-30"
        aria-label="Previous slide"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      
      <button 
        onClick={goToNext}
        className="absolute right-4 top-1/2 -translate-y-1/2 bg-black/30 hover:bg-black/50 text-white rounded-full p-2 z-30"
        aria-label="Next slide"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
      
      {/* Indicators */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex space-x-2 z-30">
        {posts.map((_, index) => (
          <button
            key={index}
            onClick={() => goToSlide(index)}
            className={`w-3 h-3 rounded-full ${
              currentIndex === index ? 'bg-white' : 'bg-white/50'
            }`}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </section>
  );
};

export default ContentSlider;
