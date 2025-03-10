'use client';

import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Add copy buttons to code blocks after render
  useEffect(() => {
    if (!containerRef.current) return;
    
    const codeBlocks = containerRef.current.querySelectorAll('pre > code');
    const currentContainerRef = containerRef.current; // Store ref value to use in cleanup
    
    codeBlocks.forEach((codeBlock) => {
      const pre = codeBlock.parentElement;
      if (!pre || pre.querySelector('.copy-button')) return;
      
      // Create wrapper div for positioning
      const wrapperDiv = document.createElement('div');
      wrapperDiv.className = 'relative my-6';
      pre.parentNode?.insertBefore(wrapperDiv, pre);
      wrapperDiv.appendChild(pre);
      
      pre.className = 'p-4 rounded-md bg-gray-800 text-gray-200 overflow-x-auto';
      
      const code = codeBlock.textContent || '';
      
      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button fixed z-10 p-2 rounded-md bg-gray-700 text-gray-200 hover:bg-gray-600 transition-all duration-200';
      copyButton.ariaLabel = 'Copy code to clipboard';
      copyButton.style.position = 'absolute';
      copyButton.style.top = '0.5rem';
      copyButton.style.right = '0.5rem';
      copyButton.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      `;
      
      wrapperDiv.appendChild(copyButton);
      
      // Add copy functionality
      copyButton.addEventListener('click', async () => {
        await navigator.clipboard.writeText(code);
        
        // Show success state
        copyButton.classList.remove('bg-gray-700');
        copyButton.classList.add('bg-green-600', 'text-white');
        copyButton.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        `;
        
        // Reset after 2 seconds
        setTimeout(() => {
          copyButton.classList.remove('bg-green-600', 'text-white');
          copyButton.classList.add('bg-gray-700');
          copyButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          `;
        }, 2000);
      });
      
      // Add scroll event listener to show/hide button
      const updateButtonVisibility = () => {
        const rect = wrapperDiv.getBoundingClientRect();
        const isVisible = 
          rect.top < window.innerHeight &&
          rect.bottom > 0;
        
        copyButton.style.opacity = isVisible ? '1' : '0';
        copyButton.style.pointerEvents = isVisible ? 'auto' : 'none';
      };
      
      window.addEventListener('scroll', updateButtonVisibility);
      updateButtonVisibility(); // Initial check
      
      // Store the event listener reference for cleanup
      wrapperDiv.dataset.scrollListener = 'true';
    });
    
    // Cleanup function
    return () => {
      const wrappers = currentContainerRef.querySelectorAll('div[data-scroll-listener="true"]');
      wrappers.forEach((wrapper) => {
        window.removeEventListener('scroll', () => {});
      });
    };
  }, [content]);

  return (
    <div ref={containerRef} className={`prose prose-lg dark:prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[
          rehypeRaw,
          rehypeSlug,
          [rehypeAutolinkHeadings, { behavior: 'wrap' }],
          [rehypeHighlight, { ignoreMissing: true }]
        ]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
