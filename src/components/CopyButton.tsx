'use client';

import React, { useState, useEffect, useRef } from 'react';

interface CopyButtonProps {
  code: string;
  containerRef: React.RefObject<HTMLDivElement | null>;
  codeBlockRef: React.RefObject<HTMLPreElement | null>;
}

export const CopyButton: React.FC<CopyButtonProps> = ({ 
  code, 
  containerRef,
  codeBlockRef 
}) => {
  const [copied, setCopied] = useState(false);
  const [visible, setVisible] = useState(true);
  const buttonRef = useRef<HTMLButtonElement>(null);
  
  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    
    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  useEffect(() => {
    if (!containerRef.current || !codeBlockRef.current || !buttonRef.current) return;

    const container = containerRef.current;
    const codeBlock = codeBlockRef.current;
    const button = buttonRef.current;
    
    const updateButtonPosition = () => {
      if (!container || !codeBlock || !button) return;
      
      const containerRect = container.getBoundingClientRect();
      const codeRect = codeBlock.getBoundingClientRect();
      
      // Check if code block is in view
      const isCodeVisible = 
        codeRect.top < window.innerHeight &&
        codeRect.bottom > 0;
      
      setVisible(isCodeVisible);
      
      if (isCodeVisible) {
        // Position the button at the top-right of the code block, but don't let it go outside container
        const topPosition = Math.max(
          Math.min(codeRect.top - containerRect.top, containerRect.height - button.offsetHeight),
          0
        );
        
        button.style.top = `${topPosition}px`;
        button.style.right = '1rem';
      }
    };
    
    // Update on scroll and resize
    window.addEventListener('scroll', updateButtonPosition);
    window.addEventListener('resize', updateButtonPosition);
    
    // Initial position
    updateButtonPosition();
    
    return () => {
      window.removeEventListener('scroll', updateButtonPosition);
      window.removeEventListener('resize', updateButtonPosition);
    };
  }, [containerRef, codeBlockRef]);

  return (
    <button
      ref={buttonRef}
      onClick={copyToClipboard}
      className={`
        fixed z-10 p-2 rounded-md transition-all duration-200 
        ${visible ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        ${copied ? 'bg-green-600 text-white success-pulse' : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}
      `}
      aria-label="Copy code to clipboard"
    >
      {copied ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      )}
    </button>
  );
};
