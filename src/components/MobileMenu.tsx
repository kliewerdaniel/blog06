'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function MobileMenu() {
  const [isOpen, setIsOpen] = useState(false);

  // Close the menu when user navigates to a new page
  useEffect(() => {
    const handleRouteChange = () => {
      setIsOpen(false);
    };

    // Add listener for route changes
    window.addEventListener('popstate', handleRouteChange);

    // Clean up event listener
    return () => {
      window.removeEventListener('popstate', handleRouteChange);
    };
  }, []);

  // Prevent scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <div className="md:hidden z-50 relative">
      <button 
        onClick={() => setIsOpen(!isOpen)} 
        className="mobile-menu-toggle p-2 focus:outline-none"
        aria-expanded={isOpen}
        aria-label="Toggle navigation menu"
      >
        <div className={`hamburger ${isOpen ? 'is-active' : ''}`}>
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
        </div>
      </button>

      {/* Full screen mobile menu */}
      <div 
        className={`mobile-menu-container ${isOpen ? 'mobile-menu-open' : ''}`}
        aria-hidden={!isOpen}
      >
        <div className="mobile-menu-header">
          <button 
            onClick={() => setIsOpen(false)} 
            className="mobile-menu-close p-2"
            aria-label="Close menu"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        
        <nav className="mobile-menu-nav">
          <ul className="mobile-menu-list">
            <li className="mobile-menu-item">
              <Link href="/" onClick={() => setIsOpen(false)} className="mobile-menu-link">
                Home
              </Link>
            </li>
            <li className="mobile-menu-item">
              <Link href="/about" onClick={() => setIsOpen(false)} className="mobile-menu-link">
                About
              </Link>
            </li>
            <li className="mobile-menu-item">
              <Link href="/blog" onClick={() => setIsOpen(false)} className="mobile-menu-link">
                Blog
              </Link>
            </li>
            <li className="mobile-menu-item">
              <Link href="/art" onClick={() => setIsOpen(false)} className="mobile-menu-link">
                Art
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
}
