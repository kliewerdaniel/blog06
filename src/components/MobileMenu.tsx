'use client';

import Link from 'next/link';

export default function MobileMenu() {
  return (
    <div className="md:hidden z-50 relative">
      <nav className="horizontal-mobile-nav">
        <ul className="horizontal-mobile-list">
          <li className="horizontal-mobile-item">
            <Link href="/" className="horizontal-mobile-link">
              Home
            </Link>
          </li>
          <li className="horizontal-mobile-item">
            <Link href="/about" className="horizontal-mobile-link">
              About
            </Link>
          </li>
          <li className="horizontal-mobile-item">
            <Link href="/blog" className="horizontal-mobile-link">
              Blog
            </Link>
          </li>
          <li className="horizontal-mobile-item">
            <Link href="/art" className="horizontal-mobile-link">
              Art
            </Link>
          </li>
        </ul>
      </nav>
    </div>
  );
}
