import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Middleware to handle page transitions for Google Analytics 4
 * This runs on the edge and works with the client-side GA functions
 */
export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  // Add a custom header to detect navigation in client components
  // This helps us properly track page views in the SPA context
  response.headers.set('x-middleware-cache', 'no-cache');
  
  return response;
}

// Specify paths that this middleware applies to
// Apply to all paths except for static files, images, api routes, etc.
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder files (robots.txt, sitemap.xml, etc.)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|robots.txt|sitemap.xml).*)',
  ],
};
