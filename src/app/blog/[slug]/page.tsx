import fs from "fs";
import path from "path";
import matter from "gray-matter";
import Link from "next/link";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { BlogJsonLd } from "@/components/BlogJsonLd";
import { Metadata, ResolvingMetadata } from "next/types";
import BreadcrumbNavigation from "@/components/BreadcrumbNavigation";
import RelatedPosts from "@/components/RelatedPosts";
import SocialShare from "@/components/SocialShare";
import { RelatedPost } from "@/types/blog";
import PageTransition from "@/components/PageTransition";

type Props = {
  params: { slug: string };
  searchParams: { [key: string]: string | string[] | undefined };
};

export async function generateMetadata(
  { params }: Props,
  parent: ResolvingMetadata
): Promise<Metadata> {
  // Read the post file
  const { slug } = params;
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filePath = path.join(postsDirectory, `${slug}.md`);
  
  try {
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);
    
    // Get the first paragraph for description (up to 160 characters)
    const firstParagraph = content
      .split('\n\n')
      .find(p => p.trim() && !p.startsWith('#'))
      ?.replace(/[#*`_]/g, '')
      .trim();
      
    const description = firstParagraph ? 
      (firstParagraph.length > 160 ? `${firstParagraph.substring(0, 157)}...` : firstParagraph) : 
      "Read this insightful article by Daniel Kliewer on AI, software engineering, and technology.";
    
    // Extract tags or categories for keywords
    const tags = data.tags || [];
    const categories = data.categories || [];
    const keywords = [...tags, ...categories].join(", ");
    
    // Create canonical URL
    const canonical = `https://danielkliewer.com/blog/${slug}`;
    
    // Get dates for published/modified
    const publishDate = data.date ? new Date(data.date).toISOString() : undefined;
    
    // Get parent metadata (e.g., default OG image)
    const previousImages = (await parent).openGraph?.images || [];
    
    return {
      title: `${data.title || "Article"} | Daniel Kliewer`,
      description: description,
      keywords: keywords.length > 0 ? keywords : "AI, development, software engineering, LLMs, technology",
      authors: [{ name: "Daniel Kliewer" }],
      openGraph: {
        title: data.title,
        description: description,
        url: canonical,
        siteName: "Daniel Kliewer | Software Engineer & AI Developer",
        images: [...previousImages],
        type: "article",
        publishedTime: publishDate,
      },
      twitter: {
        card: "summary_large_image",
        title: data.title,
        description: description,
      },
      alternates: {
        canonical: canonical,
      },
    };
  } catch {
    // Removed unused 'error' variable
    return {
      title: "Post Not Found | Daniel Kliewer",
      description: "The blog post you're looking for does not exist.",
    };
  }
}

export async function generateStaticParams() {
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filenames = fs.readdirSync(postsDirectory);
  
  return filenames
    .filter(filename => filename.endsWith(".md") && !filename.startsWith("_template"))
    .map(filename => ({
      slug: filename.replace(/\.md$/, ""),
    }));
}

// Function to find related posts based on categories and tags
function findRelatedPosts(currentSlug: string, currentCategories: string[] = [], currentTags: string[] = []): RelatedPost[] {
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filenames = fs.readdirSync(postsDirectory);
  
  // Score map to rank related posts
  const relatedScores: Map<string, { score: number; post: RelatedPost }> = new Map();
  
  filenames
    .filter(filename => filename.endsWith(".md") && !filename.startsWith("_template") && !filename.includes(currentSlug))
    .forEach(filename => {
      const filePath = path.join(postsDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data, content } = matter(fileContents);
      
      const slug = filename.replace(/\.md$/, "");
      const postCategories = data.categories || [];
      const postTags = data.tags || [];
      
      // Calculate relevance score - more shared categories/tags = higher score
      let score = 0;
      
      // Category matches (weighted more heavily)
      postCategories.forEach((category: string) => {
        if (currentCategories.includes(category)) {
          score += 2;
        }
      });
      
      // Tag matches
      postTags.forEach((tag: string) => {
        if (currentTags.includes(tag)) {
          score += 1;
        }
      });
      
      // Only consider posts with a relevance score
      if (score > 0) {
        // Get excerpt
        const excerpt = content
          .replace(/[#*`_]/g, '')
          .split('\n\n')
          .find(p => p.trim() && !p.includes('#')) || '';
          
        const shortExcerpt = excerpt.length > 120 ? 
          `${excerpt.substring(0, 117)}...` : excerpt;
        
        relatedScores.set(slug, { 
          score,
          post: {
            slug,
            title: data.title || "Untitled",
            date: data.date ? new Date(data.date).toISOString() : new Date().toISOString(),
            excerpt: shortExcerpt,
            categories: postCategories,
          }
        });
      }
    });
  
  // Convert to array, sort by score, and take top 3
  return Array.from(relatedScores.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(item => item.post);
}

// Use a more specific type for static exports
type BlogPostProps = {
  params: {
    slug: string;
  };
}

import { Suspense } from "react";

export default function BlogPost({ params }: BlogPostProps) {
  const { slug } = params;
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filePath = path.join(postsDirectory, `${slug}.md`);
  
  try {
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);
    
    const title = data.title || "Untitled";
    // Use a fixed date if none provided to avoid hydration mismatches
    const date = data.date ? new Date(data.date) : new Date('2025-01-01T00:00:00Z');
    const categories = data.categories || [];
    const tags = data.tags || [];
    
    // Find related posts
    const relatedPosts = findRelatedPosts(slug, categories, tags);
    
    // Calculate estimated reading time
    const wordsPerMinute = 200;
    const wordCount = content.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / wordsPerMinute);
    
    // Create breadcrumb items
    const breadcrumbItems = [
      { label: 'Home', href: '/' },
      { label: 'Blog', href: '/blog' },
      { label: title, href: `/blog/${slug}`, isCurrent: true }
    ];
    
    return (
    <PageTransition transitionType="fade">
      <div className="container mx-auto px-4 py-12">
        <BlogJsonLd 
          title={title}
          description={data.description || `${title} - Article by Daniel Kliewer`}
          datePublished={date.toISOString()}
          authorName="Daniel Kliewer"
          slug={slug}
          tags={tags}
          categories={categories}
        />
        
        {/* Breadcrumb Navigation */}
        <BreadcrumbNavigation items={breadcrumbItems} />
        
        <article className="max-w-3xl mx-auto">
          <header className="mb-8">
            {categories.length > 0 && (
              <div className="mb-4 flex flex-wrap gap-2">
                {categories.map((category: string, idx: number) => (
                  <Link 
                    key={idx}
                    href={`/blog?category=${encodeURIComponent(category)}`}
                    className="inline-block text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                  >
                    {category}
                  </Link>
                ))}
              </div>
            )}
            
            <h1 className="text-4xl font-bold mb-4">{title}</h1>
            
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <time dateTime={date.toISOString()}>
                {new Intl.DateTimeFormat('en-US', {
                  weekday: 'long',
                  year: 'numeric',
                  month: 'long', 
                  day: 'numeric',
                  timeZone: 'UTC' // Ensure consistent timezone between server and client
                }).format(date)}
              </time>
              
              <span className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 mr-1">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                {readingTime} min read
              </span>
            </div>
            
            {/* Social Sharing Buttons */}
            <SocialShare 
              title={title}
              summary={`Check out this article: ${title}`}
              className="mt-4" 
            />
          </header>
          
          <div className="mb-12">
            <MarkdownRenderer content={content} />
          </div>
          
          {tags.length > 0 && (
            <div className="mb-8 pt-6 border-t border-gray-200 dark:border-gray-800">
              <h2 className="text-lg font-semibold mb-3">Tags</h2>
              <div className="flex flex-wrap gap-2">
                {tags.map((tag: string, idx: number) => (
                  <Suspense key={idx} fallback={<span className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-full text-sm">{tag}</span>}>
                    <Link
                      key={idx}
                      href={`/blog?tag=${encodeURIComponent(tag)}`}
                      className="px-3 py-1 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full text-sm"
                    >
                      {tag}
                    </Link>
                  </Suspense>
                ))}
              </div>
            </div>
          )}
          
          {/* Related Posts */}
          {relatedPosts.length > 0 && (
            <RelatedPosts posts={relatedPosts} />
          )}
          
          {/* Social Sharing Buttons (Bottom) */}
          <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-800">
            <SocialShare 
              title={title}
              summary={`Check out this article: ${title}`}
              className="justify-center" 
            />
          </div>
          
          {/* Back to Blog link */}
          <div className="mt-12 text-center">
            <Link href="/blog" className="text-primary hover:underline">
              ← Back to Blog
            </Link>
          </div>
        </article>
      </div>
    </PageTransition>
    );
  } catch {
    // Removed unused 'error' variable
    return (
    <PageTransition transitionType="fade">
      <div className="container mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold mb-8">Post Not Found</h1>
        <p>Sorry, the blog post you&apos;re looking for does not exist.</p>
        <Link href="/blog" className="text-primary hover:underline mt-4 inline-block">
          ← Back to Blog
        </Link>
      </div>
    </PageTransition>
    );
  }
}

// Note: The SearchParamsProvider is imported but not used in this component.
// This is intentional as we're primarily using static data here.
// If you need to use search parameters, make sure to wrap them in a Suspense boundary.
