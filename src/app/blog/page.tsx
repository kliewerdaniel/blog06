import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { Metadata } from "next";
import Link from "next/link";
import BlogCard from "@/components/BlogCard";
import CategoryFilter from "@/components/CategoryFilter";
import InfiniteScroll from "@/components/InfiniteScroll";
import { PostMetadata, TaxonomyItem } from "@/types/blog";
import FeaturedPosts from "@/components/FeaturedPosts";

// Separate viewport export as recommended by Next.js
export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: "Blog | Daniel Kliewer - Software Engineer & AI Developer",
  description: "Explore articles on AI development, software engineering, and technology projects. Daniel Kliewer shares insights, tutorials, and experiences in machine learning and web development.",
  keywords: "blog, AI development, software engineering, machine learning, web development, technology, programming tutorials",
  authors: [{ name: "Daniel Kliewer" }],
  openGraph: {
    title: "Daniel Kliewer's Blog - AI & Software Engineering Insights",
    description: "Explore articles on AI development, software engineering, and technology projects. Daniel Kliewer shares insights, tutorials, and experiences in machine learning and web development.",
    url: "https://danielkliewer.com/blog",
    siteName: "Daniel Kliewer | Software Engineer & AI Developer",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Daniel Kliewer's Blog - AI & Software Engineering Insights",
    description: "Explore articles on AI development, software engineering, and technology projects. Daniel Kliewer shares insights, tutorials, and experiences in machine learning and web development.",
  },
  alternates: {
    canonical: "https://danielkliewer.com/blog",
  },
};

// Helper function to extract excerpt from content
function extractExcerpt(content: string, maxLength: number = 150): string {
  const excerpt = content
    .replace(/[#*`_]/g, '')
    .split('\n\n')
    .find(p => p.trim() && !p.includes('#')) || '';
  
  return excerpt.length > maxLength
    ? excerpt.substring(0, maxLength) + '...'
    : excerpt;
}

// Function to get all taxonomy items from posts
function getTaxonomyItems(posts: PostMetadata[], key: 'categories' | 'tags'): TaxonomyItem[] {
  const taxonomyMap = new Map<string, number>();
  
  posts.forEach(post => {
    const items = post[key] || [];
    items.forEach(item => {
      taxonomyMap.set(item, (taxonomyMap.get(item) || 0) + 1);
    });
  });
  
  return Array.from(taxonomyMap.entries())
    .map(([name, count]) => ({
      name,
      count,
      slug: name.toLowerCase().replace(/\s+/g, '-'),
    }))
    .sort((a, b) => b.count - a.count);
}

// Define popular tags that will be featured
const FEATURED_TAGS = ['Machine Learning', 'AI', 'Next.js', 'React', 'Software Engineering'];

// Get posts and filter by featured
function getFeaturedPosts(posts: PostMetadata[], limit: number = 3): PostMetadata[] {
  return posts
    .filter(post => {
      // Consider a post featured if it has at least one featured tag
      const postTags = post.tags || [];
      return postTags.some(tag => FEATURED_TAGS.includes(tag));
    })
    .slice(0, limit);
}

export default function BlogPage() {
  // Get all posts
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filenames = fs.readdirSync(postsDirectory);
  
  const posts = filenames
    .filter(filename => filename.endsWith(".md") && !filename.startsWith("_template"))
    .map(filename => {
      const filePath = path.join(postsDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data, content } = matter(fileContents);
      
      return {
        slug: filename.replace(/\.md$/, ""),
        title: data.title || "Untitled",
        date: data.date || '2025-01-01T00:00:00Z',
        categories: data.categories || [],
        tags: data.tags || [],
        excerpt: data.excerpt || extractExcerpt(content),
        content
      };
    })
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  // Extract categories and tags
  const categories = getTaxonomyItems(posts, 'categories');
  const featuredPosts = getFeaturedPosts(posts);

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Blog</h1>
      <p className="text-lg text-muted-foreground mb-12 max-w-2xl">
        Thoughts, insights, and tutorials on AI development, 
        software engineering, and personal projects.
      </p>
      
      {/* Featured Posts Section */}
      {featuredPosts.length > 0 && (
        <FeaturedPosts posts={featuredPosts} />
      )}
      
      {/* Taxonomy Filter */}
      {categories.length > 0 && (
        <div className="mb-10">
          <CategoryFilter 
            categories={categories} 
            selectedCategory={null}
            onSelectCategory={() => {}}
          />
        </div>
      )}
      
      {/* Blog Posts with Infinite Scroll */}
      <div className="mb-12">
        <InfiniteScroll 
          initialPosts={posts} 
          postContents={posts.reduce((acc, post) => {
            acc[post.slug] = post.content;
            return acc;
          }, {} as Record<string, string>)}
          postsPerPage={9}
        />
      </div>
    </div>
  );
}
