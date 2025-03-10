import fs from "fs";
import path from "path";
import matter from "gray-matter";
import Link from "next/link";
import { Metadata } from "next";

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

export default function BlogPage() {
  // Get all posts
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filenames = fs.readdirSync(postsDirectory);
  
  const posts = filenames
    .filter(filename => filename.endsWith(".md") && !filename.startsWith("_template"))
    .map(filename => {
      const filePath = path.join(postsDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data } = matter(fileContents);
      
      return {
        slug: filename.replace(/\.md$/, ""),
        title: data.title || "Untitled",
        // Use a fixed date if none provided to avoid hydration mismatches
        date: data.date || '2025-01-01T00:00:00Z',
      };
    })
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Blog</h1>
      <p className="text-lg text-muted-foreground mb-12 max-w-2xl">
        Thoughts, insights, and tutorials on AI development, 
        software engineering, and personal projects.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map((post) => (
          <Link 
            key={post.slug}
            href={`/blog/${post.slug}`}
            className="card p-6 transition-all hover:shadow-md"
          >
            <h2 className="text-xl font-bold mb-3 line-clamp-2">{post.title}</h2>
            <p className="text-sm text-muted-foreground">
              {/* Use Intl.DateTimeFormat with fixed settings to prevent hydration mismatches */}
              {new Intl.DateTimeFormat('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                timeZone: 'UTC' // Ensure consistent timezone between server and client
              }).format(new Date(post.date))}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
