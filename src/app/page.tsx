import Link from "next/link";
import Image from "next/image";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Daniel Kliewer | AI Artist & Creative Technologist",
  description: "Explore Daniel Kliewer's portfolio of AI art and creative technology projects. Blending technology with artistic vision to create innovative AI-powered experiences that challenge perspectives.",
  keywords: "Daniel Kliewer, AI artist, creative technologist, generative art, large language models, LLM, AI tools, digital art, AI-powered art, Next.js, React, Python, JavaScript, portfolio, art projects, AI medium",
  authors: [{ name: "Daniel Kliewer" }],
  creator: "Daniel Kliewer",
  publisher: "Daniel Kliewer",
  openGraph: {
    title: "Daniel Kliewer | AI Artist & Creative Technologist",
    description: "Explore Daniel Kliewer's portfolio of AI art and creative technology projects. Blending technology with artistic vision to create innovative AI-powered experiences that challenge perspectives.",
    url: "https://danielkliewer.com",
    siteName: "Daniel Kliewer | AI Artist & Creative Technologist",
    images: [
      {
        url: "/self.jpg", 
        width: 1200,
        height: 630,
        alt: "Daniel Kliewer - AI Artist & Creative Technologist",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Daniel Kliewer | AI Artist & Creative Technologist",
    description: "Explore Daniel Kliewer's portfolio of AI art and creative technology projects. Blending technology with artistic vision to create innovative AI-powered experiences.",
    images: ["/self.jpg"],
  },
  alternates: {
    canonical: "https://danielkliewer.com",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: "google-site-verification-code", // Replace with actual verification code if available
  },
};

// Function to get blog posts
async function getBlogPosts(count = 3, skip = 0) {
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filenames = fs.readdirSync(postsDirectory);
  
  const posts = filenames
    .filter(filename => filename.endsWith(".md") && !filename.startsWith("_template"))
    .map(filename => {
      const filePath = path.join(postsDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data, content } = matter(fileContents);
      
      // Extract excerpt from content if not provided in frontmatter
      let excerpt = data.excerpt;
      if (!excerpt) {
        // Simple excerpt extraction - first paragraph that's not empty and not a heading
        excerpt = content
          .split('\n\n')
          .find(p => p.trim() && !p.startsWith('#'))
          ?.replace(/[#*`_]/g, '')
          .substring(0, 160);
      }
      
      return {
        slug: filename.replace(/\.md$/, ""),
        title: data.title,
        date: data.date,
        image: data.image,
        excerpt: excerpt,
        categories: data.categories || [],
      };
    })
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  
  if (skip) {
    return posts.slice(skip, skip + count);
  }
  return posts.slice(0, count);
}

// Featured projects data
const projects = [
  {
    title: "ReasonAI",
    description: "A local-first reasoning agent framework that preserves privacy by running entirely on your machine.",
    technologies: ["Next.js", "Ollama", "Llama", "TypeScript"],
    link: "/blog/2025-03-09-Reason-AI"
  },
  {
    title: "AI Filename Generator",
    description: "A Chrome extension that uses AI to automatically generate meaningful filenames when downloading files.",
    technologies: ["Chrome Extension", "JavaScript", "OpenAI API", "Webpack"],
    link: "/blog/2025-02-25-Building-an-AI-Powered-Filename-generator-chrome-extension"
  },
  {
    title: "Insight Journal",
    description: "An AI-integrated journaling platform using locally hosted LLMs for personal feedback and reflection.",
    technologies: ["Jekyll", "Llama 3.2", "Ollama", "Netlify"],
    link: "/blog/2024-10-04-Detailed-Description-of-Insight-Journal"
  },
  {
    title: "PersonaGen",
    description: "A tool for generating AI-powered personas for more effective content creation.",
    technologies: ["Python", "LangChain", "React", "OpenAI"],
    link: "/blog/2024-12-05-PersonaGen"
  },
  {
    title: "Next.js Ollama Custom Agent Framework",
    description: "A framework for building custom AI agents using Next.js and locally hosted models.",
    technologies: ["Next.js", "Ollama", "TypeScript", "React"],
    link: "/blog/2025-03-09-NextJS-Ollama-Custom-Agent-Framework"
  }
];

import ContentSlider from '@/components/ContentSlider';

export default async function Home() {
  // Get recent posts for the grid (first 6 posts)
  const recentPosts = await getBlogPosts(6);
  
  // Convert projects to a format compatible with ContentSlider
  const projectsForSlider = projects.map(project => ({
    slug: project.link.replace(/^\/blog\//, ""),
    title: project.title,
    excerpt: project.description,
    categories: project.technologies,
    // Add required date field for PostMetadata type compatibility
    date: new Date().toISOString(),
    // Use undefined instead of null for image to match the type definition
    image: undefined
  }));
  
  // Extract all technologies from posts and projects to display in the showcase
  const technologyMap = new Map<string, number>();
  
  // Add technologies from projects
  projects.forEach(project => {
    project.technologies.forEach(tech => {
      technologyMap.set(tech, (technologyMap.get(tech) || 0) + 1);
    });
  });
  
  // Add technologies/categories from posts
  recentPosts.forEach(post => {
    if (post.categories && post.categories.length > 0) {
      post.categories.forEach((category: string) => {
        technologyMap.set(category, (technologyMap.get(category) || 0) + 1);
      });
    }
  });
  
  // Convert the map to an array of technology objects
  const technologies = Array.from(technologyMap.entries())
    .map(([name, count]) => ({ name, count }))
    .filter(tech => tech.count > 0);
  
  return (
    <div className="container mx-auto px-4 py-12">
      {/* Hero Section - Moved to the top */}
      <section className="flex flex-col items-center text-center mb-16">
        <div className="max-w-3xl">
          <div className="mb-8 relative w-48 h-48 mx-auto rounded-full overflow-hidden border-4 border-primary/20">
            <Image 
              src="/self.jpg" 
              alt="Daniel Kliewer - AI Artist & Creative Technologist"
              fill
              style={{ objectFit: 'cover' }}
              priority
            />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Daniel Kliewer
          </h1>
          <h2 className="text-2xl md:text-3xl text-primary mb-6">
            AI Artist & Creative Technologist
          </h2>
          <div className="bg-secondary/20 p-6 rounded-lg mb-8 border border-secondary/30">
            <p className="text-lg italic mb-2">👋 Welcome to my creative studio!</p>
            <p className="text-md">
              Thanks for stopping by. I explore the intersection of art and artificial intelligence, 
              using AI as both medium and collaborator. My work challenges traditional notions 
              of creativity while exploring the evolving relationship between humans and machines.
            </p>
          </div>
          <p className="text-lg text-muted-foreground mb-8">
            I create art and experiences that leverage the power of large language models and generative 
            AI. Through my work, I investigate themes of digital identity, algorithmic creativity, and 
            the blurred boundaries between human and machine artistic expression. My approach balances 
            technological innovation with artistic vision, creating works that are both conceptually rich 
            and technically sophisticated.
          </p>
          <div className="flex gap-4 flex-wrap justify-center">
            <Link 
              href="/blog" 
              className="btn btn-primary">
              Read My Blog
            </Link>
            <Link 
              href="/contact" 
              className="btn btn-outline">
              Get In Touch
            </Link>
          </div>
        </div>
      </section>
     {/* Featured Projects Section */}      {/* Featured Projects Slider - Using the previous slider component */}
      <ContentSlider posts={projectsForSlider} />

      {/* Recent Blog Posts Section */}
      <section className="mb-20">
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-2xl md:text-3xl font-bold">Recent Blog Posts</h2>
          <Link href="/blog" className="text-primary hover:underline">
            View All →
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {recentPosts.map((post) => (
            <Link 
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="card p-6 transition-all hover:shadow-md"
            >
              <h3 className="text-xl font-bold mb-3 line-clamp-2">{post.title}</h3>
              
              {/* Display technologies/categories as tags */}
              {post.categories && post.categories.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {post.categories.map((category: string, idx: number) => (
                    <span 
                      key={idx} 
                      className="inline-block text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                    >
                      {category}
                    </span>
                  ))}
                </div>
              )}
              
              <p className="text-sm text-muted-foreground mb-3">
                {new Date(post.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </p>
              
              {post.excerpt && (
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {post.excerpt}
                </p>
              )}
            </Link>
          ))}
        </div>
      </section>


    </div>
  );
}
