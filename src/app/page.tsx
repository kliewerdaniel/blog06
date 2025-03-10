import Link from "next/link";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Daniel Kliewer | Software Engineer & AI Developer Portfolio",
  description: "Explore Daniel Kliewer's portfolio of AI development and software engineering projects. Specializing in LLMs, Next.js, React, and privacy-focused AI solutions that prioritize user control.",
  keywords: "Daniel Kliewer, software engineer, AI developer, large language models, LLM, Next.js, React, Python, JavaScript, TypeScript, portfolio, AI projects, privacy-focused AI",
  authors: [{ name: "Daniel Kliewer" }],
  creator: "Daniel Kliewer",
  publisher: "Daniel Kliewer",
  openGraph: {
    title: "Daniel Kliewer | Software Engineer & AI Developer",
    description: "Explore Daniel Kliewer's portfolio of AI development and software engineering projects. Specializing in LLMs, Next.js, React, and privacy-focused AI solutions that prioritize user control.",
    url: "https://danielkliewer.com",
    siteName: "Daniel Kliewer | Software Engineer & AI Developer",
    images: [
      {
        url: "/og-image.jpg", // Add a proper OG image in the public folder
        width: 1200,
        height: 630,
        alt: "Daniel Kliewer - Software Engineer & AI Developer",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Daniel Kliewer | Software Engineer & AI Developer",
    description: "Explore Daniel Kliewer's portfolio of AI development and software engineering projects. Specializing in LLMs, Next.js, React, and privacy-focused AI solutions.",
    images: ["/og-image.jpg"], // Add a proper Twitter card image in the public folder
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
  // Get featured posts for the slider (first 5 posts)
  const featuredPosts = await getBlogPosts(5);
  // Get recent posts for the grid (next 3 posts)
  const recentPosts = await getBlogPosts(3, 5);
  
  return (
    <div className="container mx-auto px-4 py-12">
      {/* Featured Content Slider */}
      <ContentSlider posts={featuredPosts} />
      
      {/* Hero Section */}
      <section className="flex flex-col items-center text-center mb-16">
        <div className="max-w-3xl">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Daniel Kliewer
          </h1>
          <h2 className="text-2xl md:text-3xl text-primary mb-6">
            Software Engineer & AI Developer
          </h2>
          <div className="bg-secondary/20 p-6 rounded-lg mb-8 border border-secondary/30">
            <p className="text-lg italic mb-2">👋 Welcome to my digital home!</p>
            <p className="text-md">
              Thanks for stopping by. I'm passionate about building technology that makes AI more accessible, 
              ethical, and human-centered. Whether you're here to explore my projects, read my latest thoughts, 
              or connect professionally, I'm glad you're here.
            </p>
          </div>
          <p className="text-lg text-muted-foreground mb-8">
            I build innovative applications that leverage the power of large language models
            and modern web technologies to create useful tools and platforms.
            My focus is on creating accessible AI solutions that prioritize privacy and user control.
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
              <p className="text-sm text-muted-foreground">
                {new Date(post.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Featured Projects Section */}
      <section>
        <h2 className="text-2xl md:text-3xl font-bold mb-8">Featured Projects</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div key={index} className="card overflow-hidden">
              <div className="card-body p-6">
                <h3 className="text-xl font-bold mb-2">{project.title}</h3>
                <p className="text-muted-foreground mb-4">{project.description}</p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.technologies.map((tech, i) => (
                    <span key={i} className="badge badge-outline text-xs px-2 py-1">
                      {tech}
                    </span>
                  ))}
                </div>
                <Link href={project.link} className="text-primary hover:underline">
                  Learn More →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
