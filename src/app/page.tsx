import Link from "next/link";
import Image from "next/image";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { Metadata } from "next";
import PageTransition from "@/components/PageTransition";
import ContentSlider from '@/components/ContentSlider';
import ClientHomePage from './page.client';

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
  }
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

export default async function Home() {
  // Get recent posts for the grid (first 4 posts)
  const recentPosts = await getBlogPosts(4);
  
  // Convert projects to a format compatible with ContentSlider
  const projectsForSlider = projects.map(project => ({
    slug: project.link.replace(/^\/blog\//, ""),
    title: project.title,
    excerpt: project.description,
    categories: project.technologies,
    date: new Date().toISOString(),
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
    .filter(tech => tech.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 12); // Only take top 12 technologies
  
  // Pass the data to the client component
  return (
    <ClientHomePage 
      recentPosts={recentPosts}
      projects={projects}
      technologies={technologies}
    />
  );
}
