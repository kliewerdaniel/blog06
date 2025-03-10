import fs from "fs";
import path from "path";
import matter from "gray-matter";
import Link from "next/link";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { BlogJsonLd } from "@/components/BlogJsonLd";
import { Metadata, ResolvingMetadata } from "next/types";

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
  } catch (error) {
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

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const resolvedParams = await params;
  const { slug } = resolvedParams;
  const postsDirectory = path.join(process.cwd(), "_posts");
  const filePath = path.join(postsDirectory, `${slug}.md`);
  
  try {
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);
    
    const title = data.title || "Untitled";
    // Use a fixed date if none provided to avoid hydration mismatches
    const date = data.date ? new Date(data.date) : new Date('2025-01-01T00:00:00Z');
    
    return (
      <div className="container mx-auto px-4 py-12">
        <BlogJsonLd 
          title={title}
          description={data.description || `${title} - Article by Daniel Kliewer`}
          datePublished={date.toISOString()}
          authorName="Daniel Kliewer"
          slug={slug}
          tags={data.tags || []}
          categories={data.categories || []}
        />
        
        <Link href="/blog" className="text-primary hover:underline mb-6 inline-block">
          ← Back to Blog
        </Link>
        
        <article className="mt-8 max-w-3xl mx-auto">
          <h1 className="text-4xl font-bold mb-4">{title}</h1>
          <time dateTime={date.toISOString()} className="text-muted-foreground block mb-8">
            {/* Use a fixed date format to prevent hydration mismatches caused by locale differences */}
            {new Intl.DateTimeFormat('en-US', {
              weekday: 'long',
              year: 'numeric',
              month: 'long', 
              day: 'numeric',
              timeZone: 'UTC' // Ensure consistent timezone between server and client
            }).format(date)}
          </time>
          
          <MarkdownRenderer content={content} />
        </article>
      </div>
    );
  } catch (error) {
    return (
      <div className="container mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold mb-8">Post Not Found</h1>
        <p>Sorry, the blog post you're looking for does not exist.</p>
        <Link href="/blog" className="text-primary hover:underline mt-4 inline-block">
          ← Back to Blog
        </Link>
      </div>
    );
  }
}
