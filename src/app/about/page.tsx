import { Metadata } from "next";
import PageTransition from "@/components/PageTransition";

// Separate viewport export as recommended by Next.js
export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: "About Daniel Kliewer | Software Engineer & AI Developer",
  description: "Learn about Daniel Kliewer's journey through technology and creativity, featuring his expertise in data annotation, software development, and artificial intelligence.",
  keywords: "Daniel Kliewer, data annotation, software development, AI, Python, JavaScript, TypeScript, Next.js, React, TailwindCSS, Ollama, LangChain, Reddit Data Analysis, contact, AI developer, collaboration, machine learning, web development, hire developer",
  authors: [{ name: "Daniel Kliewer" }],
  openGraph: {
    title: "About Daniel Kliewer | Software Engineer & AI Developer",
    description: "Learn about Daniel Kliewer's journey through technology and creativity, featuring his expertise in data annotation, software development, and artificial intelligence.",
    url: "https://danielkliewer.com/about",
    siteName: "Daniel Kliewer | Software Engineer & AI Developer",
    type: "profile",
  },
  twitter: {
    card: "summary_large_image",
    title: "About Daniel Kliewer | Software Engineer & AI Developer",
    description: "Learn about Daniel Kliewer's journey through technology and creativity, featuring his expertise in data annotation, software development, and artificial intelligence.",
  },
  alternates: {
    canonical: "https://danielkliewer.com/about",
  },
};

export default function AboutPage() {
  return (
    <PageTransition transitionType="fade">
      <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">About Me</h1>
      
      {/* Professional Overview */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Daniel Kliewer: A Journey Through Technology and Creativity</h2>
        <p className="text-lg mb-6">
          I&apos;m a multifaceted professional with a rich background in data annotation, software development, 
          and artificial intelligence. With over a decade of experience in data annotation, I&apos;ve honed my 
          skills in understanding and structuring data, providing invaluable insights into machine learning 
          processes. My passion for technology is complemented by a deep appreciation for art, allowing me 
          to approach problems with both analytical precision and creative flair.
        </p>
        
        <div className="flex flex-wrap gap-2 my-6">
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Data Annotation</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Software Development</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Artificial Intelligence</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Python</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">JavaScript</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">TypeScript</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Next.js</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">React</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">TailwindCSS</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">Ollama</span>
          <span className="badge bg-primary/10 text-primary px-4 py-2 rounded-full">LangChain</span>
        </div>
      </section>
      
      {/* Professional Experience */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Professional Experience</h2>
        
        <div className="space-y-8">
          <div className="card p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-2">
              <h3 className="text-xl font-bold">Data Annotation Specialist</h3>
              <div className="text-sm text-muted-foreground italic">Freelance | 2015 – Present</div>
            </div>
            <ul className="space-y-2 list-disc pl-5">
              <li>Collaborated with various organizations to annotate and structure datasets, enhancing machine learning model accuracy</li>
              <li>Developed guidelines and protocols to ensure consistency and quality in data labeling</li>
              <li>Trained and supervised teams of annotators, leading to increased productivity and reduced error rates</li>
            </ul>
          </div>
          
          <div className="card p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-2">
              <h3 className="text-xl font-bold">Software Developer & AI Enthusiast</h3>
              <div className="text-sm text-muted-foreground italic">Self-Employed | 2018 – Present</div>
            </div>
            <ul className="space-y-2 list-disc pl-5">
              <li>Designed and implemented the <strong>Reddit Data Analysis</strong> program, fetching and analyzing Reddit content to generate professional blog posts. This project leverages local models via Ollama to process data efficiently</li>
              <li>Developed the <strong>Next-Ollama-App</strong>, a privacy-focused AI agent framework built with Next.js, Ollama, React, TypeScript, and TailwindCSS. This tool integrates local large language models (LLMs) seamlessly into web-based workflows without relying on cloud APIs</li>
              <li>Created a <strong>Multi-Agent Reddit Analysis System</strong>, utilizing a network of specialized agents to process Reddit content, offering deep insights into user communication patterns and behavioral tendencies</li>
            </ul>
          </div>
          
          <div className="card p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-2">
              <h3 className="text-xl font-bold">Community Engagement</h3>
              <div className="text-sm text-muted-foreground italic">2020 – Present</div>
            </div>
            <ul className="space-y-2 list-disc pl-5">
              <li>Active contributor to Reddit communities such as r/ArtificialIntelligence, r/ChatGPTCoding, and r/LLMDevs, sharing knowledge and engaging in discussions about AI development, coding practices, and technological advancements</li>
              <li>Organized the <strong>Loco Local LocalLLaMa Hackathon 1.2</strong>, fostering innovation and collaboration among AI enthusiasts</li>
            </ul>
          </div>
        </div>
      </section>
      
      {/* Technical Skills */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Technical Skills</h2>
        <div className="card p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-bold mb-3">Programming Languages</h3>
              <ul className="space-y-2 list-disc pl-5">
                <li>Python</li>
                <li>JavaScript</li>
                <li>TypeScript</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-3">Frameworks & Tools</h3>
              <ul className="space-y-2 list-disc pl-5">
                <li>Next.js</li>
                <li>React</li>
                <li>TailwindCSS</li>
                <li>Ollama</li>
                <li>LangChain</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-3">Data Annotation</h3>
              <ul className="space-y-2 list-disc pl-5">
                <li>Expertise in labeling and structuring datasets for machine learning applications</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-3">AI Integration</h3>
              <ul className="space-y-2 list-disc pl-5">
                <li>Experience in integrating local LLMs into applications, enhancing functionality and user experience</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
      
      {/* Personal Projects and Insights */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Personal Projects and Insights</h2>
        <p className="text-lg mb-6">
          My journey is characterized by a relentless pursuit of knowledge and a desire to bridge the gap 
          between technology and human behavior. My projects often stem from personal curiosity and a 
          commitment to open-source principles, aiming to empower others through shared knowledge.
        </p>
        
        <div className="space-y-8">
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-2">Reddit Data Analysis</h3>
            <p className="mb-4">
              This project reflects my interest in understanding online communities. By analyzing Reddit content, 
              I provide insights into user behavior and communication patterns, contributing to a deeper 
              understanding of digital interactions.
            </p>
          </div>
          
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-2">Next-Ollama-App</h3>
            <p className="mb-4">
              Demonstrating my commitment to privacy and efficiency, I developed this framework to allow 
              seamless integration of local AI models into web applications, reducing reliance on cloud 
              services and enhancing data security.
            </p>
          </div>
          
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-2">Multi-Agent Reddit Analysis System</h3>
            <p className="mb-4">
              This system showcases my innovative approach to data analysis, employing multiple agents to 
              dissect and interpret complex datasets, offering valuable perspectives on user behavior.
            </p>
          </div>
        </div>
      </section>
      
      {/* Philosophy and Approach */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Philosophy and Approach</h2>
        
        <div className="bg-muted p-6 rounded-lg border-l-4 border-accent my-8">
          <blockquote className="text-xl italic font-medium">
            &quot;Technology should be accessible and serve as a tool for understanding human behavior.&quot;
          </blockquote>
        </div>
        
        <p className="text-lg mb-6">
          I believe in the power of community and open-source collaboration. My work is driven by a desire 
          to make technology accessible and to use it as a tool for understanding human behavior. By combining 
          my technical skills with my artistic sensibilities, I approach problems holistically, ensuring that 
          my solutions are both functional and empathetic.
        </p>
        
        <p className="text-lg mb-6">
          Through my journey of overcoming challenges and rebuilding, I&apos;ve developed a unique perspective that 
          informs my approach to technology and problem-solving. I strive to create solutions that not only 
          meet technical requirements but also consider the human element, making technology more accessible 
          and beneficial for all.
        </p>
      </section>
      
      {/* Technology & AI Development */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Technology & AI Development</h2>
        <div className="card p-6">
          <ul className="space-y-2 list-disc pl-5">
            <li><strong>Machine Learning & AI Integration</strong> – Developing local AI models, optimizing LLM workflows, and integrating AI into applications.</li>
            <li><strong>Software Engineering</strong> – Creating robust, scalable solutions with a focus on security, automation, and performance.</li>
            <li><strong>Web Development & SEO</strong> – Building modern, efficient web applications using frameworks like Jekyll, Hugo, and React while ensuring high search engine visibility.</li>
            <li><strong>Open-Source & Decentralized AI</strong> – Advocating for self-reliant computing, fostering open-source AI projects, and promoting ethical AI usage.</li>
            <li><strong>Linguistic & Persona Modeling</strong> – Crafting AI-driven personas, analyzing writing styles, and experimenting with computational creativity.</li>
          </ul>
        </div>
      </section>
      
      {/* Artistic Creations */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Artistic Creations</h2>
        <div className="card p-6">
          <ul className="space-y-2 list-disc pl-5">
            <li><strong>Digital & Traditional Art</strong> – Exploring abstract, surreal, and conceptual art forms through digital painting, graphic design, and mixed media.</li>
            <li><strong>AI-Generated Art & Writing</strong> – Experimenting with AI-generated literature, poetry, and visual storytelling.</li>
            <li><strong>Music & Audio Experiments</strong> – Creating AI-assisted music compositions, soundscapes, and voice synthesis projects.</li>
            <li><strong>Experimental Literature & Fiction</strong> – Blending human and AI-generated narratives, refining unique writing styles, and pushing the boundaries of digital storytelling.</li>
            <li><strong>Diss Track AI</strong> – Developing an AI-powered diss track generator that scrapes online conversations and transforms them into rap battles.</li>
            <li><strong>Interactive & Generative Media</strong> – Exploring algorithmic art, computational creativity, and AI-assisted storytelling.</li>
          </ul>
        </div>
      </section>
      
      {/* Contact Section */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Get in Touch</h2>
        <div className="prose prose-lg dark:prose-invert mb-8">
          <p>
            As a software engineer, AI developer, and multidisciplinary artist, I am passionate about leveraging cutting-edge technology to solve complex problems while exploring the intersection of art and computation.
          </p>
          <p>
            If you're interested in discussing a project, collaboration, or just exchanging ideas about technology, AI, or creative work, I'd love to connect.
          </p>
          <p>
            I am dedicated to leveraging technology to address complex challenges, and I welcome the opportunity to discuss:
          </p>
          <ul>
            <li>Machine Learning projects and implementations</li>
            <li>Software development best practices</li>
            <li>Web development with modern frameworks</li>
            <li>AI integration into existing applications</li>
            <li>Potential collaborations and opportunities</li>
          </ul>
        </div>
        
        <div className="card p-6">
          <h3 className="text-xl font-bold mb-4">Contact Information</h3>
          <div className="flex flex-col space-y-4">
            <p className="flex items-center"><span className="mr-2">📧</span><a href="mailto:danielkliewer@gmail.com" className="text-primary hover:underline">danielkliewer@gmail.com</a></p>
            <p className="flex items-center"><span className="mr-2">💻</span><a href="https://github.com/kliewerdaniel" className="text-primary hover:underline">github.com/kliewerdaniel</a></p>
            <p className="flex items-center"><span className="mr-2">🌐</span><a href="https://www.danielkliewer.com" className="text-primary hover:underline">danielkliewer.com</a></p>
            <p className="flex items-center"><span className="mr-2">🤖</span><a href="https://www.reddit.com/user/KonradFreeman" className="text-primary hover:underline">u/KonradFreeman</a></p>
            <p className="flex items-center"><span className="mr-2">📍</span>Austin, Texas, United States</p>
          </div>
        </div>
      </section>
      
      {/* Copyright Section */}
      <section className="mt-16 text-center text-sm text-muted-foreground">
        <p>© 2025 Daniel Kliewer. All rights reserved.</p>
      </section>
      </div>
    </PageTransition>
  );
}
