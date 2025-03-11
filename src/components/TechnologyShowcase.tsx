'use client';

import React from 'react';
import Link from 'next/link';

interface Technology {
  name: string;
  count: number;
}

interface TechnologyShowcaseProps {
  technologies: Technology[];
  className?: string;
}

const TechnologyShowcase: React.FC<TechnologyShowcaseProps> = ({ technologies, className = '' }) => {
  if (!technologies || technologies.length === 0) return null;
  
  // Sort technologies by count (most used first)
  const sortedTechnologies = [...technologies].sort((a, b) => b.count - a.count);
  
  return (
    <div className={`${className}`}>
      <h2 className="text-2xl md:text-3xl font-bold mb-4">Technologies</h2>
      <p className="text-muted-foreground mb-6">
        Explore the technologies I work with across my projects and blog posts.
      </p>
      
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {sortedTechnologies.map((tech, idx) => (
          <Link 
            key={idx}
            href={`/blog?category=${encodeURIComponent(tech.name)}`}
            className="flex items-center justify-between p-4 bg-card hover:shadow-md transition-shadow rounded-lg border border-border"
          >
            <span className="font-medium">{tech.name}</span>
            <span className="text-sm px-2 py-1 bg-primary/10 text-primary rounded-full">
              {tech.count}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default TechnologyShowcase;
