'use client';

import React from 'react';

interface TechTagsProps {
  technologies: string[];
  className?: string;
}

const TechTags: React.FC<TechTagsProps> = ({ technologies, className = '' }) => {
  if (!technologies || technologies.length === 0) return null;
  
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {technologies.map((tech, idx) => (
        <span 
          key={idx} 
          className="inline-block text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
        >
          {tech}
        </span>
      ))}
    </div>
  );
};

export default TechTags;
