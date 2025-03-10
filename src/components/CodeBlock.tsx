'use client';

import React, { useRef } from 'react';
import { CopyButton } from './CopyButton';

interface CodeBlockProps {
  code: string;
  language: string;
  className?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ 
  code, 
  language,
  className = ''
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const codeBlockRef = useRef<HTMLPreElement>(null);

  return (
    <div ref={containerRef} className="relative my-6">
      <pre
        ref={codeBlockRef}
        className={`p-4 rounded-md bg-gray-800 text-gray-200 overflow-x-auto ${className}`}
      >
        <code className={`language-${language || 'text'}`}>
          {code}
        </code>
      </pre>
      <CopyButton 
        code={code} 
        containerRef={containerRef} 
        codeBlockRef={codeBlockRef} 
      />
    </div>
  );
};
