'use client';

import React, { useState } from 'react';
import { TaxonomyItem } from '@/types/blog';

interface CategoryFilterProps {
  categories: TaxonomyItem[];
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
}

const CategoryFilter: React.FC<CategoryFilterProps> = ({
  categories,
  selectedCategory,
  onSelectCategory,
}) => {
  const handleCategoryClick = (category: string | null) => {
    onSelectCategory(category);
  };

  return (
    <div className="mb-8">
      <h3 className="text-lg font-medium mb-3">Categories</h3>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => handleCategoryClick(null)}
          className={`px-3 py-1 rounded-full text-sm ${
            selectedCategory === null
              ? 'bg-primary text-primary-foreground'
              : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
        >
          All
        </button>
        
        {categories.map((category) => (
          <button
            key={category.slug}
            onClick={() => handleCategoryClick(category.name)}
            className={`px-3 py-1 rounded-full text-sm ${
              selectedCategory === category.name
                ? 'bg-primary text-primary-foreground'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            {category.name} <span className="text-xs opacity-75">({category.count})</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default CategoryFilter;
