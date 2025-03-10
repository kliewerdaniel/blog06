export interface PostMetadata {
  slug: string;
  title?: string;
  date: string | Date;
  image?: string;
  excerpt?: string;
  categories?: string[];
  tags?: string[];
  // Additional metadata that might be available from front matter
  author?: string;
  readingTime?: string;
  description?: string;
}

export interface BlogPost extends PostMetadata {
  content: string;
}

export interface TaxonomyItem {
  name: string;
  count: number;
  slug: string;
}

export interface RelatedPost {
  slug: string;
  title: string;
  date: string;
  excerpt?: string;
  image?: string;
  categories?: string[];
}
