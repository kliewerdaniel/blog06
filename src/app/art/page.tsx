'use client';

import Image from 'next/image';
import type { Metadata } from 'next';

// Metadata must be exported from a Server Component, so we'll define it here but it won't be used
// The actual metadata will be handled by a separate layout file if needed
const metadata: Metadata = {
  title: 'Art Gallery',
  description: 'Gallery of artwork by Daniel Kliewer',
};

interface ArtImage {
  src: string;
  alt: string;
  width: number;
  height: number;
}

export default function ArtGallery() {
  const artImages: ArtImage[] = [
    { src: '/art/FB_IMG_1686089299754.jpg', alt: 'Artwork 1', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089748977.jpg', alt: 'Artwork 2', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089761548.jpg', alt: 'Artwork 3', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089772880.jpg', alt: 'Artwork 4', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089783737.jpg', alt: 'Artwork 5', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089791426.jpg', alt: 'Artwork 6', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089798139.jpg', alt: 'Artwork 7', width: 800, height: 800 },
    { src: '/art/FB_IMG_1686089803343.jpg', alt: 'Artwork 8', width: 800, height: 800 },
    { src: '/art/IMG_0004.JPG', alt: 'Artwork 9', width: 800, height: 800 },
    { src: '/art/IMG_0009.PNG', alt: 'Artwork 10', width: 800, height: 800 },
    { src: '/art/IMG_0010.JPG', alt: 'Artwork 11', width: 800, height: 800 },
    { src: '/art/IMG_0019.JPG', alt: 'Artwork 12', width: 800, height: 800 },
    { src: '/art/IMG_0041.JPG', alt: 'Artwork 13', width: 800, height: 800 },
    { src: '/art/IMG_0052.JPG', alt: 'Artwork 14', width: 800, height: 800 },
  ];

  return (
    <div className="container py-12">
      <h1 className="text-3xl font-bold mb-8">Art Gallery</h1>
      
      <p className="text-lg mb-8">
        Welcome to my art gallery. Here you can find a collection of my artwork and photos of me with my art.
      </p>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {artImages.map((image, index) => (
          <div key={index} className="art-card hover:-translate-y-1 transition-transform duration-300">
            <div className="relative overflow-hidden rounded-lg shadow-md hover:shadow-xl transition-all duration-300 h-64">
              <Image
                src={image.src}
                alt={image.alt}
                fill
                sizes="(max-width: 640px) 100vw, (max-width: 768px) 50vw, (max-width: 1024px) 33vw, 25vw"
                className="object-cover"
                priority={index < 4}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
