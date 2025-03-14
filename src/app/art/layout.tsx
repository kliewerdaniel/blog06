import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Art Gallery',
  description: 'Gallery of artwork by Daniel Kliewer',
};

export default function ArtLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <>{children}</>;
}
