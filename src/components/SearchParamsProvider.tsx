'use client';

import { useSearchParams as useNextSearchParams } from 'next/navigation';
import { ReactNode } from 'react';

interface SearchParamsProviderProps {
  children: (searchParams: ReturnType<typeof useNextSearchParams>) => ReactNode;
}

export function SearchParamsProvider({ children }: SearchParamsProviderProps) {
  const searchParams = useNextSearchParams();
  return <>{children(searchParams)}</>;
}
