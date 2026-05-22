import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const sans = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'MeshThatWorks — frontier AI on the Macs you already own',
  description:
    'Run frontier MoE models across consumer Apple Silicon devices. Treats your SSD as memory and splits models across paired Macs. Local, private, no cloud. MIT.',
  openGraph: {
    title: 'MeshThatWorks',
    description: 'Frontier AI on the Macs you already own. Local, private, MIT-licensed.',
    url: 'https://meshthatworks.vercel.app',
    siteName: 'MeshThatWorks',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <div className="space-bg" />
        <div className="space-stars" />
        {children}
      </body>
    </html>
  );
}
