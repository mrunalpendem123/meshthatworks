import type { Metadata } from 'next';
import { JetBrains_Mono } from 'next/font/google';
import './globals.css';

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'meshthatworks — frontier AI on the Macs you already own',
  description:
    'Run frontier MoE models across consumer Apple Silicon devices. Treats your SSD as memory and splits models across paired Macs. Local, private, no cloud. MIT.',
  openGraph: {
    title: 'meshthatworks',
    description: 'Frontier AI on the Macs you already own. Local, private, MIT-licensed.',
    url: 'https://github.com/mrunalpendem123/meshthatworks',
    siteName: 'meshthatworks',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={mono.variable}>
      <body>{children}</body>
    </html>
  );
}
