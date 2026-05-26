'use client';

import createGlobe from 'cobe';
import { useEffect, useRef } from 'react';

interface GlobeProps {
  size?: number;
  speed?: number;
  className?: string;
}

// Dotted WebGL globe (cobe) — the same centrepiece as the desktop app.
// Drag to spin; auto-rotates otherwise; peers plotted as green markers.
export function Globe({ size = 560, speed = 0.0026, className }: GlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointer = useRef<number | null>(null);
  const movement = useRef(0);

  useEffect(() => {
    let phi = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const px = size * dpr;

    const globe = createGlobe(canvasRef.current!, {
      devicePixelRatio: dpr,
      width: px,
      height: px,
      phi: 0,
      theta: 0.22,
      dark: 1,
      diffuse: 1.1,
      mapSamples: 20000,
      mapBrightness: 5.2,
      baseColor: [0.13, 0.15, 0.22],
      markerColor: [0.36, 0.91, 0.65],
      glowColor: [0.14, 0.18, 0.28],
      markers: [
        { location: [37.77, -122.42], size: 0.09 },
        { location: [51.5, -0.12], size: 0.06 },
        { location: [35.68, 139.69], size: 0.06 },
        { location: [1.35, 103.82], size: 0.05 },
        { location: [19.07, 72.87], size: 0.05 },
      ],
      onRender: (state) => {
        if (pointer.current === null) phi += speed;
        state.phi = phi + movement.current / 200;
        state.width = px;
        state.height = px;
      },
    });

    const el = canvasRef.current!;
    requestAnimationFrame(() => (el.style.opacity = '1'));
    return () => globe.destroy();
  }, [size, speed]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        // Clamp BOTH dimensions to the viewport so the canvas stays square on
        // mobile (a fixed height with a clamped width made it non-square →
        // the globe rendered off-center / stretched).
        width: `min(${size}px, 78vw)`,
        height: `min(${size}px, 78vw)`,
        cursor: 'grab',
        opacity: 0,
        transition: 'opacity 1.2s ease',
        contain: 'layout paint size',
      }}
      onPointerDown={(e) => {
        pointer.current = e.clientX - movement.current;
        e.currentTarget.style.cursor = 'grabbing';
      }}
      onPointerUp={(e) => {
        pointer.current = null;
        e.currentTarget.style.cursor = 'grab';
      }}
      onPointerOut={(e) => {
        pointer.current = null;
        e.currentTarget.style.cursor = 'grab';
      }}
      onPointerMove={(e) => {
        if (pointer.current !== null) movement.current = e.clientX - pointer.current;
      }}
    />
  );
}
