import createGlobe from 'cobe';
import { useEffect, useMemo, useRef } from 'react';

export interface GlobeMarker {
  location: [number, number];
  size: number;
}

interface GlobeProps {
  /** Rendered CSS size in px (square). */
  size?: number;
  markers?: GlobeMarker[];
  /** Auto-rotation speed; 0 to freeze. */
  speed?: number;
  className?: string;
  /** Brighten markers / glow when the node is live. */
  active?: boolean;
}

// A dotted WebGL globe (cobe) — the centrepiece of the UI, echoing the
// LayerEdge reference. Drag to spin; it auto-rotates otherwise. Peers are
// plotted as glowing green markers.
export function Globe({ size = 480, markers = [], speed = 0.004, className, active = false }: GlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointerInteracting = useRef<number | null>(null);
  const pointerMovement = useRef(0);
  const markerKey = useMemo(() => JSON.stringify(markers), [markers]);

  useEffect(() => {
    let phi = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelSize = size * dpr;

    const globe = createGlobe(canvasRef.current!, {
      devicePixelRatio: dpr,
      width: pixelSize,
      height: pixelSize,
      phi: 0,
      theta: 0.22,
      dark: 1,
      diffuse: 1.1,
      mapSamples: 20000,
      mapBrightness: active ? 7 : 4.2,
      baseColor: [0.13, 0.15, 0.22],
      markerColor: [0.36, 0.91, 0.65],
      glowColor: active ? [0.16, 0.34, 0.28] : [0.13, 0.16, 0.26],
      markers,
      onRender: (state) => {
        if (pointerInteracting.current === null) phi += speed;
        state.phi = phi + pointerMovement.current / 200;
        state.width = pixelSize;
        state.height = pixelSize;
      },
    });

    // Fade the canvas in once the first frame paints.
    const el = canvasRef.current!;
    requestAnimationFrame(() => (el.style.opacity = '1'));
    return () => globe.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, markerKey, speed, active]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        width: size,
        height: size,
        maxWidth: '100%',
        aspectRatio: '1',
        cursor: 'grab',
        opacity: 0,
        transition: 'opacity 1s ease',
        contain: 'layout paint size',
      }}
      onPointerDown={(e) => {
        pointerInteracting.current = e.clientX - pointerMovement.current;
        (e.currentTarget as HTMLCanvasElement).style.cursor = 'grabbing';
      }}
      onPointerUp={(e) => {
        pointerInteracting.current = null;
        (e.currentTarget as HTMLCanvasElement).style.cursor = 'grab';
      }}
      onPointerOut={(e) => {
        pointerInteracting.current = null;
        (e.currentTarget as HTMLCanvasElement).style.cursor = 'grab';
      }}
      onPointerMove={(e) => {
        if (pointerInteracting.current !== null) {
          pointerMovement.current = e.clientX - pointerInteracting.current;
        }
      }}
    />
  );
}

// Spread N peer markers around the globe deterministically so the same peer
// set always lands in the same spots.
export function peerMarkers(count: number): GlobeMarker[] {
  const presets: [number, number][] = [
    [37.77, -122.42], // SF — "this node" anchor
    [40.71, -74.0], // NYC
    [51.5, -0.12], // London
    [35.68, 139.69], // Tokyo
    [1.35, 103.82], // Singapore
    [-33.86, 151.2], // Sydney
    [52.52, 13.4], // Berlin
    [19.07, 72.87], // Mumbai
  ];
  const markers: GlobeMarker[] = [{ location: presets[0], size: 0.1 }];
  for (let i = 0; i < count; i++) {
    markers.push({ location: presets[(i + 1) % presets.length], size: 0.06 });
  }
  return markers;
}
