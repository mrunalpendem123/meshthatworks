interface MarkProps {
  size?: number;
  className?: string;
}

// A small "mesh" glyph: three nodes wired together, one of them lit — the
// network-of-devices idea distilled into a logomark.
export function MeshMark({ size = 22, className }: MarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-hidden
    >
      <path
        d="M6 7 L12 4 L18 7 M6 7 L6 16 L12 20 M18 7 L18 16 L12 20 M6 7 L18 16 M18 7 L6 16"
        stroke="rgba(255,255,255,0.28)"
        strokeWidth="1"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx="12" cy="4" r="2.1" fill="#5ce8a6" />
      <circle cx="6" cy="7" r="1.7" fill="#e7ebf5" />
      <circle cx="18" cy="7" r="1.7" fill="#e7ebf5" />
      <circle cx="6" cy="16" r="1.7" fill="#9aa3bd" />
      <circle cx="18" cy="16" r="1.7" fill="#9aa3bd" />
      <circle cx="12" cy="20" r="1.7" fill="#9aa3bd" />
    </svg>
  );
}

interface WordmarkProps {
  className?: string;
}

export function Wordmark({ className }: WordmarkProps) {
  return (
    <div className={`flex items-center gap-2 ${className ?? ''}`}>
      <MeshMark size={20} />
      <span className="text-[15px] font-semibold tracking-tight text-ink-0">
        Mesh<span className="text-ink-2 font-normal">That</span>Works
      </span>
    </div>
  );
}
