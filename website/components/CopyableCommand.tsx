'use client';

import { useState } from 'react';

export function CopyableCommand({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      const sel = window.getSelection();
      const range = document.createRange();
      const node = document.getElementById('install-cmd');
      if (sel && node) {
        sel.removeAllRanges();
        range.selectNodeContents(node);
        sel.addRange(range);
      }
    }
  };

  return (
    <div className="flex items-center gap-3">
      <code
        id="install-cmd"
        className="mono flex-1 select-all break-all text-[12.5px] leading-relaxed text-ink-1"
      >
        {cmd}
      </code>
      <button
        onClick={onCopy}
        className="shrink-0 rounded-lg border border-line px-2.5 py-1 text-[11.5px] text-ink-2 transition-colors hover:border-white/25 hover:text-ink-0"
        aria-label="Copy install command"
      >
        {copied ? '✓ copied' : 'copy'}
      </button>
    </div>
  );
}
