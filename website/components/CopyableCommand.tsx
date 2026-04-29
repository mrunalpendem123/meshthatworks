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
      // clipboard write can fail in restricted contexts; fall back to selection
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
    <div className="flex items-center gap-3 px-4 py-3">
      <code
        id="install-cmd"
        className="text-fg select-all break-all flex-1 leading-relaxed text-[13px]"
      >
        {cmd}
      </code>
      <button
        onClick={onCopy}
        className="shrink-0 text-xs text-fgDim hover:text-fg border border-line rounded px-2 py-1 transition-colors"
        aria-label="Copy install command"
      >
        {copied ? '✓ copied' : 'copy'}
      </button>
    </div>
  );
}
