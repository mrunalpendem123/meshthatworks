export function formatBytes(bytes: number): string {
  if (!bytes || bytes < 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(val >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

/** 0x-style short id, like the wallet chip in the reference. */
export function shortId(id: string | undefined | null, lead = 6, tail = 4): string {
  if (!id) return '—';
  const clean = id.replace(/^0x/i, '');
  if (clean.length <= lead + tail) return id;
  return `${clean.slice(0, lead)}…${clean.slice(-tail)}`;
}

export function uptime(startedUnix: number | undefined): string {
  if (!startedUnix) return '—';
  const secs = Math.max(0, Math.floor(Date.now() / 1000) - startedUnix);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}
