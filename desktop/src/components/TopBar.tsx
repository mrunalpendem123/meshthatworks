import { Boxes, MessageSquare, Radar, Share2 } from 'lucide-react';
import { Wordmark } from './Brand';
import { StatusDot } from './ui';
import { useApp, type Route } from '../lib/store';
import { shortId, cx } from '../lib/util';

const NAV: { route: Route; label: string; Icon: typeof Radar }[] = [
  { route: 'node', label: 'Node', Icon: Radar },
  { route: 'chat', label: 'Chat', Icon: MessageSquare },
  { route: 'mesh', label: 'Mesh', Icon: Share2 },
  { route: 'models', label: 'Models', Icon: Boxes },
];

// The frameless overlay title bar: draggable, with the macOS traffic lights
// floating top-left (hence the left inset). Wordmark, segmented nav, node chip.
export function TopBar() {
  const route = useApp((s) => s.route);
  const setRoute = useApp((s) => s.setRoute);
  const engine = useApp((s) => s.engine);
  const nodeId = engine.status?.endpoint_id;

  return (
    <div className="drag-region relative z-20 flex h-14 items-center justify-between pl-[88px] pr-4">
      <div className="no-drag">
        <Wordmark />
      </div>

      <div className="no-drag glass flex items-center gap-1 rounded-full p-1">
        {NAV.map(({ route: r, label, Icon }) => (
          <button
            key={r}
            onClick={() => setRoute(r)}
            className={cx(
              'flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[13px] transition-colors',
              route === r ? 'bg-white/10 text-ink-0' : 'text-ink-2 hover:text-ink-1',
            )}
          >
            <Icon size={15} strokeWidth={2} />
            {label}
          </button>
        ))}
      </div>

      <div className="no-drag flex items-center gap-2">
        <div className="pill px-3 py-1.5 text-[12.5px] text-ink-1">
          <StatusDot phase={engine.phase} />
          <span className="font-mono">{shortId(nodeId, 6, 4)}</span>
        </div>
      </div>
    </div>
  );
}
