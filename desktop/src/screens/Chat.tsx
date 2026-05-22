import { memo, useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { ArrowUp, Square, Sparkles } from 'lucide-react';
import { MeshMark } from '../components/Brand';
import { StatusDot } from '../components/ui';
import { chatStream } from '../lib/api';
import { useApp } from '../lib/store';
import { cx } from '../lib/util';

interface Msg {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  done: boolean;
  error?: boolean;
}

const SUGGESTIONS = [
  'Write a haiku about distributed systems',
  'Explain Mixture-of-Experts like I’m five',
  'Refactor this Python function for clarity',
  'What can I run on an 8 GB Mac?',
];

export function Chat() {
  const engine = useApp((s) => s.engine);
  const setRoute = useApp((s) => s.setRoute);
  const ready = engine.phase === 'ready';
  const modelName = engine.status?.model?.name;

  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const idRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [msgs]);

  async function send(text: string) {
    const prompt = text.trim();
    if (!prompt || streaming || !ready) return;
    setInput('');
    const userMsg: Msg = { id: ++idRef.current, role: 'user', content: prompt, done: true };
    const botMsg: Msg = { id: ++idRef.current, role: 'assistant', content: '', done: false };
    const history = [...msgs, userMsg];
    setMsgs([...history, botMsg]);
    setStreaming(true);

    const wire = history.map((m) => ({ role: m.role, content: m.content }));
    try {
      await chatStream(wire, { maxTokens: 1024, temperature: 0.7 }, (e) => {
        if (e.type === 'delta') {
          setMsgs((cur) =>
            cur.map((m) => (m.id === botMsg.id ? { ...m, content: m.content + e.content } : m)),
          );
        } else if (e.type === 'done') {
          setMsgs((cur) => cur.map((m) => (m.id === botMsg.id ? { ...m, done: true } : m)));
          setStreaming(false);
        } else if (e.type === 'error') {
          setMsgs((cur) =>
            cur.map((m) =>
              m.id === botMsg.id ? { ...m, content: e.message, done: true, error: true } : m,
            ),
          );
          setStreaming(false);
        }
      });
    } catch {
      setStreaming(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col">
      {/* messages */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 pt-2">
        {msgs.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <MeshMark size={40} />
            <h2 className="mt-4 text-xl font-semibold text-ink-0">
              {ready ? 'Ask your mesh anything' : 'Your node is offline'}
            </h2>
            <p className="mt-1.5 max-w-sm text-[13px] text-ink-2">
              {ready
                ? `Running ${modelName ?? 'a local model'} across your own hardware. Nothing leaves your machine.`
                : 'Start the node from the Node tab, then come back here.'}
            </p>
            {ready && (
              <div className="mt-6 grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="glass rounded-xl px-4 py-3 text-left text-[12.5px] text-ink-1 transition-colors hover:text-ink-0"
                  >
                    <Sparkles size={13} className="mb-1 text-aurora-violet" />
                    <div>{s}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-5 pb-4">
            {msgs.map((m) => (
              <Bubble key={m.id} msg={m} streaming={streaming} />
            ))}
          </div>
        )}
      </div>

      {/* composer */}
      <div className="px-4 pb-5 pt-2">
        {!ready && (
          <button
            onClick={() => setRoute('node')}
            className="mb-2 flex w-full items-center justify-center gap-2 rounded-xl border border-aurora-amber/25 bg-aurora-amber/10 py-2 text-[12.5px] text-aurora-amber"
          >
            <StatusDot phase={engine.phase} /> {engine.message} — start it on the Node tab
          </button>
        )}
        <div className="input-glass flex items-end gap-2 rounded-2xl p-2">
          <textarea
            ref={taRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder={ready ? 'Message your mesh…' : 'Node offline'}
            disabled={!ready}
            className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-3 py-2 text-[14px] text-ink-0 placeholder:text-ink-3 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={() => send(input)}
            disabled={!ready || streaming || !input.trim()}
            className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-node text-space-0 transition-opacity disabled:opacity-30"
          >
            {streaming ? <Square size={15} fill="currentColor" /> : <ArrowUp size={18} strokeWidth={2.5} />}
          </button>
        </div>
        <p className="mt-2 text-center text-[11px] text-ink-3">
          {modelName ? `${modelName} · ` : ''}local & private · Enter to send, Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}

const Bubble = memo(function Bubble({ msg, streaming }: { msg: Msg; streaming: boolean }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-white/10 px-4 py-2.5 text-[14px] leading-relaxed text-ink-0">
          {msg.content}
        </div>
      </div>
    );
  }
  const isStreamingThis = streaming && !msg.done;
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-node/12">
        <MeshMark size={15} />
      </div>
      <div className="min-w-0 flex-1">
        {msg.content ? (
          <div className={cx('prose-chat text-ink-1', msg.error && 'text-aurora-amber')}>
            <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {msg.content}
            </Markdown>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 py-1 text-ink-3">
            <span className="h-2 w-2 animate-breathe rounded-full bg-node" />
            <span className="text-[12.5px]">thinking…</span>
          </div>
        )}
        {isStreamingThis && msg.content && (
          <span className="ml-0.5 inline-block h-4 w-[2px] animate-breathe bg-node align-middle" />
        )}
      </div>
    </div>
  );
});
