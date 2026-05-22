import { motion } from 'motion/react';
import { ArrowRight } from 'lucide-react';
import { Globe } from '../components/Globe';
import { Wordmark } from '../components/Brand';
import { GlowButton } from '../components/ui';
import { openUrl } from '../lib/api';

export function Welcome({ onStart }: { onStart: () => void }) {
  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden">
      {/* Globe layer — dead-centre behind the content, drifting up slightly. */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 60 }}
        animate={{ opacity: 1, scale: 1, y: 40 }}
        transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
        className="pointer-events-none absolute inset-0 flex items-center justify-center"
      >
        <Globe size={600} speed={0.0025} />
      </motion.div>

      {/* Foreground column — guaranteed centred via flex, top-to-bottom. */}
      <div className="relative z-10 flex h-full w-full flex-col items-center justify-between py-[10vh]">
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex flex-col items-center"
        >
          <Wordmark className="scale-[1.4]" />
          <p className="mt-6 max-w-sm text-center text-[13px] leading-relaxed text-ink-2">
            Frontier AI on the Macs you already own.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="flex flex-col items-center gap-4"
        >
          <GlowButton variant="primary" onClick={onStart} className="px-8 py-3 text-[15px]">
            Start now
            <ArrowRight size={17} />
          </GlowButton>
          <button
            onClick={() => openUrl('https://github.com/mrunalpendem123/meshthatworks')}
            className="text-[12.5px] text-ink-3 transition-colors hover:text-ink-1"
          >
            New here? <span className="text-ink-1 underline underline-offset-2">How it works</span>
          </button>
        </motion.div>
      </div>
    </div>
  );
}
