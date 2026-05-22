import { useCallback, useEffect } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { TopBar } from './components/TopBar';
import { Welcome } from './screens/Welcome';
import { Onboarding } from './screens/Onboarding';
import { Dashboard } from './screens/Dashboard';
import { Chat } from './screens/Chat';
import { Mesh } from './screens/Mesh';
import { Models } from './screens/Models';
import { getSetupState, onEngineStatus } from './lib/api';
import { useApp } from './lib/store';

export default function App() {
  const route = useApp((s) => s.route);
  const setRoute = useApp((s) => s.setRoute);
  const setEngine = useApp((s) => s.setEngine);
  const setSetup = useApp((s) => s.setSetup);

  const refreshSetup = useCallback(async () => {
    const s = await getSetupState();
    setSetup(s);
    return;
  }, [setSetup]);

  useEffect(() => {
    refreshSetup();
    const un = onEngineStatus(setEngine);
    return () => {
      un.then((f) => f());
    };
  }, [refreshSetup, setEngine]);

  async function start() {
    const s = await getSetupState();
    setSetup(s);
    setRoute(s.ready ? 'node' : 'onboarding');
  }

  const isShell = route !== 'welcome' && route !== 'onboarding';

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <div className="space-bg" />
      <div className="space-stars" />

      <div className="relative z-10 flex h-full flex-col">
        {isShell && <TopBar />}

        <div className="relative min-h-0 flex-1">
          <AnimatePresence mode="wait">
            <motion.div
              key={route}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="absolute inset-0"
            >
              {route === 'welcome' && <Welcome onStart={start} />}
              {route === 'onboarding' && (
                <Onboarding refreshSetup={refreshSetup} onEnter={() => setRoute('node')} />
              )}
              {route === 'node' && <Dashboard />}
              {route === 'chat' && <Chat />}
              {route === 'mesh' && <Mesh />}
              {route === 'models' && (
                <div className="mx-auto h-full w-full max-w-4xl px-8 pb-8 pt-2">
                  <div className="mb-4">
                    <h2 className="text-xl font-semibold text-ink-0">Models</h2>
                    <p className="text-[13px] text-ink-2">
                      Download and switch the model your node runs. Changing it restarts the engine.
                    </p>
                  </div>
                  <div className="h-[calc(100%-4rem)]">
                    <Models onActivated={refreshSetup} />
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
