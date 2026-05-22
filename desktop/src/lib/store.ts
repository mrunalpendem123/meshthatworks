import { create } from 'zustand';
import type { EngineStatus, SetupState } from './types';

export type Route = 'welcome' | 'onboarding' | 'node' | 'chat' | 'mesh' | 'models';

interface AppState {
  route: Route;
  setRoute: (r: Route) => void;

  engine: EngineStatus;
  setEngine: (e: EngineStatus) => void;

  setup: SetupState | null;
  setSetup: (s: SetupState) => void;

  /** Welcome splash has been dismissed at least once this session. */
  entered: boolean;
  setEntered: (v: boolean) => void;
}

export const useApp = create<AppState>((set) => ({
  route: 'welcome',
  setRoute: (route) => set({ route }),

  engine: { phase: 'stopped', healthy: false, message: 'Engine offline', status: null },
  setEngine: (engine) => set({ engine }),

  setup: null,
  setSetup: (setup) => set({ setup }),

  entered: false,
  setEntered: (entered) => set({ entered }),
}));
