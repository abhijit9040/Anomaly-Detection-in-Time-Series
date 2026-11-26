import { describe, it, expect, vi, beforeEach } from 'vitest';

// Provide a mock localStorage for node test runner
const storage = {};
globalThis.localStorage = {
  getItem: (key) => storage[key] ?? null,
  setItem: (key, val) => { storage[key] = String(val); },
  removeItem: (key) => { delete storage[key]; },
  clear: () => { Object.keys(storage).forEach(k => delete storage[k]); }
};

import { playAnomalyAlert, testAudioAlert, unlockAudio } from './audioAlert';

describe('audioAlert', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('playAnomalyAlert does nothing if setting_sound_alerts is not true', () => {
    localStorage.setItem('setting_sound_alerts', 'false');
    expect(() => playAnomalyAlert()).not.toThrow();
  });

  it('unlockAudio and testAudioAlert execute safely in browser/test environment', () => {
    expect(() => unlockAudio()).not.toThrow();
    expect(() => testAudioAlert()).not.toThrow();
  });

  it('playAnomalyAlert executes safely when setting is true', () => {
    localStorage.setItem('setting_sound_alerts', 'true');
    expect(() => playAnomalyAlert()).not.toThrow();
  });
});
