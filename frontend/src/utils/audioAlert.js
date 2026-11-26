/**
 * Audio alert utility — High-Tech Industrial Anomaly Alert Synthesizer.
 *
 * Uses the Web Audio API (zero external assets, zero latency) with multiple
 * sound profiles designed specifically for anomaly detection & telemetry monitoring.
 *
 * Available profiles:
 * - 'industrial': (Default) Dual-pulse resonant warning bursts (SCADA / IoT anomaly signature).
 * - 'siren': Alternating high-urgency two-phase alarm warble.
 * - 'sonar': High-tech telemetry sonar ping with resonant decay.
 * - 'chime': Soft two-tone notification chime.
 */

let audioCtx = null;

function getAudioContext() {
  if (typeof window === 'undefined') return null;
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  return audioCtx;
}

/**
 * Call from any user interaction to unlock the Web Audio context.
 */
export function unlockAudio() {
  const ctx = getAudioContext();
  if (ctx && ctx.state === 'suspended') {
    ctx.resume().catch(() => {});
  }
}

// Auto-register window listener for first click/touch/keypress to unlock audio context
if (typeof window !== 'undefined') {
  const autoUnlock = () => {
    unlockAudio();
    window.removeEventListener('click', autoUnlock);
    window.removeEventListener('keydown', autoUnlock);
    window.removeEventListener('touchstart', autoUnlock);
  };
  window.addEventListener('click', autoUnlock, { once: true, passive: true });
  window.addEventListener('keydown', autoUnlock, { once: true, passive: true });
  window.addEventListener('touchstart', autoUnlock, { once: true, passive: true });
}

// ── Profile Synthesizers ─────────────────────────────────────────────────────

/**
 * Industrial Anomaly Pulse: Dual rapid-burst frequency sweep with analog resonance.
 * Sounds authoritative and clearly signals an anomaly.
 */
function playIndustrialPulse(ctx, masterGain) {
  const now = ctx.currentTime;

  const playBurst = (startTime) => {
    // Resonant lowpass filter for analog hardware warmth
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(2600, startTime);
    filter.Q.setValueAtTime(4.5, startTime);
    filter.connect(masterGain);

    // Primary alert voice (triangle wave with pitch drop)
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(1080, startTime);
    osc.frequency.exponentialRampToValueAtTime(740, startTime + 0.085);

    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(1, startTime + 0.006);
    gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.095);

    osc.connect(gain);
    gain.connect(filter);
    osc.start(startTime);
    osc.stop(startTime + 0.1);

    // Sub-harmonic oscillator for punchy weight
    const sub = ctx.createOscillator();
    const subGain = ctx.createGain();
    sub.type = 'sine';
    sub.frequency.setValueAtTime(370, startTime);

    subGain.gain.setValueAtTime(0, startTime);
    subGain.gain.linearRampToValueAtTime(0.6, startTime + 0.006);
    subGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.095);

    sub.connect(subGain);
    subGain.connect(masterGain);
    sub.start(startTime);
    sub.stop(startTime + 0.1);
  };

  // Burst 1
  playBurst(now);
  // Burst 2 (delayed 125ms)
  playBurst(now + 0.125);
}

/**
 * Critical Siren: Rapid two-step urgent warning warble.
 */
function playCriticalSiren(ctx, masterGain) {
  const now = ctx.currentTime;
  const filter = ctx.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.setValueAtTime(2200, now);
  filter.connect(masterGain);

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sawtooth';

  // Rapid 3-cycle warble (1050 Hz <-> 820 Hz)
  osc.frequency.setValueAtTime(1050, now);
  osc.frequency.setValueAtTime(820, now + 0.08);
  osc.frequency.setValueAtTime(1050, now + 0.16);
  osc.frequency.setValueAtTime(820, now + 0.24);

  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(0.8, now + 0.01);
  gain.gain.setValueAtTime(0.8, now + 0.3);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.36);

  osc.connect(gain);
  gain.connect(filter);
  osc.start(now);
  osc.stop(now + 0.38);
}

/**
 * Sonar Ping: High-tech aerospace telemetry ping with long resonant decay.
 */
function playSonarPing(ctx, masterGain) {
  const now = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'sine';
  osc.frequency.setValueAtTime(1280, now);
  osc.frequency.exponentialRampToValueAtTime(860, now + 0.35);

  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(1, now + 0.008);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

  osc.connect(gain);
  gain.connect(masterGain);
  osc.start(now);
  osc.stop(now + 0.46);
}

/**
 * Soft Chime: Gentle two-tone sine notification.
 */
function playSoftChime(ctx, masterGain) {
  const now = ctx.currentTime;

  // Tone 1: 880 Hz
  const osc1 = ctx.createOscillator();
  const gain1 = ctx.createGain();
  osc1.type = 'sine';
  osc1.frequency.setValueAtTime(880, now);
  gain1.gain.setValueAtTime(0, now);
  gain1.gain.linearRampToValueAtTime(1, now + 0.015);
  gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
  osc1.connect(gain1);
  gain1.connect(masterGain);
  osc1.start(now);
  osc1.stop(now + 0.19);

  // Tone 2: 659.25 Hz
  const osc2 = ctx.createOscillator();
  const gain2 = ctx.createGain();
  osc2.type = 'sine';
  osc2.frequency.setValueAtTime(659.25, now + 0.12);
  gain2.gain.setValueAtTime(0, now + 0.12);
  gain2.gain.linearRampToValueAtTime(0.85, now + 0.135);
  gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
  osc2.connect(gain2);
  gain2.connect(masterGain);
  osc2.start(now + 0.12);
  osc2.stop(now + 0.36);
}

/**
 * Play a specific sound profile.
 */
function synthesizeAlert(style = 'industrial', volume = 0.35) {
  const ctx = getAudioContext();
  if (!ctx) return;

  const run = () => {
    try {
      const now = ctx.currentTime;
      const masterGain = ctx.createGain();
      masterGain.gain.setValueAtTime(volume, now);
      masterGain.connect(ctx.destination);

      switch (style) {
        case 'siren':
          playCriticalSiren(ctx, masterGain);
          break;
        case 'sonar':
          playSonarPing(ctx, masterGain);
          break;
        case 'chime':
          playSoftChime(ctx, masterGain);
          break;
        case 'industrial':
        default:
          playIndustrialPulse(ctx, masterGain);
          break;
      }
    } catch (err) {
      console.warn('Audio alert playback error:', err);
    }
  };

  if (ctx.state === 'suspended') {
    ctx.resume().then(run).catch(() => {});
  } else {
    run();
  }
}

/**
 * Play alert if setting_sound_alerts is enabled in localStorage.
 * Automatically uses the sound style saved in localStorage ('setting_sound_style').
 */
export function playAnomalyAlert(volume = 0.35) {
  if (typeof window === 'undefined' || typeof localStorage === 'undefined') return;
  const setting = localStorage.getItem('setting_sound_alerts');
  if (setting !== 'true') return;

  const style = localStorage.getItem('setting_sound_style') || 'industrial';
  synthesizeAlert(style, volume);
}

/**
 * Test chime with chosen sound style (bypasses preference check for testing in Settings).
 */
export function testAudioAlert(style = null, volume = 0.35) {
  unlockAudio();
  const chosenStyle = style || (typeof localStorage !== 'undefined' ? localStorage.getItem('setting_sound_style') : null) || 'industrial';
  synthesizeAlert(chosenStyle, volume);
}
