import { describe, it, expect } from 'vitest';
import { advanceAlertState, initialAlertState, ALERT_COOLDOWN } from './alertEvents';

function runSequence(sequence, cooldown = ALERT_COOLDOWN) {
  let state = initialAlertState;
  let eventCount = 0;
  for (const isAnomaly of sequence) {
    const step = advanceAlertState(state, isAnomaly, cooldown);
    state = step.state;
    if (step.isNewEvent) {
      eventCount += 1;
    }
  }
  return { eventCount, finalState: state };
}

describe('advanceAlertState', () => {
  it('(a) 5 normal + 300 anomalous + 20 normal produces exactly 1 new event', () => {
    const seq = [
      ...Array(5).fill(false),
      ...Array(300).fill(true),
      ...Array(20).fill(false),
    ];
    const { eventCount, finalState } = runSequence(seq);
    expect(eventCount).toBe(1);
    expect(finalState.active).toBe(false);
    expect(finalState.quiet).toBe(0);
  });

  it('(b) 20 anomalous, 3 normal, 20 anomalous produces exactly 1 event (short dip <= 10 does not split)', () => {
    const seq = [
      ...Array(20).fill(true),
      ...Array(3).fill(false),
      ...Array(20).fill(true),
    ];
    const { eventCount, finalState } = runSequence(seq);
    expect(eventCount).toBe(1);
    expect(finalState.active).toBe(true);
    expect(finalState.quiet).toBe(0);
  });

  it('(c) 5 anomalous, 30 normal, 5 anomalous produces exactly 2 events', () => {
    const seq = [
      ...Array(5).fill(true),
      ...Array(30).fill(false),
      ...Array(5).fill(true),
    ];
    const { eventCount, finalState } = runSequence(seq);
    expect(eventCount).toBe(2);
    expect(finalState.active).toBe(true);
  });

  it('(d) 100 normal points produces 0 events', () => {
    const seq = Array(100).fill(false);
    const { eventCount, finalState } = runSequence(seq);
    expect(eventCount).toBe(0);
    expect(finalState.active).toBe(false);
  });

  it('(e) Cooldown boundary: 9 normal points keeps event open; 10th closes it', () => {
    // 1 anomaly opens event
    let step = advanceAlertState(initialAlertState, true);
    expect(step.isNewEvent).toBe(true);
    expect(step.state.active).toBe(true);

    // 9 consecutive normal points: event remains active
    for (let i = 1; i <= 9; i++) {
      step = advanceAlertState(step.state, false);
      expect(step.isNewEvent).toBe(false);
      expect(step.state.active).toBe(true);
      expect(step.state.quiet).toBe(i);
    }

    // Still active at 9 normal points: a new anomaly does NOT create a new event
    const continuationStep = advanceAlertState(step.state, true);
    expect(continuationStep.isNewEvent).toBe(false);
    expect(continuationStep.state.active).toBe(true);
    expect(continuationStep.state.quiet).toBe(0);

    // Now test 10th normal point: closes the event
    // Re-run to 9 normal points
    let closeTestState = { active: true, quiet: 9 };
    const closeStep = advanceAlertState(closeTestState, false);
    expect(closeStep.isNewEvent).toBe(false);
    expect(closeStep.state.active).toBe(false);
    expect(closeStep.state.quiet).toBe(0);

    // After 10 normal points, next anomaly DOES create a new event
    const reopenStep = advanceAlertState(closeStep.state, true);
    expect(reopenStep.isNewEvent).toBe(true);
    expect(reopenStep.state.active).toBe(true);
  });
});
