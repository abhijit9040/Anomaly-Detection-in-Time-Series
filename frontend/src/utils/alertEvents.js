/**
 * Alert de-duplication.
 * One real anomaly lasts hundreds of consecutive stream points. We want ONE alert
 * per anomaly event, not one per anomalous point.
 * An event starts on the first anomalous point after a quiet period and ends only
 * after `cooldown` consecutive normal points, so a short dip inside the same event
 * does not create a second alert.
 */
export const ALERT_COOLDOWN = 10; // consecutive normal points that close an event

export const initialAlertState = { active: false, quiet: 0 };

export function advanceAlertState(state, isAnomaly, cooldown = ALERT_COOLDOWN) {
  if (isAnomaly) {
    return { state: { active: true, quiet: 0 }, isNewEvent: !state.active };
  }
  if (!state.active) {
    return { state, isNewEvent: false };
  }
  const quiet = state.quiet + 1;
  return {
    state: quiet >= cooldown ? initialAlertState : { active: true, quiet },
    isNewEvent: false,
  };
}

/**
 * Advance alert state based on post-processed alert_active signal.
 * Produces a new event ONLY on a rising edge (inactive -> active).
 * If alert_active is null or undefined, falls back to raw-flag logic.
 *
 * @param {boolean|object} prevActive - Whether the alert was previously active
 * @param {boolean|null|undefined} alertActive - Post-processed alert signal from model
 * @param {boolean} [rawIsAnomaly=false] - Optional raw is_anomaly fallback flag
 * @returns {{ isNewEvent: boolean, active: boolean }}
 */
export function advanceFromAlertActive(prevActive, alertActive, rawIsAnomaly = false) {
  const wasActive = typeof prevActive === 'object' && prevActive !== null
    ? Boolean(prevActive.active)
    : Boolean(prevActive);

  if (alertActive === null || alertActive === undefined) {
    const isAnomaly = Boolean(rawIsAnomaly);
    return {
      isNewEvent: !wasActive && isAnomaly,
      active: isAnomaly,
    };
  }

  const currentActive = Boolean(alertActive);
  return {
    isNewEvent: !wasActive && currentActive,
    active: currentActive,
  };
}
