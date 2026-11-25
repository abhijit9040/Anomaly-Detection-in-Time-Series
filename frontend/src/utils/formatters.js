/**
 * Number and delta formatting utilities.
 */

/**
 * Format a delta value with a strict sign (+ or -), never "+-".
 *
 * @param {number|null|undefined} value
 * @param {string} unit - optional unit (e.g. "pp", "%")
 * @param {number} decimals - decimal places (default 1)
 * @returns {string} - formatted signed string (e.g. "+3.2 pp", "-0.8%", "0.0")
 */
export function formatDelta(value, unit = '', decimals = 1) {
  if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) {
    return 'N/A';
  }

  const num = Number(value);
  // Avoid negative zero
  const isZero = Math.abs(num) < 1e-9;
  const fixed = isZero ? (0).toFixed(decimals) : num.toFixed(decimals);
  const sign = num > 0 ? '+' : '';

  if (!unit) {
    return `${sign}${fixed}`;
  }

  if (unit === '%') {
    return `${sign}${fixed}%`;
  }

  return `${sign}${fixed} ${unit}`;
}

/**
 * Get badge styling based on percentage point delta:
 * - Green if > 0
 * - Amber if between -2.0 and 0 pp
 * - Red if worse than -2.0 pp
 *
 * @param {number} deltaPp
 * @returns {object} style object
 */
export function getDeltaBadgeStyle(deltaPp) {
  if (deltaPp > 0) {
    return {
      background: 'rgba(16, 185, 129, 0.15)',
      color: '#10b981',
      border: '1px solid rgba(16, 185, 129, 0.3)',
      padding: '5px 12px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: 600,
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
    };
  }

  if (deltaPp >= -2.0) {
    return {
      background: 'rgba(245, 158, 11, 0.15)',
      color: '#f59e0b',
      border: '1px solid rgba(245, 158, 11, 0.3)',
      padding: '5px 12px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: 600,
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
    };
  }

  return {
    background: 'rgba(239, 68, 68, 0.15)',
    color: '#ef4444',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    padding: '5px 12px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: 600,
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
  };
}
