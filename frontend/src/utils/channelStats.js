/**
 * Utilities for computing statistics and smoothed errors across a channel window.
 */

/**
 * Compute causal trailing moving average of errors with window k.
 *
 * @param {number[]} errors - array of reconstruction errors
 * @param {number} k - trailing window size (default 5)
 * @returns {number[]} smoothed errors
 */
export function calculateCausalSmoothedErrors(errors = [], k = 5) {
  if (!Array.isArray(errors) || errors.length === 0) return [];
  const safeK = Math.max(1, k);
  return errors.map((_, i, arr) => {
    const start = Math.max(0, i - safeK + 1);
    const slice = arr.slice(start, i + 1);
    const sum = slice.reduce((acc, v) => acc + (typeof v === 'number' ? v : 0), 0);
    return sum / slice.length;
  });
}

/**
 * Compute window summary statistics for a sensor channel.
 *
 * @param {number[]} values - array of sensor values
 * @param {number[]} errors - array of reconstruction errors
 * @param {number} threshold - anomaly cutoff threshold
 * @returns {object} statistics summary
 */
export function calculateWindowStats(values = [], errors = [], threshold) {
  if (!Array.isArray(values) || values.length === 0) {
    return { min: null, max: null, mean: null, std: null, anomalies: 0, count: 0 };
  }

  const numericVals = values.filter((v) => typeof v === 'number' && !Number.isNaN(v));
  if (numericVals.length === 0) {
    return { min: null, max: null, mean: null, std: null, anomalies: 0, count: 0 };
  }

  const min = Math.min(...numericVals);
  const max = Math.max(...numericVals);
  const mean = numericVals.reduce((a, b) => a + b, 0) / numericVals.length;
  const variance = numericVals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / numericVals.length;
  const std = Math.sqrt(variance);

  const safeErrors = Array.isArray(errors) ? errors : [];
  const hasThreshold = typeof threshold === 'number' && Number.isFinite(threshold);
  const anomalies = hasThreshold
    ? safeErrors.filter((e) => typeof e === 'number' && e > threshold).length
    : 0;

  return {
    min,
    max,
    mean,
    std,
    anomalies,
    count: numericVals.length,
  };
}
