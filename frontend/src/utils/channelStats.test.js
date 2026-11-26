import { describe, it, expect } from 'vitest';
import { calculateCausalSmoothedErrors, calculateWindowStats } from './channelStats';

describe('calculateCausalSmoothedErrors', () => {
  it('returns empty array for empty inputs', () => {
    expect(calculateCausalSmoothedErrors([])).toEqual([]);
    expect(calculateCausalSmoothedErrors(null)).toEqual([]);
  });

  it('computes causal trailing moving average with window k=5', () => {
    const errors = [1, 2, 3, 4, 5, 6];
    const smoothed = calculateCausalSmoothedErrors(errors, 5);
    expect(smoothed[0]).toBe(1); // [1] -> 1
    expect(smoothed[1]).toBe(1.5); // [1, 2] -> 1.5
    expect(smoothed[2]).toBe(2); // [1, 2, 3] -> 2
    expect(smoothed[3]).toBe(2.5); // [1, 2, 3, 4] -> 2.5
    expect(smoothed[4]).toBe(3); // [1, 2, 3, 4, 5] -> 3
    expect(smoothed[5]).toBe(4); // [2, 3, 4, 5, 6] -> 4
  });

  it('handles window k=1 correctly', () => {
    const errors = [0.1, 0.4, 0.8];
    const smoothed = calculateCausalSmoothedErrors(errors, 1);
    expect(smoothed).toEqual([0.1, 0.4, 0.8]);
  });
});

describe('calculateWindowStats', () => {
  it('handles empty inputs gracefully', () => {
    const res = calculateWindowStats([], []);
    expect(res.min).toBeNull();
    expect(res.max).toBeNull();
    expect(res.mean).toBeNull();
    expect(res.std).toBeNull();
    expect(res.anomalies).toBe(0);
  });

  it('computes correct min, max, mean, std, and anomaly counts', () => {
    const values = [10, 20, 30, 40, 50];
    const errors = [0.1, 0.6, 0.2, 0.8, 0.3];
    const threshold = 0.5;

    const stats = calculateWindowStats(values, errors, threshold);
    expect(stats.min).toBe(10);
    expect(stats.max).toBe(50);
    expect(stats.mean).toBe(30);
    expect(stats.std).toBeCloseTo(14.142, 2);
    expect(stats.anomalies).toBe(2); // 0.6 and 0.8
    expect(stats.count).toBe(5);
  });
});
