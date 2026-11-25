import { describe, it, expect } from 'vitest';
import { formatDelta, getDeltaBadgeStyle } from './formatters';

describe('formatDelta', () => {
  it('formats positive values with plus sign', () => {
    expect(formatDelta(3.2)).toBe('+3.2');
    expect(formatDelta(3.245, 'pp', 2)).toBe('+3.25 pp');
    expect(formatDelta(15.6, '%')).toBe('+15.6%');
  });

  it('formats negative values with minus sign (never +-) ', () => {
    expect(formatDelta(-0.8)).toBe('-0.8');
    expect(formatDelta(-1.33, '%', 2)).toBe('-1.33%');
    expect(formatDelta(-0.79, 'pp', 2)).toBe('-0.79 pp');
    expect(formatDelta(-0.8)).not.toContain('+-');
  });

  it('formats zero cleanly without sign or negative zero', () => {
    expect(formatDelta(0)).toBe('0.0');
    expect(formatDelta(0, 'pp')).toBe('0.0 pp');
    expect(formatDelta(-0.0)).toBe('0.0');
  });

  it('handles null, undefined, empty, or NaN gracefully', () => {
    expect(formatDelta(null)).toBe('N/A');
    expect(formatDelta(undefined)).toBe('N/A');
    expect(formatDelta('')).toBe('N/A');
    expect(formatDelta(NaN)).toBe('N/A');
  });
});

describe('getDeltaBadgeStyle', () => {
  it('returns green for positive delta', () => {
    const style = getDeltaBadgeStyle(1.5);
    expect(style.color).toBe('#10b981');
  });

  it('returns amber for delta between -2.0 and 0 pp', () => {
    const style = getDeltaBadgeStyle(-0.79);
    expect(style.color).toBe('#f59e0b');
    const styleZero = getDeltaBadgeStyle(0);
    expect(styleZero.color).toBe('#f59e0b');
  });

  it('returns red for delta worse than -2.0 pp', () => {
    const style = getDeltaBadgeStyle(-2.5);
    expect(style.color).toBe('#ef4444');
  });
});
