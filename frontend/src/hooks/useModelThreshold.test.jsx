import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useModelThreshold } from './useModelThreshold';
import * as api from '../services/api';

vi.mock('../services/api', () => ({
  getModelMetadata: vi.fn(),
}));

describe('useModelThreshold hook', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('1. success first time: loads threshold on immediate successful response', async () => {
    api.getModelMetadata.mockResolvedValueOnce({
      data: { threshold: 0.1541 },
    });

    const { result } = renderHook(() => useModelThreshold());

    // Initially loading
    expect(result.current.loaded).toBe(false);
    expect(result.current.threshold).toBe(null);

    // Wait for microtasks
    await act(async () => {
      await Promise.resolve();
    });

    expect(result.current.loaded).toBe(true);
    expect(result.current.threshold).toBe(0.1541);
  });

  it('2. failure then success (fake timers): retries at 2s interval and succeeds on attempt 2', async () => {
    vi.useFakeTimers();

    api.getModelMetadata
      .mockRejectedValueOnce(new Error('503 Backend unavailable'))
      .mockResolvedValueOnce({ data: { threshold: 0.1541 } });

    const { result } = renderHook(() => useModelThreshold());

    // Initial attempt rejected
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.loaded).toBe(false);
    expect(result.current.threshold).toBe(null);

    // Advance 2 seconds for retry #2
    await act(async () => {
      vi.advanceTimersByTime(2000);
      await Promise.resolve();
    });

    expect(result.current.loaded).toBe(true);
    expect(result.current.threshold).toBe(0.1541);
  });

  it('3. never loads: stays not loaded when backend repeatedly fails', async () => {
    vi.useFakeTimers();

    api.getModelMetadata.mockRejectedValue(new Error('Persistent offline'));

    const { result } = renderHook(() => useModelThreshold());

    // Advance through 35 retry cycles (past the 30-attempt boundary into 10s intervals)
    for (let i = 0; i < 35; i++) {
      await act(async () => {
        vi.advanceTimersByTime(10000);
        await Promise.resolve();
      });
    }

    expect(result.current.loaded).toBe(false);
    expect(result.current.threshold).toBe(null);
  });

  it('4. stream value overrides: stream threshold updates state and overrides prior metadata', async () => {
    api.getModelMetadata.mockResolvedValueOnce({
      data: { threshold: 0.1541 },
    });

    const { result, rerender } = renderHook(
      ({ streamThresh }) => useModelThreshold(streamThresh),
      { initialProps: { streamThresh: null } }
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.threshold).toBe(0.1541);
    expect(result.current.loaded).toBe(true);

    // Stream arrives with a new threshold value (e.g. 0.182)
    rerender({ streamThresh: 0.182 });
    expect(result.current.threshold).toBe(0.182);
    expect(result.current.loaded).toBe(true);

    // Imperative updateFromStream also updates (latest value wins)
    act(() => {
      result.current.updateFromStream(0.195);
    });
    expect(result.current.threshold).toBe(0.195);
  });
});
