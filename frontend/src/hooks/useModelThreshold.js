import { useState, useEffect, useRef, useCallback } from 'react';
import { getModelMetadata } from '../services/api';

/**
 * Custom hook to manage the model threshold as a single reliable source of truth.
 *
 * Requirements:
 * - Keeps `threshold` in React state (not a ref).
 * - Fetches /api/v1/model with retry:
 *   - Every 2 seconds for up to 30 attempts, then every 10 seconds until a finite number is returned.
 * - Updates from the `threshold` field of each stream response (latest value wins).
 * - Returns { threshold, loaded, updateFromStream }
 */
export function useModelThreshold(streamThreshold) {
  const [threshold, setThreshold] = useState(null);
  const [loaded, setLoaded] = useState(false);

  const attemptCountRef = useRef(0);
  const timerRef = useRef(null);
  const loadedRef = useRef(false);

  // Method to imperatively push a stream threshold update
  const updateFromStream = useCallback((val) => {
    if (typeof val === 'number' && Number.isFinite(val)) {
      setThreshold(val);
      setLoaded(true);
      loadedRef.current = true;
    }
  }, []);

  // Sync when streamThreshold is passed as an argument
  useEffect(() => {
    if (typeof streamThreshold === 'number' && Number.isFinite(streamThreshold)) {
      setThreshold(streamThreshold);
      setLoaded(true);
      loadedRef.current = true;
    }
  }, [streamThreshold]);

  useEffect(() => {
    let isCancelled = false;

    async function pollModelMetadata() {
      if (isCancelled || loadedRef.current) return;

      attemptCountRef.current += 1;

      try {
        const res = await getModelMetadata();
        const val = res?.data?.threshold;

        if (typeof val === 'number' && Number.isFinite(val)) {
          if (!isCancelled) {
            setThreshold(val);
            setLoaded(true);
            loadedRef.current = true;
          }
          return;
        }
      } catch {
        // Failed or returned non-finite value; proceed to retry
      }

      if (isCancelled || loadedRef.current) return;

      // Retry rule: every 2s for up to 30 tries, then every 10s
      const delay = attemptCountRef.current < 30 ? 2000 : 10000;
      timerRef.current = setTimeout(pollModelMetadata, delay);
    }

    pollModelMetadata();

    return () => {
      isCancelled = true;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  return { threshold, loaded, updateFromStream };
}

export default useModelThreshold;
