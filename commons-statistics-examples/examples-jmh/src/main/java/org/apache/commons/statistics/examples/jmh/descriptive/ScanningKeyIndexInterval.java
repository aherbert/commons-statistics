/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.statistics.examples.jmh.descriptive;

/**
 * An {@link IndexInterval} backed by an array of ordered keys. The interval is searched using
 * a linear scan from either end.
 *
 * <p>The scan is fast when the number of keys is small.
 */
final class ScanningKeyIndexInterval implements IndexInterval {
    /** The ordered keys. */
    private final int[] keys;
    /** The original number of keys + 1. This is more convenient to store for the use cases. */
    private final int np1;

    /**
     * Create an instance with the provided keys.
     *
     * @param indices Indices.
     * @param n Number of indices.
     */
    ScanningKeyIndexInterval(int[] indices, int n) {
        np1 = n + 1;
        // Copy indices with room for sentinal values.
        keys = new int[n + 2];
        System.arraycopy(indices, 0, keys, 1, n);
        // Set sentinal values
        keys[0] = -1;
        keys[np1] = Integer.MAX_VALUE;
    }

    /**
     * Initialise an instance with the {@code indices}. The indices are copied.
     *
     * <p>This will error if a memory allocation of an integer array of size {@code n + 2}
     * is invalid. In practice this should be used with a small number of indices.
     *
     * @param indices Indices.
     * @param n Number of indices.
     * @return the interval
     * @throws IllegalArgumentException if the indices are not unique and ordered; or not
     * in the range {@code [0, 2^31-1)}; or {@code n <= 0}
     */
    static ScanningKeyIndexInterval of(int[] indices, int n) {
        // Check the indices are uniquely ordered
        if (n <= 0) {
            throw new IllegalArgumentException("No indices to define the range");
        }
        int p = indices[0];
        for (int i = 0; ++i < n;) {
            int c = indices[i];
            if (c <= p) {
                throw new IllegalArgumentException("Indices are not unique and ordered");
            }
            p = c;
        }
        if (indices[0] < 0) {
            throw new IllegalArgumentException("Unsupported min value: " + indices[0]);
        }
        if (indices[n - 1] == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Unsupported max value: " + Integer.MAX_VALUE);
        }
        return new ScanningKeyIndexInterval(indices, n);
    }

    @Override
    public int left() {
        return keys[1];
    }

    @Override
    public int right() {
        return keys[np1 - 1];
    }

    @Override
    public int previousIndex(int k) {
        // Scan the sorted keys from the end.
        // No index checks required as the key (assumed to be a positive index)
        // cannot be less than the sentinal value -1.
        // IndexOutOfBoundsException indicates incorrect usage by the caller which should
        // call using a valid positive index k in [0, 2^31-1).
        for (int i = np1;;) {
            if (keys[--i] <= k) {
                return keys[i];
            }
        }
    }

    @Override
    public int nextIndex(int k) {
        // Scan the sorted keys from the start.
        // No index checks required as the key (assumed to be a positive index)
        // cannot be greater than the sentinal value MAX_VALUE. It could
        // be equal to it but then k would not be a valid index into an array.
        // IndexOutOfBoundsException indicates incorrect usage by the caller which should
        // call using a valid positive index k in [0, 2^31-1).
        for (int i = 0;;) {
            if (keys[++i] >= k) {
                return keys[i];
            }
        }
    }
}
