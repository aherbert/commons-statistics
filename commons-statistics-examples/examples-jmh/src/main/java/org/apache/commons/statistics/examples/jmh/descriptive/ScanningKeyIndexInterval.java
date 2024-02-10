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
    /** The original number of keys. */
    private final int n;

    /**
     * Create an instance with the provided keys.
     *
     * @param indices Indices.
     * @param n Number of indices.
     */
    ScanningKeyIndexInterval(int[] indices, int n) {
        keys = indices;
        this.n = n;
    }

    /**
     * Initialise an instance with {@code n} initial {@code indices}. The indices are used in place.
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
        return keys[0];
    }

    @Override
    public int right() {
        return keys[n - 1];
    }

    @Override
    public int previousIndex(int k) {
        // Scan the sorted keys from the end.
        // Assume left <= k <= right thus no index checks required.
        // IndexOutOfBoundsException indicates incorrect usage by the caller.
        for (int i = n;;) {
            if (keys[--i] <= k) {
                return keys[i];
            }
        }
    }

    @Override
    public int nextIndex(int k) {
        // Scan the sorted keys from the start.
        // Assume left <= k <= right thus no index checks required.
        // IndexOutOfBoundsException indicates incorrect usage by the caller.
        for (int i = -1;;) {
            if (keys[++i] >= k) {
                return keys[i];
            }
        }
    }
}
