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
 * An {@link UpdatingInterval} backed by an array of ordered keys.
 */
final class KeyUpdatingInterval implements UpdatingInterval {
    /** Size to use a scan of the keys when splitting instead of binary search. */
    private static final int SCAN_SIZE = 32;

    /** The ordered keys. */
    private final int[] keys;
    /** Index of the left key. */
    private int l;
    /** Index of the right key. */
    private int r;
    /** Left key. Always equal to keys[l]. This is cached here to avoid cache misses
     * when splitting an interval as the keys may no longer be in cache memory.
     * This is true for a large keys[] array for the lower side of the split (the
     * search has ended near the splitting index), and for the upper side of the split
     * which may be used a long time after the split (since the partition algorithm
     * uses left-most precedence when processing data). */
    private int leftKey;

    /**
     * Create an instance with the provided {@code indices}.
     *
     * @param indices Indices.
     * @param n Number of indices.
     */
    private KeyUpdatingInterval(int[] indices, int n) {
        this(indices, 0, n - 1, indices[0]);
    }

    /**
     * @param indices Indices.
     * @param l Index of left key.
     * @param r Index of right key.
     * @param leftKey Left key (must be equal to indices[l]).
     */
    private KeyUpdatingInterval(int[] indices, int l, int r, int leftKey) {
        keys = indices;
        this.l = l;
        this.r = r;
        this.leftKey = leftKey;
    }

    /**
     * Initialise an instance with the {@code indices}. The indices are used in place.
     *
     * @param indices Indices.
     * @param n Number of indices.
     * @return the interval
     * @throws IllegalArgumentException if the indices are not unique and ordered;
     * or {@code n <= 0}
     */
    static KeyUpdatingInterval of(int[] indices, int n) {
        // Check the indices are uniquely ordered
        if (n <= 0) {
            throw new IllegalArgumentException("No indices to define the range");
        }
        int p = indices[0];
        for (int i = 0; ++i < n;) {
            final int c = indices[i];
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
        return new KeyUpdatingInterval(indices, n);
    }

    @Override
    public int left() {
        // Cache key. Useful when the left is used a long time after a split
        // and keys may not be in cache memory.
        return leftKey;
    }

    @Override
    public int right() {
        return keys[r];
    }

    @Override
    public int updateLeft(int k) {
        // Assume left < k <= right (i.e. we must move left at least 1)
        // Search using a scan on the assumption that k is close to the end
        int i = l;
        do {
            ++i;
        } while (keys[i] < k);
        setLeft(i);
        return leftKey;
    }

    @Override
    public int updateRight(int k) {
        // Assume left <= k < right (i.e. we must move right at least 1)
        // Search using a scan on the assumption that k is close to the end
        int i = r;
        do {
            --i;
        } while (keys[i] > k);
        r = i;
        return right();
    }

    @Override
    public UpdatingInterval split(int ka, int kb) {
        // left < ka <= kb < right

        // Update the current left bound, save the old one
        final int lower = l;
        final int lowerKey = leftKey;

        // Find the new left bound for the upper interval.
        // Switch to a linear scan if (r - l) is small.
        int i;
        if (r - l < SCAN_SIZE) {
            i = r;
            do {
                --i;
            } while (keys[i] > kb);
        } else {
            // Binary search
            i = Partition.searchLessOrEqual(keys, l, r, kb);
        }
        setLeft(i + 1);

        // Find the new right bound for the lower interval using a scan since a
        // typical use case has ka == kb and this is faster than a second binary search.
        while (keys[i] >= ka) {
            --i;
        }
        return new KeyUpdatingInterval(keys, lower, i, lowerKey);
    }

    /**
     * Sets the left index and update the cache of the left key.
     *
     * @param i Left index.
     */
    private void setLeft(int i) {
        l = i;
        // Cache the key value
        leftKey = keys[i];
    }
}
