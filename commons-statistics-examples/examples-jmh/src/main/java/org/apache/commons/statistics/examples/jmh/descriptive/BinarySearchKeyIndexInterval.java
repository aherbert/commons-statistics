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
 * a binary search.
 */
final class BinarySearchKeyIndexInterval implements IndexInterval {
    /** The ordered keys for descending search. */
    private final int[] keys;
    /** The original number of keys - 1. This is more convenient to store for the use cases. */
    private final int nm1;

    /**
     * Create an instance with the provided keys.
     *
     * @param indices Indices.
     * @param n Number of indices.
     */
    BinarySearchKeyIndexInterval(int[] indices, int n) {
        nm1 = n - 1;
        keys = indices;
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
    static BinarySearchKeyIndexInterval of(int[] indices, int n) {
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
        return new BinarySearchKeyIndexInterval(indices, n);
    }

    @Override
    public int left() {
        return keys[0];
    }

    @Override
    public int right() {
        return keys[nm1];
    }

    @Override
    public int previousIndex(int k) {
        // Assume left <= k <= right thus no index checks required.
        // IndexOutOfBoundsException indicates incorrect usage by the caller.
        return keys[Partition.searchLessOrEqual(keys, 0, nm1, k)];
    }

    @Override
    public int nextIndex(int k) {
        // Assume left <= k <= right thus no index checks required.
        // IndexOutOfBoundsException indicates incorrect usage by the caller.
        return keys[Partition.searchGreaterOrEqual(keys, 0, nm1, k)];
    }

    @Override
    public int splitLower(int k, int[] upper) {
        int i = Partition.searchLessOrEqual(keys, 0, nm1, k - 1);
        final int lower = keys[i];
        // Find the upper.
        // Note: Should never be called when n == 1 as it requires left < k <= right.
        // We know the keys are ascending and unique so check the neighbour.
        if (keys[++i] > k) {
            // Keys (i, i+1) are non-consecutive and k is between them.
            upper[0] = keys[i];
        } else {
            // Either keys are consecutive, or k == right()
            upper[0] = i < nm1 ? keys[++i] : right() + 1;
        }
        return lower;
    }

    @Override
    public int splitUpper(int k, int[] lower) {
        int i = Partition.searchGreaterOrEqual(keys, 0, nm1, k + 1);
        final int upper = keys[i];
        // Find the lower.
        // Note: Should never be called when n == 1 as it requires left <= k < right.
        // We know the keys are ascending and unique so check the neighbour.
        if (keys[--i] < k) {
            // Keys (i, i+1) are non-consecutive and k is between them.
            lower[0] = keys[i];
        } else {
            // Either keys are consecutive, or k == left()
            lower[0] = i > 0 ? keys[--i] : left() - 1;
        }
        return upper;
    }
}
