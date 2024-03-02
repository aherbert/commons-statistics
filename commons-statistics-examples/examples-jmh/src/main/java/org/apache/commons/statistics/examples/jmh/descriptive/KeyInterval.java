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
 * a linear scan of the data. The scan start point is chosen from reference points within the data.
 *
 * <p>The scan is fast when the number of keys is small.
 */
final class KeyInterval implements Interval {
    /** The ordered keys for descending search. */
    private final int[] keys;
    /** Index of the left key. */
    private int l;
    /** Index of the right key. */
    private int r;

    /**
     * Create an instance with the provided keys.
     *
     * @param indices Indices.
     * @param n Number of indices.
     */
    KeyInterval(int[] indices, int n) {
        this(indices, 0, n - 1);
    }

    /**
     * @param indices Indices.
     * @param left Index of left key.
     * @param right Index of right key.
     */
    private KeyInterval(int[] indices, int left, int right) {
        keys = indices;
        l = left;
        r = right;
    }

    @Override
    public int left() {
        return keys[l];
    }

    @Override
    public int right() {
        return keys[r];
    }

    @Override
    public int updateLeft(int k) {
        // Assume left <= k < right (i.e. we must move left at least 1)
        // Search using a scan on the assumption that k is close to the end
        int i = l;
        do {
            ++i;
        } while (keys[i] <= k);
        l = i;
        return left();
    }

    @Override
    public int updateRight(int k) {
        // Assume left < k <= right (i.e. we must move right at least 1)
        // Search using a scan on the assumption that k is close to the end
        int i = r;
        do {
            --i;
        } while (keys[i] >= k);
        r = i;
        return right();
    }

    @Override
    public Interval split(int ka, int kb) {
        // left < ka <= kb < right

        // We could test if ka/kb is above or below the
        // median (keys[l] + keys[r]) >>> 1 to pick the side to search

        // Update the current left bound, save the old one
        final int lower = l;
        int i = Partition.searchGreaterOrEqual(keys, l, r, kb + 1);
        l = i;

        // Find the new right bound for the lower interval using a scan since a
        // typical use case has ka == kb and this is faster than a second binary search.
        do {
            --i;
        } while (keys[i] >= ka);
        return new KeyInterval(keys, lower, i);
    }
}
