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
 * Support for creating {@link IndexInterval} implementations.
 *
 * @since 1.1
 */
final class IndexIntervals {
    /** No instances. */
    private IndexIntervals() {}

    /**
     * Returns an interval that covers all indices.
     *
     * @return the interval
     */
    static IndexInterval anyIndex() {
        return AnyIndex.INSTANCE;
    }

    /**
     * Returns an interval that covers the specified indices {@code k}.
     *
     * @param k Indices.
     * @param n Count of indices (must be strictly positive).
     * @return the interval
     */
    static IndexInterval create(int[] k, int n) {
        // Here we use a simple test based on the number of comparisons required
        // to perform the expected next/previous look-ups.
        // It is expected that we can cut n keys a maximum of n-1 times.
        // Each cut requires a scan next/previous to divide the interval into two intervals:
        //
        //            cut
        //             |
        //        k1--------k2---------k3---- ... ---------kn    initial interval
        //         <--| find previous
        //    find next |-->
        //        k1        k2---------k3---- ... ---------kn    divided intervals
        //
        // A ScanningKeyIndexInterval will scan n keys in both directions using n comparisons
        // (if next takes m comparisons then previous will take n - m comparisons): Order(n^2)
        // An IndexSet will scan from the cut location and find a match proportional to
        // the index density. Average density is (size / n) and the scanning covers 64
        // indices together: Order(2 * (size / n) / 64) = Order((size / n) / 32)

        // Get the range. This will throw an exception if there are no indices.
        int min = k[n - 1];
        int max = min;
        for (int i = n - 1; --i >= 0;) {
            min = Math.min(min, k[i]);
            max = Math.max(max, k[i]);
        }

        // Use (max - min) as an approximate size.
        // n * n < (size / n) / 32
        // n * n * n < size / 32
        // Transition point: n = (size/32)^(1/3)
        // size  n
        // 2^5   3.175
        // 2^10  10.08
        // 2^15  32.00
        // 2^20  101.6
        // 2^25  322.6
        // 2^30  1024
        // TODO: run benchmark around these transition points.
        // Also using a BinarySearchIndexInterval to see if this is worth supporting.
        if ((double) n * n * n < ((max - min) >> 5)) {
            final int unique = Sorting.sortIndices2(k, n);
            return ScanningKeyIndexInterval.of(k, unique);
        }

        // We know the required min/max here so do not call IndexSet.of(k, n)
        final IndexSet interval = IndexSet.ofRange(min, max);
        for (int i = n; --i >= 0;) {
            interval.set(k[i]);
        }
        return interval;
    }

    /**
     * {@link IndexInterval} for range {@code [0, MAX_VALUE]}.
     */
    private static final class AnyIndex implements IndexInterval {
        /** Singleton instance. */
        private static final AnyIndex INSTANCE = new AnyIndex();

        @Override
        public int left() {
            return 0;
        }

        @Override
        public int right() {
            return Integer.MAX_VALUE;
        }

        @Override
        public int previousIndex(int k) {
            return k;
        }

        @Override
        public int nextIndex(int k) {
            return k;
        }
    }
}
