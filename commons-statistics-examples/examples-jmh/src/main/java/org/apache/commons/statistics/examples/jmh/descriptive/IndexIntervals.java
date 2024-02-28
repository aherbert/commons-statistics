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
    /** Size to perform key analysis. This avoids key analysis for a small number of keys. */
    private static final int KEY_ANALYSIS_SIZE = 10;

    /** Size to use a {@link BinarySearchKeyIndexInterval}. Note that the
     * {@link ScanningKeyIndexInterval} uses points within the range to fast-forward
     * scanning which improves performanse significantly for a few hundred indices.
     * Performance is similar when indices are in the thousands. Binary search is
     * much faster when there are multiple thousands of indices. */
    private static final int BINARY_SEARCH_SIZE = 2048;

    /** No instances. */
    private IndexIntervals() {}

    /**
     * Returns an interval that covers all indices ({@code [0, MAX_VALUE)}).
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
        // Note: A typical use case is to have a few indices. Thus the heuristics
        // in this method should be very fast when n is small. Here we skip them
        // completely when the number of keys is tiny.

        if (n > KEY_ANALYSIS_SIZE) {
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
            // An IndexSet will scan from the cut location and find a match in time proportional to
            // the index density. Average density is (size / n) and the scanning covers 64
            // indices together: Order(2 * n * (size / n) / 64) = Order(size / 32)

            // Get the range. This will throw an exception if there are no indices.
            int min = k[n - 1];
            int max = min;
            for (int i = n - 1; --i >= 0;) {
                min = Math.min(min, k[i]);
                max = Math.max(max, k[i]);
            }

            // Transition when n * n ~ size / 32
            // Benchmarking shows this is a reasonable approximation when size is small.
            // Speed of the IndexSet is approximately independent of n and proportional to size.
            // Large size observes the effect of memory cache degrading the IndexSet performance
            // more than expected from a linear relationship.
            // Note the memory required is approximately (size / 8) bytes.
            // We introduce a penalty for each 4x increase over size = 2^20 (== 128KiB).
            // n * n = size/32 * 2^log4(size / 2^20)

            // Transition point: n = sqrt(size/32)
            // size n
            // 2^10 5.66
            // 2^15 32.0
            // 2^20 181.0

            // Transition point: n = sqrt(size/32 * 2^(log4(size/2^20))))
            // size n
            // 2^22 512.0
            // 2^24 1448.2
            // 2^28 11585
            // 2^31 55108

            final int size = max - min + 1;

            // Divide by 32 is a shift of 5. This is reduced for each 4-fold size above 2^20.
            int shift = 5;
            if (size > (1 << 20)) {
                // log4(size/2^20) == (log2(size) - 20) / 2
                shift -= (ceilLog2(size) - 20) >>> 1;
            }

            if ((long) n * n > ((long) size >> shift)) {
                // Do not call IndexSet.of(k, n) which repeats the min/max search
                // (especially given n is likely to be large).
                final IndexSet interval = IndexSet.ofRange(min, max);
                for (int i = n; --i >= 0;) {
                    interval.set(k[i]);
                }
                return interval;
            }

            // Switch to binary search above a threshold.
            // Note this invalidates the speed assumptions based on the number of comparisons.
            // Benchmarking shows this is useful when the keys are in the thousands so this
            // would be used when data size is in the millions.
            if (n > BINARY_SEARCH_SIZE) {
                final int unique = Sorting.sortIndices2(k, n);
                return BinarySearchKeyIndexInterval.of(k, unique);
            }

            // Fall-though to the ScanningKeyIndexInterval...
        }

        // This is the typical use case.
        // Here n is small, or small compared to the min/max range of indices.
        // Use a special method to sort unique indices (detects already sorted indices).
        final int unique = Sorting.sortIndices2(k, n);

        return ScanningKeyIndexInterval.of(k, unique);
    }

    /**
     * Compute {@code ceil(log2(x))}. This is valid for all strictly positive {@code x}.
     *
     * <p>Returns -1 for {@code x = 0} in place of -infinity.
     *
     * @param x Value.
     * @return {@code ceil(log2(x))}
     */
    static int ceilLog2(int x) {
        return 32 - Integer.numberOfLeadingZeros(x - 1);
    }

    /**
     * {@link IndexInterval} for range {@code [0, MAX_VALUE)}.
     */
    private static final class AnyIndex implements IndexInterval {
        /** Singleton instance. */
        static final AnyIndex INSTANCE = new AnyIndex();

        @Override
        public int left() {
            return 0;
        }

        @Override
        public int right() {
            return Integer.MAX_VALUE - 1;
        }

        @Override
        public int previousIndex(int k) {
            return k;
        }

        @Override
        public int nextIndex(int k) {
            return k;
        }

        @Override
        public int split(int ka, int kb, int[] upper) {
            upper[0] = kb + 1;
            return ka - 1;
        }
    }
}
