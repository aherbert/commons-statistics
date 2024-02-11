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
 * A strategy to pick two pivot indices of an array for partitioning.
 *
 * <p>An ideal strategy will pick the tertiles across a variety of data so
 * to divide the data into [1/3, 1/3, 1/3].
 *
 * <a href="https://en.wiktionary.org/wiki/tertile">Tertile (Wiktionary)</a>
 */
enum DualPivotingStrategy {
    /**
     * Pivot around the medians at 1/3 and 2/3 of the range.
     *
     * <p>Requires {@code right - left >= 2}.
     */
    MEDIANS {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Original 'medians' method from the dual-pivot quicksort paper by Vladimir Yaroslavskiy
            final int len = right - left;
            // Do not pivot at the ends by setting 1/3 to at least 1.
            // This is safe if len >= 2.
            final int third = Math.max(1, len / 3);
            final int m1 = left + third;
            final int m2 = right - third;
            // Ensure p1 is lower
            if (data[m1] < data[m2]) {
                pivot2[0] = m2;
                return m1;
            }
            pivot2[0] = m1;
            return m2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int third = Math.max(1, len / 3);
            final int m1 = left + third;
            final int m2 = right - third;
            return new int[] {m1, m2};
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- sixths from the median: 1/6, 1/3, 1/2, 2/3, 5/6.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    SORT_5 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/6 = 5/30 ~ 1/8 + 1/32 + 1/64 : 0.1666 ~ 0.1719
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int sixth = 1 + (len >>> 3) + (len >>> 5) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - sixth;
            final int p1 = p2 - sixth;
            final int p4 = p3 + sixth;
            final int p5 = p4 + sixth;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int sixth = 1 + (len >>> 3) + (len >>> 5) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - sixth;
            final int p1 = p2 - sixth;
            final int p4 = p3 + sixth;
            final int p5 = p4 + sixth;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- sevenths from the median: 3/14, 5/14, 1/2, 9/14, 11/14.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    SORT_5B {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/7 = 5/35 ~ 1/8 + 1/64 : 0.1429 ~ 0.1406
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int seventh = 1 + (len >>> 3) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - seventh;
            final int p1 = p2 - seventh;
            final int p4 = p3 + seventh;
            final int p5 = p4 + seventh;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int seventh = 1 + (len >>> 3) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - seventh;
            final int p1 = p2 - seventh;
            final int p4 = p3 + seventh;
            final int p5 = p4 + seventh;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- eights from the median: 1/4, 3/8, 1/2, 5/8, 3/4.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    SORT_5C {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/8 = 0.125
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int eighth = 1 + (len >>> 3);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p4 = p3 + eighth;
            final int p5 = p4 + eighth;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int eighth = 1 + (len >>> 3);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p4 = p3 + eighth;
            final int p5 = p4 + eighth;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 1st and 5th values from 5 approximately uniformly spaced within the range.
     *
     * <p>Requires {@code right - left >= 3}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    SORT_5J {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // JDK 11 method

            // Does not work well for small range
            // step = size * 3 / 8 + 3
            int step = ((right - left + 1) >> 3) * 3 + 3;
            int p1 = left + step;
            int p5 = right - step;
            int p3 = (p1 + p5) >>> 1;
            int p2 = (p1 + p3) >>> 1;
            int p4 = (p3 + p5) >>> 1;

            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p5;
            return p1;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            int step = ((right - left + 1) >> 3) * 3 + 3;
            int p1 = left + step;
            int p5 = right - step;
            int p3 = (p1 + p5) >>> 1;
            int p2 = (p1 + p3) >>> 1;
            int p4 = (p3 + p5) >>> 1;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    };

    /** Sampled points are unchanged. */
    static final int UNCHANGED = 0;
    /** Sampled points are partially sorted. */
    static final int PARTIAL_SORT = 0x1;
    /** Sampled points are sorted. */
    static final int SORT = 0x2;

    /**
     * Find two pivot indices of the array so that partitioning into 3-regions can be made.
     *
     * <pre>{@code
     * left <= p1 <= p2 <= right
     * }</pre>
     *
     * <p>Returns two pivots so that {@code data[p1] <= data[p2]}.
     *
     * @param data Array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param pivot2 Second pivot.
     * @return first pivot
     */
    abstract int pivotIndex(double[] data, int left, int right, int[] pivot2);

    // The following methods allow the strategy and side effects to be tested

    /**
     * Get the indices of points that will be sampled.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the indices
     */
    abstract int[] getSampledIndices(int left, int right);

    /**
     * Get the effect on the sampled points.
     * <ul>
     * <li>0 - Unchanged
     * <li>1 - Partially sorted
     * <li>2 - Sorted
     * </ul>
     *
     * @return the effect
     */
    abstract int samplingEffect();
}
