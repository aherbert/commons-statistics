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
 * A strategy to pick a pivoting index of an array for partitioning.
 *
 * <p>An ideal strategy will pick [1/2, 1/2] across a variety of data.
 */
enum PivotingStrategy {
    /**
     * Pivot around the centre of the range.
     */
    CENTRAL {
        @Override
        int pivotIndex(double[] data, int left, int right) {
            return (left + right) >>> 1;
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            return (left + right) >>> 1;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            return new int[] {(left + right) >>> 1};
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the median of 3 values within the range: the first; the centre; and the last.
     */
    MEDIAN_OF_3 {
        @Override
        int pivotIndex(double[] data, int left, int right) {
            return med3(data, left, (left + right) >>> 1, right);
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            return med3(data, left, (left + right) >>> 1, right);
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            return new int[] {left, (left + right) >>> 1, right};
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the median of 9 values within the range.
     */
    MEDIAN_OF_9 {
        @Override
        int pivotIndex(double[] data, int left, int right) {
            final int s = (right - left) >>> 3;
            final int m = (left + right) >>> 1;
            final int x = med3(data, left, left + s, left + (s << 1));
            final double a = data[x];
            final int y = med3(data, m - s, m, m + s);
            final double b = data[y];
            final int z = med3(data, right - (s << 1), right - s, right);
            return med3(a, b, data[z], x, y, z);
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            final int s = (right - left) >>> 3;
            final int m = (left + right) >>> 1;
            final int x = med3(data, left, left + s, left + (s << 1));
            final double a = data[x];
            final int y = med3(data, m - s, m, m + s);
            final double b = data[y];
            final int z = med3(data, right - (s << 1), right - s, right);
            return med3(a, b, data[z], x, y, z);
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int s = (right - left) >>> 3;
            final int m = (left + right) >>> 1;
            return new int[] {
                left, left + s, left + (s << 1),
                m - s, m, m + s,
                right - (s << 1), right - s, right
            };
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the median of 3 or 9 values within the range.
     *
     * <p>Note: Bentley & McIlroy (1993) choose a size of 40 to pivot around 9 values;
     * and a lower size of 7 to use the central; otherwise the median of 3.
     * This method does not switch to the central method for small sizes.
     */
    DYNAMIC {
        @Override
        int pivotIndex(double[] data, int left, int right) {
            if (right - left >= MED_9) {
                return MEDIAN_OF_9.pivotIndex(data, left, right);
            }
            return MEDIAN_OF_3.pivotIndex(data, left, right);
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            if (right - left >= MED_9) {
                return MEDIAN_OF_9.pivotIndex(data, left, right);
            }
            return MEDIAN_OF_3.pivotIndex(data, left, right);
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            if (right - left >= MED_9) {
                return MEDIAN_OF_9.getSampledIndices(left, right);
            }
            return MEDIAN_OF_3.getSampledIndices(left, right);
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the median of 5 values within the range.
     * Requires that {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     *
     * <p>Uses the same spacing as {@link DualPivotingStrategy#SORT_5}.
     */
    MEDIAN_OF_5 {
        @Override
        int pivotIndex(double[] data, int left, int right) {
            // 1/6 = 5/30 ~ 1/8 + 1/32 + 1/64 : 0.1666 ~ 0.1719
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            int len = right - left;
            final int sixth = 1 + (len >>> 3) + (len >>> 5) + (len >>> 6);
            int p3 = left + (len >>> 1);
            int p2 = p3 - sixth;
            int p1 = p2 - sixth;
            int p4 = p3 + sixth;
            int p5 = p4 + sixth;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            return p3;
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            throw new IllegalStateException("Unsupported");
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
     * Pivot around the median of 5 values within the range.
     * Requires that {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     *
     * <p>Uses the same spacing as {@link DualPivotingStrategy#SORT_5B}.
     */
    MEDIAN_OF_5B {
        @Override
        int pivotIndex(double[] data, int left, int right) {
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
            return p3;
        }

        @Override
        int pivotIndex(int[] data, int left, int right) {
            throw new IllegalStateException("Unsupported");
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
    };

    /** Sampled points are unchanged. */
    static final int UNCHANGED = 0;
    /** Sampled points are sorted. */
    static final int SORT = 0x2;
    /** Size to pivot around the median of 9. */
    private static final int MED_9 = 40;

    /**
     * Find the median index of 3.
     *
     * @param data Values.
     * @param i Index.
     * @param j Index.
     * @param k Index.
     * @return the median index
     */
    private static int med3(double[] data, int i, int j, int k) {
        return med3(data[i], data[j], data[k], i, j, k);
    }

    /**
     * Find the median index of 3 values.
     *
     * @param a Value.
     * @param b Value.
     * @param c Value.
     * @param i Index of a.
     * @param j Index of b.
     * @param k Index of c.
     * @return the median index
     */
    private static int med3(double a, double b, double c, int i, int j, int k) {
        if (a < b) {
            if (b < c) {
                return j;
            }
            return a < c ? k : i;
        }
        if (b > c) {
            return j;
        }
        return a > c ? k : i;
    }

    /**
     * Find the median index of 3.
     *
     * @param data Values.
     * @param i Index.
     * @param j Index.
     * @param k Index.
     * @return the median index
     */
    private static int med3(int[] data, int i, int j, int k) {
        return med3(data[i], data[j], data[k], i, j, k);
    }

    /**
     * Find the median index of 3 values.
     *
     * @param a Value.
     * @param b Value.
     * @param c Value.
     * @param i Index of a.
     * @param j Index of b.
     * @param k Index of c.
     * @return the median index
     */
    private static int med3(int a, int b, int c, int i, int j, int k) {
        if (a < b) {
            if (b < c) {
                return j;
            }
            return a < c ? k : i;
        }
        if (b > c) {
            return j;
        }
        return a > c ? k : i;
    }

    /**
     * Find pivot index of the array so that partition and K<sup>th</sup> element
     * selection can be made.
     *
     * @param data Array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the index of the pivot element chosen between the first and the last
     * element of the array slice
     */
    abstract int pivotIndex(double[] data, int left, int right);

    /**
     * Find pivot index of the array so that partition and K<sup>th</sup> element
     * selection can be made.
     *
     * @param data Array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the index of the pivot element chosen between the first and the last
     * element of the array slice
     */
    abstract int pivotIndex(int[] data, int left, int right);

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
