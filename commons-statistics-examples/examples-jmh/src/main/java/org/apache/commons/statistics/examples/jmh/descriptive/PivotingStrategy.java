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
 * A strategy to pick a pivoting index of an array for doing partitioning.
 */
enum PivotingStrategy {
    /**
     * Pivot around the centre of the range.
     */
    CENTRAL {
        @Override
        int pivotIndex(double[] data, int begin, int end) {
            return (begin + end) >>> 1;
        }

        @Override
        int pivotIndex(int[] data, int begin, int end) {
            return (begin + end) >>> 1;
        }
    },
    /**
     * Pivot around the median of 3 values within the range: the first; the centre; and the last.
     */
    MEDIAN_OF_3 {
        @Override
        int pivotIndex(double[] data, int begin, int end) {
            return med3(data, begin, (begin + end) >>> 1, end - 1);
        }

        @Override
        int pivotIndex(int[] data, int begin, int end) {
            return med3(data, begin, (begin + end) >>> 1, end - 1);
        }
    },
    /**
     * Pivot around the median of 9 values within the range.
     */
    MEDIAN_OF_9 {
        @Override
        int pivotIndex(double[] data, int begin, int end) {
            final int s = (end - begin) >>> 3;
            final int m = (begin + end) >>> 1;
            int z = end - 1;
            final int x = med3(data, begin, begin + s, begin + (s << 1));
            final double a = data[x];
            final int y = med3(data, m - s, m, m + s);
            final double b = data[y];
            z = med3(data, z - (s << 1), z - s, z);
            return med3(a, b, data[z], x, y, z);
        }

        @Override
        int pivotIndex(int[] data, int begin, int end) {
            final int s = (end - begin) >>> 3;
            final int m = (begin + end) >>> 1;
            int z = end - 1;
            final int x = med3(data, begin, begin + s, begin + (s << 1));
            final double a = data[x];
            final int y = med3(data, m - s, m, m + s);
            final double b = data[y];
            z = med3(data, z - (s << 1), z - s, z);
            return med3(a, b, data[z], x, y, z);
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
        int pivotIndex(double[] data, int begin, int end) {
            if (end - begin > MED_9) {
                return MEDIAN_OF_9.pivotIndex(data, begin, end);
            }
            return MEDIAN_OF_3.pivotIndex(data, begin, end);
        }

        @Override
        int pivotIndex(int[] data, int begin, int end) {
            if (end - begin > MED_9) {
                return MEDIAN_OF_9.pivotIndex(data, begin, end);
            }
            return MEDIAN_OF_3.pivotIndex(data, begin, end);
        }
    },
    /**
     * Pivot around the median of 5 values within the range.
     * Requires that {@code end - begin >= 5}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    MEDIAN_OF_5 {
        @Override
        int pivotIndex(double[] data, int begin, int end) {
            // Here we sort 5 points and choose 2 and 4 as the pivots: 1/6, 1/3, 1/2, 2/3, 5/6
            // 1/6 ~ 1/8 + 1/32. Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            int len = end - begin - 1;
            int sixth = 1 + (len >>> 3) + (len >>> 5);
            int p3 = begin + (len >>> 1);
            int p2 = p3 - sixth;
            int p1 = p2 - sixth;
            int p4 = p3 + sixth;
            int p5 = p4 + sixth;
            KthSelector.insertionSort5(data, p1, p2, p3, p4, p5);
            return p3;
        }

        @Override
        int pivotIndex(int[] data, int begin, int end) {
            throw new IllegalStateException("Unsupported");
        }
    };

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
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (exclusive).
     * @return the index of the pivot element chosen between the first and the last
     * element of the array slice
     */
    abstract int pivotIndex(double[] data, int begin, int end);

    /**
     * Find pivot index of the array so that partition and K<sup>th</sup> element
     * selection can be made.
     *
     * @param data Array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (exclusive).
     * @return the index of the pivot element chosen between the first and the last
     * element of the array slice
     */
    abstract int pivotIndex(int[] data, int begin, int end);
}
