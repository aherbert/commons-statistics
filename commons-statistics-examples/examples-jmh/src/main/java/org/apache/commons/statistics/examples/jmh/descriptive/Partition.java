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

import java.util.Arrays;

/**
 * Partition array data.
 *
 * <p>Arranges elements such that indices {@code k} correspond to their correctly
 * sorted value in the equivalent fully sorted array. For all indices {@code k}
 * and any index {@code i}:
 *
 * <pre>{@code
 * data[i < k] <= data[k] <= data[k < i]
 * }</pre>
 *
 * <p>Examples:
 *
 * <pre>
 * data    [0, 1, 2, 1, 2, 5, 2, 3, 3, 6, 7, 7, 7, 7]
 *
 *
 * k=4   : [0, 1, 2, 1], [2], [5, 2, 3, 3, 6, 7, 7, 7, 7]
 * k=4,8 : [0, 1, 2, 1], [2], [3, 3, 2], [5], [6, 7, 7, 7, 7]
 * </pre>
 *
 * @since 1.1
 */
final class Partition {
    // This class contains implementations for use in benchmarking.

    /** Minimum selection size for insertion sort rather than selection.
     * Dual-pivot quicksort used 27 in the original paper. */
    private static final int MIN_SELECT_SIZE = 17;

    /** A {@link PivotingStrategy} used for pivoting. */
    private final PivotingStrategy pivotingStrategy;

    /** Minimum selection size for insertion sort rather than selection. */
    private final int minSelectSize;

    /**
     * Partition function. Used to benchmark different implementations.
     */
    private interface PartitionFunction {
        /**
         * Partition (partially sort) the array.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>Ranges of the array to sort are provided as a predicate.
         *
         * <ul>
         * <li>If {@code sortRange} returns {@code true} for all {@code [k, k]}
         * ranges where {@code [left <= k <= right]} then a complete sort is performed.
         * <li>If {@code sortRange} returns {@code false} for {@code [left, right]}
         * then nothing is performed.
         * <li>Otherwise the array is partially sorted such that all {@code k}
         * within ranges that test as {@code true} using the {@code sortRange}
         * correspond to their correctly sorted value in the equivalent fully sorted array.
         * </ul>
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param sortRange Predicate.
         */
        void partition(double[] a, int left, int right, IntIntBiPredicate sortRange);
    }

    /**
     * Constructor with default {@link PivotingStrategy#MEDIAN_OF_3 median of 3} pivoting
     * strategy.
     */
    Partition() {
        this(PivotingStrategy.MEDIAN_OF_3);
    }

    /**
     * Constructor with specified pivoting strategy.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     */
    Partition(PivotingStrategy pivotingStrategy) {
        this(pivotingStrategy, MIN_SELECT_SIZE);
    }

    /**
     * Constructor with specified pivoting strategy and select size.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param minSelectSize Minimum selection size for insertion sort rather than selection.
     */
    Partition(PivotingStrategy pivotingStrategy, int minSelectSize) {
        this.pivotingStrategy = pivotingStrategy;
        this.minSelectSize = minSelectSize;
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array. For all indices {@code k}
     * and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * @param part Partition function.
     * @param data Values.
     * @param k Indices.
     */
    private void partition(PartitionFunction part, double[] data, int... k) {
        final int n = k.length;
        if (n < 1) {
            return;
        }
        // Handle NaN
        final int right = sortNaN(data);
        if (right < 1) {
            return;
        }
        // TODO
        // How to handle just computing the min/max (edge case).
        if (n == 1) {
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0]));
        } else if (n == 2) {
            // Could create a range here using the minSelectSize
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0], k[1]));
        } else if (n == 3) {
            // TODO: remove when the RangePredicates is complete to use the select size
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0], k[1], k[2]));
        // TODO Support 4 indices to allow two quantile to be estimated
        } else {
            final IntIntBiPredicate test = RangePredicates.ofIndex(minSelectSize, k);
            // Singleton allows use of reference comparison
            if (test == RangePredicates.anyRange()) {
                // Default
                Arrays.sort(data, 0, right + 1);
            } else {
                part.partition(data, 0, right, test);
            }
        }
    }

    /**
     * Sort the data.
     *
     * @param part Partition function.
     * @param data Values.
     */
    private void sort(PartitionFunction part, double[] data) {
        // Handle NaN
        final int right = sortNaN(data);
        if (right < 1) {
            return;
        }
        part.partition(data, 0, right, RangePredicates.anyRange());
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array. For all indices {@code k}
     * and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Values.
     * @param k Indices.
     */
    void partitionSBM(double[] data, int... k) {
        partition(this::partitionSBM, data, k);
    }

    /**
     * Sort the data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * @param data Values.
     */
    void sortSBM(double[] data) {
        sort(this::partitionSBM, data);
    }

    /**
     * Sort an array within the ranges identified by the {@code sortRange}.
     *
     * <p>Note: Requires that the range contains no NaN values.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Data array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param sortRange Predicate.
     */
    private void partitionSBM(double[] data, int begin, int end, IntIntBiPredicate sortRange) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys (Sedgewick's algorithm).
        //
        // Partition data using pivot P into less-than, greater-than or equal.
        // P is placed at the end to act as a sentinal.
        // k traverses the unknown region ??? and values moved if equal (l) or greater (g):
        //
        // left    p       i            j         q    right
        // |  ==P  |  <P   |     ???    |   >P    | ==P  |P|
        //
        // At the end P and additional equal values are swapped back to the centre.
        //
        // |         <P        | ==P |            >P        |
        //
        // Adapted from Sedgewick "Quicksort is optimal"
        // https://sedgewick.io/wp-content/themes/sedgewick/talks/2002QuicksortIsOptimal.pdf
        //
        // The algorithm has been changed so that:
        // - A pivot point must be provided.
        // - An edge case where the search meets in the middle is handled.
        // - Equal value data is not swapped to the end. Since the value is fixed then
        //   only the less than / greater than value must be moved from the end inwards.
        //   The end is then assumed to be the equal value. This would not work with
        //   object references. Equivalent swap calls are commented.
        // - Added a fast-forward over initial range containing the pivot.

        // Switch to insertion sort for small range
        if (end - begin <= minSelectSize) {
            insertionSort(data, begin, end, begin != 0);
            fixSignedZeros(data, begin, end);
            return;
        }

        final int l = begin;
        final int r = end;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinal value
        final int pivot = pivotingStrategy.pivotIndex(data, begin, end + 1);
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Special case: count signed zeros
        int c = 0;
        if (v == 0) {
            c = countSignedZeros(data, begin, end);
        }

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                if (c != 0) {
                    sortZero(data, begin, end);
                }
                return;
            }
        }
        // Cannot overrun as the prior scan using p stopped before the end
        while (data[q - 1] == v) {
            q--;
        }

        int i = p - 1;
        int j = q;

        for (;;) {
            do {
                ++i;
            } while (data[i] < v);
            while (v < data[--j]) {
                // Stop at l (not i) allows scan loops to be independent
                if (j == l) {
                    break;
                }
            }
            if (i >= j) {
                // Edge-case if search met on an internal pivot value
                // (not at the greater equal region, i.e. i < q).
                // Move this to the lower-equal region.
                if (i == j && v == data[i]) {
                    //swap(data, i++, p++)
                    //data[i++] = data[p++];
                    data[i++] = data[p];
                    data[p++] = v;
                }
                break;
            }
            //swap(data, i, j)
            final double vj = data[i];
            final double vi = data[j];
            data[i] = vi;
            data[j] = vj;
            if (vi == v) {
                //swap(data, i, p++)
                //data[i] = data[p++];
                data[i] = data[p];
                data[p++] = v;
            }
            if (vj == v) {
                //swap(data, j, --q)
                data[j] = data[--q];
                data[q] = v;
            }
        }
        // i is at the end (exclusive) of the less-than region

        // Place pivot value in centre
        //swap(data, r, i)
        data[r] = data[i];
        data[i] = v;

        // Move equal regions to the centre.
        // Set the pivot range [j, i) and move this outward for equal values.
        j = i++;

        // less-equal:
        //   for (int k = l; k < p; k++):
        //     swap(data, k, --j)
        // greater-equal:
        //   for (int k = r; k-- > q; i++) {
        //     swap(data, k, i)

        // Move the minimum of less-equal or less-than
        int move = Math.min(p - l, j - p);
        int lower = j - (p - l);
        for (int k = l; move-- > 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        int upper = i + (r - q);
        for (int k = r; move-- > 0; i++) {
            data[--k] = data[i];
            data[i] = v;
        }

        // Special case: fixed signed zeros
        if (c != 0) {
            p = lower;
            while (c-- > 0) {
                data[p++] = -0.0;
            }
            while (p < upper) {
                data[p++] = 0.0;
            }
        }

        // Equal in [lower, upper)

        // Recurse for the less and greater regions
        if (begin < lower - 1 && sortRange.test(begin, lower - 1)) {
            partitionSBM(data, begin, lower - 1, sortRange);
        }
        if (upper < end && sortRange.test(upper, end)) {
            partitionSBM(data, upper, end, sortRange);
        }
    }

    /**
     * Sorts an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>This method is fast up to approximately 40 - 80 values.
     *
     * <p>The {@code internal} flag indicates that the value at {@code data[begin - 1]}
     * is sorted.
     *
     * @param data Data array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param internal Internal flag.
     */
    static void insertionSort(double[] data, int begin, int end, boolean internal) {
        int j;
        if (internal) {
            // Assume data[begin - 1] is a pivot and acts as a sentinal on the range.
            // => no requirement to check j >= begin.
            for (int i = begin; ++i <= end;) {
                final double v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; v < data[--j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        } else {
            for (int i = begin; ++i <= end;) {
                final double v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; --j >= begin && v < data[j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        }
    }

    /**
     * Move NaN values to the end of the array.
     * This allows all other values to be compared using {@code <, ==, >} operators (with
     * the exception of signed zeros).
     *
     * @param data Values.
     * @return index of last non-NaN value (or -1)
     */
    static int sortNaN(double[] data) {
        int end = data.length;
        // Find first non-NaN
        while (--end >= 0) {
            if (!Double.isNaN(data[end])) {
                break;
            }
        }
        for (int i = end; --i >= 0;) {
            final double v = data[i];
            if (Double.isNaN(v)) {
                // swap(data, i, end--)
                data[i] = data[end];
                data[end] = v;
                end--;
            }
        }
        return end;
    }

    /**
     * Count the number of signed zeros (-0.0).
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @return the count
     */
    static int countSignedZeros(double[] data, int begin, int end) {
        // Count negative zeros
        int c = 0;
        for (int i = begin; i <= end; i++) {
            if (data[i] == 0 && Double.doubleToRawLongBits(data[i]) < 0) {
                c++;
            }
        }
        return c;
    }

    /**
     * Sort a range of all zero values.
     * This orders -0.0 before 0.0.
     *
     * <p>Warning: The range must contain only zeros.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void sortZero(double[] data, int begin, int end) {
        // Count negative zeros
        int c = 0;
        for (int i = begin; i <= end; i++) {
            if (Double.doubleToRawLongBits(data[i]) < 0) {
                c++;
            }
        }
        // Replace
        if (c != 0) {
            int i = begin;
            while (c-- > 0) {
                data[i++] = -0.0;
            }
            while (i <= end) {
                data[i++] = 0.0;
            }
        }
    }

    /**
     * Detect and fix the sort order of signed zeros. Assumes the data may have been
     * partially ordered around zero.
     *
     * <p>Searches for zeros if {@code data[begin] <= 0} and {@code data[end - 1] >= 0}.
     * If zeros are discovered in the range then they are assumed to be continuous.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    private static void fixSignedZeros(double[] data, int begin, int end) {
        int j;
        if (data[begin] <= 0 && data[end] >= 0) {
            int i = begin;
            while (data[i] < 0) {
                i++;
            }
            j = end;
            while (data[j] > 0) {
                j--;
            }
            sortZero(data, i, j);
        }
    }
}
