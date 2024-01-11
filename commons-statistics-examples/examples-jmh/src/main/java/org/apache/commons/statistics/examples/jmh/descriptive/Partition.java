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
    /** Minimum length between 2 pivots {@code p2 - p1} that requires a full sort. */
    private static final int SORT_BETWEEN_SIZE = 2;
    /** Mask to extract the positive index from an integer. */
    private static final int INDEX_MASK = Integer.MAX_VALUE;
    /** Mask to extract the sign-bit index from an integer. */
    private static final int SIGN_MASK = Integer.MIN_VALUE;
    /** Shift to extract the sign-bit from an integer.*/
    private static final int EXTRACT_SIGN_BIT = 31;
    /** Flag to indicate the value {@code right + 1} is a pivot. */
    private static final int RIGHT_PIVOT = 0x1;

    /** PivotStore that ignore pivots. */
    private static final PivotStore IGNORE_PIVOTS = new PivotStore() {
        @Override
        public void add(int fromIndex, int toIndex) {
            // No-op
        }
        @Override
        public void add(int index) {
            // No-op
        }
    };

    /** A {@link PivotingStrategy} used for pivoting. */
    private final PivotingStrategy pivotingStrategy;

    /** Minimum size for quickselect. Below this threshold partitioning using quickselect
     * is stopped. The strategy below this threshold varies, e.g. sort the remaining
     * range; or use heapselect. */
    private final int minQuickSelectSize;

    /** Setting to indicate strategy for processing of multiple keys. */
    private final KeyStrategy keyStrategy;

    /**
     * Define the strategy for processing multiple keys.
     */
    enum KeyStrategy {
        /** Sort unique keys, collate ranges and process in ascending order. */
        SEQUENTIAL,
        /** Process in input order using an IndexSet to cover the entire range. */
        INDEX_SET,
        /** Process in input order using a PivotSet to cover the minimum range. */
        PIVOT_CACHE;
    }

    /**
     * Partition function. Used to benchmark different implementations.
     */
    private interface TargetedPartitionFunction {
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
     * Partition function. Used to benchmark different implementations.
     *
     * <p>Note: The function is applied within a {@code [left, right]} bound. This bound
     * is set using the entire range of the data to process, or it may be a sub-range
     * due to previous partitioning. In this case the value at {@code left - 1} and/or
     * {@code right + 1} can be a pivot. The value at these pivot points will be {@code <=} or
     * {@code >=} respectively to all values within the range. This information is valuable
     * during recursive partitioning and is passed as flags to the partition method.
     */
    private interface PartitionFunction {

        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * a central region {@code [ka, kb]}. The central region should be entirely
         * sorted.
         *
         * <pre>{@code
         * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower bound (inclusive) of the central region.
         * @param kb Upper bound (inclusive) of the central region.
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         */
        void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner);

        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * a central region {@code [ka, kb]}. The central region should be entirely
         * sorted.
         *
         * <pre>{@code
         * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>The {@link PivotStore} is only required to record pivots after {@code kb}.
         * This is to support sequential ascending order processing of regions to partition.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower bound (inclusive) of the central region.
         * @param kb Upper bound (inclusive) of the central region.
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @param pivots Used to store sorted regions.
         */
        void partitionSequential(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots);

        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * a central region {@code [ka, kb]}. The central region should be entirely
         * sorted.
         *
         * <pre>{@code
         * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>The {@link PivotStore} records all pivots and sorted regions.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower bound (inclusive) of the central region.
         * @param kb Upper bound (inclusive) of the central region.
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @param pivots Used to store sorted regions.
         */
        void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots);

        /**
         * Sort the array in the range {@code [left, right]}.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         */
        void sort(double[] a, int left, int right, boolean leftInner, boolean rightInner);
    }

    /**
     * Partition function. Used to benchmark different implementations.
     *
     * <p>Note: The function is applied within a {@code [left, right]} bound. This bound
     * is set using the entire range of the data to process, or it may be a sub-range
     * due to previous partitioning. In this case the value at {@code left - 1} and/or
     * {@code right + 1} can be a pivot. The value at these pivot points will be {@code <=} or
     * {@code >=} respectively to all values within the range. This information is valuable
     * during recursive partitioning and is passed as flags to the partition method.
     */
    @FunctionalInterface
    private interface PairedPartitionFunction {
        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * an index {@code k}.
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>The {@link PivotStore} records all pivots and sorted regions.
         *
         * <p>Flags:
         * <ul>
         * <li>If the sign bit is set the function will ensure the position {@code k+1}
         * is correctly sorted.
         * <li>If the lowest bit is set the position {@code right + 1} is a pivot.
         * <li>Currently this function is used on full arrays; if {@code left > 0}
         * it can be assumed that {@code left - 1} is a pivot.
         * </ul>
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param k Index.
         * @param flags Flags.
         * @param pivots Used to store sorted regions.
         */
        void partitionPaired(double[] a, int left, int right, int k,
            int flags, PivotStore pivots);
    }

    /**
     * Partition function. Used to benchmark different implementations.
     *
     * <p>Note: The function is applied within a {@code [left, right]} bound. This bound
     * is set using the entire range of the data to process, or it may be a sub-range
     * due to previous partitioning. In this case the value at {@code left - 1} and/or
     * {@code right + 1} can be a pivot. The value at these pivot points will be {@code <=} or
     * {@code >=} respectively to all values within the range. This information is valuable
     * during recursive partitioning and is passed as flags to the partition method.
     */
    private interface KPartitionFunction {
        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * an index {@code k}.
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This function returns a value with information about {@code k + 1}.
         * <ul>
         * <li>The lower 32-bits contain {@code s}, the highest sorted index {@code s >= k}
         * <li>The upper 32-bits contain {@code p}, the closest known pivot {@code p > k},
         *  or the end of the range {@code right + 1}.
         * </ul>
         *
         * <p>The {@link PivotStore} records all pivots and sorted regions.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param k Index.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @param pivots Used to store sorted regions.
         * @return the partition information
         */
        long partition(double[] a, int left, int right, int k, boolean rightInner, PivotStore pivots);
    }

    /**
     * Partition function. Used to benchmark different implementations.
     *
     * <p>Note: The function is applied within a {@code [left, right]} bound. This bound
     * is set using the entire range of the data to process, or it may be a sub-range
     * due to previous partitioning. In this case the value at {@code left - 1} and/or
     * {@code right + 1} can be a pivot. The value at these pivot points will be {@code <=} or
     * {@code >=} respectively to all values within the range. This information is valuable
     * during recursive partitioning and is passed as flags to the partition method.
     */
    private interface K1PartitionFunction {
        /**
         * Partition (partially sort) the array in the range {@code [left, right]} around
         * an index {@code k}.
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This function should ensure that {@code k + 1} is also corrected ordered.
         *
         * <p>The {@link PivotStore} records all pivots and sorted regions.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param k Index.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @param pivots Used to store sorted regions.
         */
        void partition(double[] a, int left, int right, int k, boolean rightInner, PivotStore pivots);
    }

    /**
     * Single-pivot partition method.
     */
    private interface SPPartitionFunction {
        /**
         * Partition an array slice around a single pivot. Partitioning exchanges array
         * elements such that all elements smaller than pivot are before it and all
         * elements larger than pivot are after it.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This method returns the pivot.
         * <pre>{@code
         *                     |k0 |
         * |         <P        | P |            >P        |
         * }</pre>
         * <ul>
         * <li>k0: pivot point
         * </ul>
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @return pivot point
         */
        int partition(double[] a, int left, int right,
            boolean leftInner, boolean rightInner);
    }

    /**
     * Single-pivot partition method handling equal values.
     */
    @FunctionalInterface
    private interface SPEPartitionFunction extends PartitionFunction {
        /**
         * Partition an array slice around a single pivot. Partitioning exchanges array
         * elements such that all elements smaller than pivot are before it and all
         * elements larger than pivot are after it.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This method returns 2 points describing the pivot range of equal values.
         * <pre>{@code
         *                     |k0 k1|
         * |         <P        | ==P |            >P        |
         * }</pre>
         * <ul>
         * <li>k0: lower pivot point
         * <li>k1: upper pivot point
         * </ul>
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param upper Upper bound (inclusive) of the pivot range [k1].
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         * @return Lower bound (inclusive) of the pivot range [k0].
         */
        int partition(double[] a, int left, int right, int[] upper,
            boolean leftInner, boolean rightInner);

        // Add support to have a pivot cache. Assume it is to store pivots after kb.
        // Switch to not using it when right < kb, or doing a full sort between
        // left and right (pivots are irrelevant).

        @Override
        default void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner) {
            // Skip when [left, right] does not overlap [ka, kb]
            if (right - left < 1) {
                return;
            }
            // Assume: left <= right && ka <= kb
            // Ranges may overlap either way:
            // left ---------------------- right
            //        ka --- kb
            //
            // Requires full sort:
            // ka ------------------------- kb
            //        left ---- right
            //
            // TODO: how to add edge case for a full sort?
            // This will naturally perform a full sort when ka < left and kb > right

            // Edge case for a single point
            if (ka == right) {
                partitionMax(a, left, ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
            } else {
                final int[] upper = {0};
                final int k0 = partition(a, left, right, upper, leftInner, rightInner);
                final int k1 = upper[0];
                // Sorted in [k0, k1]
                // Unsorted in [left, k0) and (k1, right]
                if (ka < k0) {
                    partition(a, left, k0 - 1, ka, kb, leftInner, true);
                }
                if (kb > k1) {
                    partition(a, k1 + 1, right, ka, kb, true, rightInner);
                }
            }
        }

        @Override
        default void partitionSequential(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots) {
            // This method is a copy of the above method except:
            // - It records all sorted ranges to the cache
            // - It switches to the above method when the cache is not required
            if (right - left < 1) {
                return;
            }
            if (ka == right) {
                partitionMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
                pivots.add(kb);
            } else {
                final int[] upper = {0};
                final int k0 = partition(a, left, right, upper, leftInner, rightInner);
                final int k1 = upper[0];
                // Sorted in [k0, k1]
                // Unsorted in [left, k0) and (k1, right]
                pivots.add(k0, k1);

                if (ka < k0) {
                    if (k0 - 1 < kb) {
                        // Left branch entirely below kb - no cache required
                        partition(a, left, k0 - 1, ka, kb, leftInner, true);
                    } else {
                        partitionSequential(a, left, k0 - 1, ka, kb, leftInner, true, pivots);
                    }
                }
                if (kb > k1) {
                    partitionSequential(a, k1 + 1, right, ka, kb, true, rightInner, pivots);
                }
            }
        }

        @Override
        default void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots) {
            // This method is a copy of the above method except:
            // - It records all sorted ranges to the cache
            // - It switches to the above method when the cache is not required
            if (right - left < 1) {
                return;
            }
            if (ka == right) {
                partitionMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
                pivots.add(kb);
            } else {
                final int[] upper = {0};
                final int k0 = partition(a, left, right, upper, leftInner, rightInner);
                final int k1 = upper[0];
                // Sorted in [k0, k1]
                // Unsorted in [left, k0) and (k1, right]
                pivots.add(k0, k1);

                if (ka < k0) {
                    partition(a, left, k0 - 1, ka, kb, leftInner, true, pivots);
                }
                if (kb > k1) {
                    partition(a, k1 + 1, right, ka, kb, true, rightInner, pivots);
                }
            }
        }

        @Override
        default void sort(double[] a, int left, int right, boolean leftInner, boolean rightInner) {
            // Skip when [left, right] is sorted
            if (right - left < 1) {
                return;
            }
            final int[] upper = {0};
            final int k0 = partition(a, left, right, upper, leftInner, rightInner);
            final int k1 = upper[0];
            // Sorted in [k0, k1]
            // Unsorted in [left, k0) and (k1, right]
            sort(a, left, k0 - 1, leftInner, true);
            sort(a, k1 + 1, right, true, rightInner);
        }
    }

    /**
     * Partial sort function using a single-pivot partition method.
     *
     * <p>This method switches to a full sort when the range is small.
     *
     * <p>Note: This does not override the {@link #sort(double[], int, int)} method
     * to allow benchmarking of a sort driven using a single-pivot partition.
     */
    private class SPPartialSortFunction implements SPEPartitionFunction {
        /** Single-pivot partition function. */
        private final SPPartitionFunction fun;

        /**
         * @param fun Single-pivot partition function.
         */
        SPPartialSortFunction(SPPartitionFunction fun) {
            this.fun = fun;
        }

        @Override
        public int partition(double[] a, int left, int right, int[] upper,
            boolean leftInner, boolean rightInner) {
            // Cannot handle equal values so we do this by sorting
            // when the range is small, otherwise trying to divide into
            // half around the single pivot.
            if (right - left >= minQuickSelectSize) {
                final int pivot = fun.partition(a, left, right, leftInner, rightInner);
                upper[0] = pivot;
                return pivot;
            }
            Arrays.sort(a, left, right + 1);
            upper[0] = right;
            return left;
        }
    }

    /**
     * Dual-pivot partition method.
     */
    @FunctionalInterface
    private interface DPPartitionFunction extends PartitionFunction {
        /**
         * Partition an array slice around a dual pivots. Partitioning exchanges array
         * elements such that all elements smaller than pivot1 are before it and all
         * elements larger than pivot2 are after it.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This method returns 4 points describing the pivot range.
         * <pre>{@code
         *               |k0|k1                 k2|k3|
         * |  <P1        |P1|     P1<= & <= P2    |P2|      >P2   |
         * }</pre>
         * <ul>
         * <li>k0: lower pivot point
         * <li>k1: the start (inclusive) of the unsorted range between pivots
         * <li>k2: the end (inclusive) of the unsorted range between pivots
         * <li>k3: upper pivot point
         * </ul>
         *
         * <p>Bounds are set so {@code [k0, k1]} and {@code [k2, k3]} are fully sorted.
         * When the range {@code [k0, k3]} contains fully sorted elements the result is
         * set to {@code k1 = k3} and {@code k2 = k3}.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param bounds Points [k0, k1, k2, k3].
         * @param leftInner Flag to indicate {@code left - 1} is a pivot.
         * @param rightInner Flag to indicate {@code right + 1} is a pivot.
         */
        void partition(double[] a, int left, int right, int[] bounds,
            boolean leftInner, boolean rightInner);

        @Override
        default void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner) {
            // Skip when [left, right] does not overlap [ka, kb]
            //if (right - left < 1 || right < ka || left > kb) {
            if (right - left < 1) {
                return;
            }
            // Assume: left <= ka <= kb <= right
            // Edge case for a single point
            if (ka == right) {
                partitionMax(a, left, ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
            } else {
                final int[] bounds = new int[4];
                partition(a, left, right, bounds, leftInner, rightInner);
                final int k0 = bounds[0];
                final int k1 = bounds[1];
                final int k2 = bounds[2];
                final int k3 = bounds[3];
                // Sorted in [k0, k1] and [k2, k3]
                // Unsorted in [left, k0); (k1, k2); and (k3, right]
                //               |k0|k1                 k2|k3|
                // |  <P1        |P1|     P1<= & <= P2    |P2|      >P2   |
                //
                //     low                  middle                   high
                // |     ka  kb  |                           |            |  [1]
                // |     ka      |           kb              |            |  [2]
                // |     ka      |                           |     kb     |  [3]
                // |             |           ka              |     kb     |  [4]
                // |             |                           |   ka  kb   |  [5]
                // |             |         ka  kb            |            |  [6]
                if (ka < k0) {
                    partition(a, left, k0 - 1, ka, kb, leftInner, true);
                }
                if (ka < k2 || kb > k1) {
                    partition(a, k1 + 1, k2 - 1, ka, kb, true, true);
                }
                if (kb > k3) {
                    partition(a, k3 + 1, right, ka, kb, true, rightInner);
                }

//                if (RangePredicates.overlap(left, k0 - 1, ka, kb)) {
//                    //partition(a, left, k0 - 1, ka, Math.min(k0 - 1, kb));
//                    partition(a, left, k0 - 1, ka, kb);
//                }
//                if (RangePredicates.overlap(k1 + 1, k2 - 1, ka, kb)) {
//                    //partition(a, k1 + 1, k2 - 1, Math.max(k1 + 1, ka), Math.min(k2 - 1, kb));
//                    partition(a, k1 + 1, k2 - 1, ka, kb);
//                }
//                if (RangePredicates.overlap(k3 + 1, right, ka, kb)) {
//                    //partition(a, k3 + 1, right, Math.max(k3 + 1, ka), kb);
//                    partition(a, k3 + 1, right, ka, kb);
//                }

//                //               |k0|k1                 k2|k3|
//                // |  <P1        |P1|     P1<= & <= P2    |P2|      >P2   |
//                //
//                //     low                  middle                   high
//                // |     ka  kb  |                           |            |  [1]
//                // |     ka      |           kb              |            |  [2]
//                // |     ka      |                           |     kb     |  [3]
//                // |             |           ka              |     kb     |  [4]
//                // |             |                           |   ka  kb   |  [5]
//                // |             |         ka  kb            |            |  [6]
//                // We should call to partition in low, middle and high 3 times.
//                if (ka < k0) {
//                    // low: [1, 2, 3]
//                    partition(a, left, k0 - 1, ka, Math.min(k0 - 1, kb));
//                    if (kb > k1) {
//                        // middle: [2, 3]
//                        partition(a, k1 + 1, k2 - 1,
//                            k1 + 1, Math.min(k2 - 1, kb));
//                    }
//                }
//                if (kb > k3) {
//                    // high: [3, 4, 5]
//                    partition(a, k3 + 1, right, Math.max(k3 + 1, ka), kb);
//                    if (ka < k2 && ka >= k0) {
//                        // middle: [4] (avoid repeat of [2])
//                        partition(a, k1 + 1, k2 - 1,
//                            Math.max(k1 + 1, ka), k2 - 1);
//                    }
//                } else if (kb > k1 && ka >= k0) {
//                    // middle: [6] (avoid repeat of [2])
//                    partition(a, k1 + 1, k2 - 1,
//                        Math.max(k1 + 1, ka), Math.min(k2 - 1, kb));
//                }
            }
        }

        @Override
        default void partitionSequential(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots) {
            // This method is a copy of the above method except:
            // - It records all sorted ranges to the cache
            // - It switches to the above method when the cache is not required
            if (right - left < 1) {
                return;
            }
            if (ka == right) {
                partitionMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
                pivots.add(kb);
            } else {
                final int[] bounds = new int[4];
                partition(a, left, right, bounds, leftInner, rightInner);
                final int k0 = bounds[0];
                final int k1 = bounds[1];
                final int k2 = bounds[2];
                final int k3 = bounds[3];
                // Sorted in [k0, k1] and [k2, k3]
                // Unsorted in [left, k0); (k1, k2); and (k3, right]
                pivots.add(k0, k1);
                pivots.add(k2, k3);

                if (ka < k0) {
                    if (k0 - 1 < kb) {
                        // Left branch entirely below kb - no cache required
                        partition(a, left, k0 - 1, ka, kb, leftInner, true);
                    } else {
                        partitionSequential(a, left, k0 - 1, ka, kb, leftInner, true, pivots);
                    }
                }
                if (ka < k2 || kb > k1) {
                    if (k2 - 1 < kb) {
                        // Middle branch entirely below kb - no cache required
                        partition(a, k1 + 1, k2 - 1, ka, kb, true, true);
                    } else {
                        partitionSequential(a, k1 + 1, k2 - 1, ka, kb, true, true, pivots);
                    }
                }
                if (kb > k3) {
                    partitionSequential(a, k3 + 1, right, ka, kb, true, rightInner, pivots);
                }
            }
        }

        @Override
        default void partition(double[] a, int left, int right, int ka, int kb,
            boolean leftInner, boolean rightInner, PivotStore pivots) {
            // This method is a copy of the above method except:
            // - It records all sorted ranges to the cache
            if (right - left < 1) {
                return;
            }
            if (ka == right) {
                partitionMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                partitionMin(a, kb, right);
                pivots.add(kb);
            } else {
                final int[] bounds = new int[4];
                partition(a, left, right, bounds, leftInner, rightInner);
                final int k0 = bounds[0];
                final int k1 = bounds[1];
                final int k2 = bounds[2];
                final int k3 = bounds[3];
                // Sorted in [k0, k1] and [k2, k3]
                // Unsorted in [left, k0); (k1, k2); and (k3, right]
                pivots.add(k0, k1);
                pivots.add(k2, k3);

                if (ka < k0) {
                    partition(a, left, k0 - 1, ka, kb, leftInner, true, pivots);
                }
                if (ka < k2 || kb > k1) {
                    partition(a, k1 + 1, k2 - 1, ka, kb, true, true, pivots);
                }
                if (kb > k3) {
                    partition(a, k3 + 1, right, ka, kb, true, rightInner, pivots);
                }
            }
        }

        @Override
        default void sort(double[] a, int left, int right, boolean leftInner, boolean rightInner) {
            // Skip when [left, right] is sorted
            if (right - left < 1) {
                return;
            }
            final int[] bounds = new int[4];
            partition(a, left, right, bounds, leftInner, rightInner);
            final int k0 = bounds[0];
            final int k1 = bounds[1];
            final int k2 = bounds[2];
            final int k3 = bounds[3];
            // Sorted in [k0, k1] and [k2, k3]
            // Unsorted in [left, k0); (k1, k2); and (k3, right]
            sort(a, left, k0 - 1, leftInner, true);
            sort(a, k1 + 1, k2 - 1, true, true);
            sort(a, k3 + 1, right, true, rightInner);
        }
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
        this(pivotingStrategy, MIN_SELECT_SIZE, KeyStrategy.INDEX_SET);
    }

    /**
     * Constructor with specified quickselect size.
     *
     * @param minQuickSelectSize Minimum size for quickselect.
     */
    Partition(int minQuickSelectSize) {
        this(PivotingStrategy.MEDIAN_OF_3, minQuickSelectSize, KeyStrategy.INDEX_SET);
    }

    /**
     * Constructor with specified sequential key processing.
     *
     * @param keyStrategy Strategy for processing multiple keys.
     */
    Partition(KeyStrategy keyStrategy) {
        this(PivotingStrategy.MEDIAN_OF_3, MIN_SELECT_SIZE, keyStrategy);
    }

    /**
     * Constructor with specified pivoting strategy; quickselect size; and sequential key processing.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param keyStrategy Strategy for processing multiple keys.
     */
    Partition(PivotingStrategy pivotingStrategy, int minQuickSelectSize, KeyStrategy keyStrategy) {
        this.pivotingStrategy = pivotingStrategy;
        this.minQuickSelectSize = minQuickSelectSize;
        this.keyStrategy = keyStrategy;
    }

    /**
     * Move the minimum value to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMin(double[] data, int begin, int end) {
        partitionMinIgnoreZeros(data, begin, end);
        // Edge-case: if min was 0.0, check for a -0.0 above and swap.
        if (data[begin] == 0) {
            minZero(data, begin, end);
        }
    }

    /**
     * Move the two smallest values to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMin2(double[] data, int begin, int end) {
        // Note: This is a duplicate of partitionMin2IgnoreZeros
        // but with handling of signed partitioning zeros.
        // This cannot call partitionMin2IgnoreZeros as the
        // handling of a pair is different.
        // This is for comparative benchmarking.

        final int len = end - begin + 1;
        if (len <= 1) {
            return;
        }
        int j0 = begin;
        int j1 = begin + 1;
        if (DoubleMath.lessThan(data[j1], data[j0])) {
            final double v = data[j0];
            data[j0] = data[j1];
            data[j1] = v;
        }
        if (len == 2) {
            return;
        }
        double min0 = data[j0];
        double min1 = data[j1];

        for (int i = j1; ++i <= end;) {
            final double v = data[i];
            if (v < min1) {
                if (data[i] < min0) {
                    j1 = j0;
                    j0 = i;
                    min1 = min0;
                    min0 = v;
                } else {
                    j1 = i;
                    min1 = v;
                }
            }
        }

        // Move two smallest values
        // Start:
        // |j0|j1|....................
        // Possible ends:
        // |j0|j1|....................  Just overwrite the same values
        // |j0|  |......|j1|..........  Found 1 value smaller than larger of the original pair
        // |j1|  |......|j0|..........  Found 1 value smaller than smaller of the original pair **
        // |  |  |......|j0|....|j1|..  Found multiple smaller values
        // |  |  |......|j1|....|j0|..  Found multiple smaller values
        // Take care to not overwrite min values
        final double v0 = data[begin];
        final double v1 = data[begin + 1];
        data[begin] = min0;
        data[begin + 1] = min1;
        if (j1 == begin) {
            // ** Special case
            data[j0] = v1;
        } else {
            data[j0] = v0;
            data[j1] = v1;
        }

        // Edge-case: if min was 0.0, check for a -0.0 above and swap.
        if (min0 == 0) {
            minZero(data, begin, end);
        }
        if (min1 == 0) {
            minZero(data, begin + 1, end);
        }
    }

    /**
     * Move the maximum value to the end of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMax(double[] data, int begin, int end) {
        partitionMaxIgnoreZeros(data, begin, end);
        // Edge-case: if max was -0.0, check for a 0.0 below and swap.
        if (data[end] == 0) {
            maxZero(data, begin, end);
        }
    }

    /**
     * Place a negative signed zero at {@code begin} before any positive signed zero in the range,
     * {@code -0.0 < 0.0}.
     *
     * <p>Warning: Only call when {@code data[begin]} is zero.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    private static void minZero(double[] data, int begin, int end) {
        // Assume data[begin] is zero and check the sign bit
        if (Double.doubleToRawLongBits(data[begin]) >= 0) {
            // Check for a -0.0 above and swap.
            // We only require 1 swap as this is not a full sort of zeros.
            for (int k = begin; ++k <= end;) {
                if (data[k] == 0 && Double.doubleToRawLongBits(data[k]) < 0) {
                    data[k] = 0.0;
                    data[begin] = -0.0;
                    break;
                }
            }
        }
    }

    /**
     * Place a positive signed zero at {@code end} after any negative signed zero in the range,
     * {@code -0.0 < 0.0}.
     *
     * <p>Warning: Only call when {@code data[end]} is zero.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    private static void maxZero(double[] data, int begin, int end) {
        // Assume data[end] is zero and check the sign bit
        if (Double.doubleToRawLongBits(data[end]) < 0) {
            // Check for a 0.0 below and swap.
            // We only require 1 swap as this is not a full sort of zeros.
            for (int k = end; --k >= begin;) {
                if (data[k] == 0 && Double.doubleToRawLongBits(data[k]) >= 0) {
                    data[k] = -0.0;
                    data[end] = 0.0;
                    break;
                }
            }
        }
    }


    /**
     * Move the minimum value to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMinIgnoreZeros(double[] data, int begin, int end) {
        int i = begin;
        double min = data[i];
        int j = i;
        while (++i <= end) {
            if (data[i] < min) {
                min = data[i];
                j = i;
            }
        }
        //swap(data, begin, j)
        data[j] = data[begin];
        data[begin] = min;
    }

    /**
     * Move the two smallest values to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMin2IgnoreZeros(double[] data, int begin, int end) {
        final int len = end - begin + 1;
        if (len <= 1) {
            return;
        }
        int j0 = begin;
        int j1 = begin + 1;
        if (data[j1] < data[j0]) {
            final double v = data[j0];
            data[j0] = data[j1];
            data[j1] = v;
        }
        if (len == 2) {
            return;
        }
        double min0 = data[j0];
        double min1 = data[j1];

        for (int i = j1; ++i <= end;) {
            final double v = data[i];
            if (v < min1) {
                if (data[i] < min0) {
                    j1 = j0;
                    j0 = i;
                    min1 = min0;
                    min0 = v;
                } else {
                    j1 = i;
                    min1 = v;
                }
            }
        }

        // Move two smallest values
        // Start:
        // |j0|j1|....................
        // Possible ends:
        // |j0|j1|....................  Just overwrite the same values
        // |j0|  |......|j1|..........  Found 1 value smaller than larger of the original pair
        // |j1|  |......|j0|..........  Found 1 value smaller than smaller of the original pair **
        // |  |  |......|j0|....|j1|..  Found multiple smaller values
        // |  |  |......|j1|....|j0|..  Found multiple smaller values
        // Take care to not overwrite min values
        final double v0 = data[begin];
        final double v1 = data[begin + 1];
        data[begin] = min0;
        data[begin + 1] = min1;
        if (j1 == begin) {
            // ** Special case
            data[j0] = v1;
        } else {
            data[j0] = v0;
            data[j1] = v1;
        }
    }

    /**
     * Move the maximum value to the end of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * @param data Data.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    static void partitionMaxIgnoreZeros(double[] data, int begin, int end) {
        int i = end;
        double max = data[i];
        int j = i;
        while (--i >= begin) {
            if (data[i] > max) {
                max = data[i];
                j = i;
            }
        }
        //swap(data, end, j)
        data[j] = data[end];
        data[end] = max;
    }

    /**
     * Partition the minimum {@code n} elements below {@code k} where
     * {@code n = k - left + 1}. Uses a heap select algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and can be used to perform a full sort of the range below {@code k}.
     *
     * <p>For best performance this should be called with
     * {@code k - left < right - k}; and {@code k != left}, i.e.
     * to partition a value in the lower half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     * @param count Size of range to sort below k.
     * @param upper Upper bound (inclusive) of the sorted range.
     * @return Lower bound (inclusive) of the sorted range.
     * @see #partitionMin(double[], int, int)
     */
    static int partitionMinK(double[] a, int left, int right, int k, int count, int[] upper) {
        // Size of the heap
        int n = k - left + 1;
        // Build the heap using Floyd's heap-construction algorithm
        // Start at parent of the last element in the heap (n-1)
        for (int start = (n - 1) >> 1; start >= 0; start--) {
            maxHeapSiftDown(a, left, start, n);
        }
        // Scan the remaining data and insert
        // Heap is rooted at a[left]
        double max = a[left];
        for (int i = k; ++i <= right;) {
            if (a[i] < max) {
                // swap(a[left], a[i])
                a[left] = a[i];
                a[i] = max;
                maxHeapSiftDown(a, left, 0, n);
                max = a[left];
            }
        }

        // The max heap has been constructed in-place so a[left] is the max.
        // To partition a[k] we have to move elements from the top of the
        // heap to the position immediately after the end of the heap.

        // Index of a zero in the sorted region
        int zeroIndex = max == 0 ? k : -1;
        if (count > 0) {
            // Heap sort
            for (int c = count; n-- > 1 && c >= 0; c--) {
                // Move top of heap to the sorted end
                final double v = a[left];
                a[left] = a[left + n];
                a[left + n] = v;
                maxHeapSiftDown(a, left, 0, n);
            }
            if (zeroIndex < 0) {
                // min is not zero.
                // Check if the region we will claim as sorted contains zeros.
                zeroIndex = containsMixedZeros(a, left, k - 1);
            }
        } else {
            // swap(a[left], a[k])
            a[left] = a[k];
            a[k] = max;
        }

        // Fix signed zeros.
        if (zeroIndex >= 0) {
            // Partition the data around zero and return the sorted bounds
            final int lower = partitionZero(a, left, right, zeroIndex, upper);
            // Expand sorted bounds if there were many zeros in the range
            upper[0] = Math.max(upper[0], k);
            return Math.min(lower, k - count);
        }
        upper[0] = k;
        return k - count;
    }

    /**
     * Sift the top element down the max heap.
     *
     * @param a Heap data.
     * @param offset Offset of the heap in the data.
     * @param root Root of the heap.
     * @param n Size of the heap.
     */
    private static void maxHeapSiftDown(double[] a, int offset, int root, int n) {
        // For node i:
        // left child: 2i + 1
        // right child: 2i + 2
        // parent: floor((i-1) / 2)

        // Value to sift
        int p = root;
        final double v = a[offset + p];
        // Left child of root
        int c = (p << 1) + 1;
        while (c < n) {
            // Use the right child if greater
            if (c + 1 < n && a[offset + c] < a[offset + c + 1]) {
                c++;
            }
            if (v < a[offset + c]) {
                a[offset + p] = a[offset + c];
                p = c;
                c = (p << 1) + 1;
            } else {
                // Done
                break;
            }
        }
        a[offset + p] = v;
    }

    /**
     * Partition the maximum {@code n} elements above {@code k} where
     * {@code n = right - k + 1}. Uses a heap select algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and can be used to perform a full sort of the range above {@code k}.
     *
     * <p>For best performance this should be called with
     * {@code k - left > right - k}; and {@code k != right}, i.e.
     * to partition a value in the upper half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     * @param count Size of range to sort below k.
     * @param upper Upper bound (inclusive) of the sorted range.
     * @return Lower bound (inclusive) of the sorted range.
     * @see #partitionMax(double[], int, int)
     */
    static int partitionMaxK(double[] a, int left, int right, int k, int count, int[] upper) {
        // Size of the heap
        int n = right - k + 1;
        // Build the heap using Floyd's heap-construction algorithm
        // Start at parent of the last element in the heap (n-1)
        for (int start = (n - 1) >> 1; start >= 0; start--) {
            minHeapSiftDown(a, k, start, n);
        }
        // Scan the remaining data and insert
        // Heap is rooted at a[left]
        double min = a[k];
        for (int i = k; --i >= left;) {
            if (a[i] > min) {
                // swap(a[left], a[i])
                a[k] = a[i];
                a[i] = min;
                minHeapSiftDown(a, k, 0, n);
                min = a[k];
            }
        }

        // The min heap has been constructed in-place so a[k] is partitioned.
        // To partition further we have to move elements from the top of the
        // heap to the position immediately after the end of the heap
        // which will output a descending order at the end of the array.

        // Index of a zero in the sorted region
        int zeroIndex = min == 0 ? k : -1;
        if (count > 0) {
            for (int c = count; n-- > 1 && c >= 0; c--) {
                // Move top of heap to the (reverse) sorted end
                final double v = a[k];
                a[k] = a[k + n];
                a[k + n] = v;
                minHeapSiftDown(a, k, 0, n);
            }
            // Swap back the descending sequence
            for (int i = k, j = right, c = count; i < j && c >= 0; i++, j--) {
                final double v = a[i];
                a[i] = a[j];
                a[j] = v;
            }
            if (zeroIndex < 0) {
                // min is not zero.
                // Check if the region we will claim as sorted contains zeros.
                zeroIndex = containsMixedZeros(a, k + 1, right);
            }
        }

        // Fix signed zeros.
        if (zeroIndex >= 0) {
            // Partition the data around zero and return the sorted bounds
            final int lower = partitionZero(a, left, right, zeroIndex, upper);
            // Expand sorted bounds if there were many zeros in the range
            upper[0] = Math.max(upper[0], k + count);
            return Math.min(lower, k);
        }
        upper[0] = k + count;
        return k;
    }

    /**
     * Sift the top element down the max heap.
     *
     * @param a Heap data.
     * @param offset Offset of the heap in the data.
     * @param root Root of the heap.
     * @param n Size of the heap.
     */
    private static void minHeapSiftDown(double[] a, int offset, int root, int n) {
        // For node i:
        // left child: 2i + 1
        // right child: 2i + 2
        // parent: floor((i-1) / 2)

        // Value to sift
        int p = root;
        final double v = a[offset + p];
        // Left child of root
        int c = (p << 1) + 1;
        while (c < n) {
            // Use the right child if less
            if (c + 1 < n && a[offset + c] > a[offset + c + 1]) {
                c++;
            }
            if (v > a[offset + c]) {
                a[offset + p] = a[offset + c];
                p = c;
                c = (p << 1) + 1;
            } else {
                // Done
                break;
            }
        }
        a[offset + p] = v;
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
     * <p>Uses a partition function that uses a predicate to define
     * the range to sort.
     *
     * @param part Partition function.
     * @param data Values.
     * @param k Indices.
     * @param n Count of indices.
     */
    private static void partitionRange(TargetedPartitionFunction part, double[] data, int[] k, int n) {
        if (n < 1) {
            return;
        }
        // Handle NaN
        final int right = sortNaN(data);
        if (right < 1) {
            return;
        }
        // This is a partial implementation.
        // It allows partitioning without storing any pivots, even when
        // the target k are far apart.

        // Choosing when to recursively sort into a partition is provided
        // by a predicate. When the number of partitions of interest becomes
        // high the predicate is very complex. It could be implemented
        // using a BitSet to store all k of interest and then a range
        // test returns true when [left, right] surrounds a k.
        //
        // This method works well when k is a single value. When a pair
        // (k, k+1) then a pivot may occur on either k or k+1. The predicate
        // will test true to signal to partition a side where the k of
        // interest is at the end of the range:
        //  left                right
        //  |                       |
        //                      (k, k+1)
        // The function will continue to partition [left, right-1] to find k.
        // It would be faster to find the maximum value in [left, right-1]
        // This performance degradation can be observed when selecting a
        // median on small arrays. Odd arrays (single k) are faster than
        // even array (two k).
        // It is a non-ideal solution for single quantiles or a median implementation.

        if (n == 1) {
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0]));
        } else if (n == 2) {
            // Could create a range here using the minQuickSelectSize
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0], k[1]));
        } else if (n == 3) {
            part.partition(data, 0, right, RangePredicates.ofIndex(k[0], k[1], k[2]));
        } else {
            throw new IllegalStateException();
        }
    }

    /**
     * Sort the data.
     *
     * <p>Uses a partition function that uses a predicate to define
     * the range to sort. As such this reaches the maximum speed
     * of the sort as the predicate signals to recursively sort
     * all regions.
     *
     * @param part Partition function.
     * @param data Values.
     */
    private static void sortRange(TargetedPartitionFunction part, double[] data) {
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
     * @param n Count of indices.
     */
    void partitionRangeSBM(double[] data, int[] k, int n) {
        partitionRange(this::partitionRangeSBM, data, k, n);
    }

    /**
     * Sort the data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * @param data Values.
     */
    void sortRangeSBM(double[] data) {
        sortRange(this::partitionRangeSBM, data);
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
    // TODO - Remove used of instance methods for partitioning
    private void partitionRangeSBM(double[] data, int begin, int end, IntIntBiPredicate sortRange) {
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
        if (end - begin <= minQuickSelectSize) {
            Sorting.sort(data, begin, end, begin != 0);
            fixContinuousSignedZeros(data, begin, end);
            return;
        }

        final int l = begin;
        final int r = end;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinal value
        final int pivot = pivotingStrategy.pivotIndex(data, begin, end);
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Special case: count signed zeros
        int c = 0;
        if (v == 0) {
            c = countMixedSignedZeros(data, begin, end);
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
        final int lower = j - (p - l);
        for (int k = l; move-- > 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        final int upper = i + (r - q);
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
            partitionRangeSBM(data, begin, lower - 1, sortRange);
        }
        if (upper < end && sortRange.test(upper, end)) {
            partitionRangeSBM(data, upper, end, sortRange);
        }
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
     * @param right Upper bound of data (inclusive).
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    private void partition(PartitionFunction part, double[] data, int right, int[] k, int count) {
        if (count < 1 || right < 1) {
            return;
        }
        // Validate indices. Excludes indices > right.
        final int n = countIndices(k, count, right);
        if (n < 1) {
            return;
        }
        if (n == 1) {
            part.partition(data, 0, right, k[0], k[0], false, false);
        } else if (n == 2 && Math.abs(k[0] - k[1]) <= (minQuickSelectSize >>> 1)) {
            final int ka = Math.min(k[0], k[1]);
            final int kb = Math.max(k[0], k[1]);
            part.partition(data, 0, right, ka, kb, false, false);
        } else {
            // Allow non-sequential / sequential processing to be selected
            if (keyStrategy == KeyStrategy.SEQUENTIAL) {
                // Sequential processing
                final ScanningPivotCache pivots = keyAnalysis(right + 1, k, n, minQuickSelectSize >>> 1);
                if (k[0] == Integer.MIN_VALUE) {
                    // Full-sort recommended. Assume the partition function
                    // can choose to switch to using Arrays.sort.
                    part.sort(data, 0, right, false, false);
                } else {
                    partitionSequential(part, data, k, n, right, pivots);
                }
            } else if (keyStrategy == KeyStrategy.INDEX_SET) {
                // Non-sequential processing using non-optimised storage
                final IndexSet pivots = IndexSet.ofRange(0, right);
                // First index must partition the entire range
                part.partition(data, 0, right, k[0], k[0], false, false, pivots);
                for (int i = 1; i < n; i++) {
                    final int ki = k[i];
                    if (pivots.get(ki)) {
                        continue;
                    }
                    final int l = pivots.previousSetBit(ki);
                    int r = pivots.nextSetBit(ki);
                    if (r < 0) {
                        r = right + 1;
                    }
                    part.partition(data, l + 1, r - 1, ki, ki, l >= 0, r <= right, pivots);
                }
            } else if (keyStrategy == KeyStrategy.PIVOT_CACHE) {
                // Non-sequential processing using a pivot cache to optimise storage
                final PivotCache pivots = createPivotCacheForIndices(k, n);

                // Handle single-point or tiny range
                if ((pivots.right() - pivots.left()) <= (minQuickSelectSize >>> 1)) {
                    part.partition(data, 0, right, pivots.left(), pivots.right(), false, false);
                    return;
                }

                // TODO - estimate density of indices.
                // E.g. compress range by a power of 2 (e.g. next-power-of-2 after min select size).
                // If all bits in this compressed range are set then do a full sort.

                // Bracket the range so the rest is internal.
                // Note: Partition function handles min/max searching if ka/kb are at the end
                // of the range.
                final int ka = pivots.left();
                part.partition(data, 0, right, ka, ka, false, false, pivots);
                final int kb = pivots.right();
                int l = pivots.previousPivot(kb);
                int r = pivots.nextPivot(kb);
                if (r < 0) {
                    // Partition did not visit downstream
                    r = right + 1;
                }
                part.partition(data, l + 1, r - 1, kb, kb, true, r <= right, pivots);
                for (int i = 0; i < n; i++) {
                    final int ki = k[i];
                    if (pivots.contains(ki)) {
                        continue;
                    }
                    l = pivots.previousPivot(ki);
                    r = pivots.nextPivot(ki);
                    part.partition(data, l + 1, r - 1, ki, ki, true, true, pivots);
                }
            } else {
                throw new IllegalStateException("Unsupported: " + keyStrategy);
            }
        }
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
     * @param right Upper bound of data (inclusive).
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    private void partitionK(KPartitionFunction part, double[] data, int right, int[] k, int n) {
        if (n < 1 || right < 1) {
            return;
        }

        // Special cases
        // These must check the bounds of input indices.

        // Single point
        if (n == 1) {
            if (k[0] <= right) {
                part.partition(data, 0, right, k[0], false, null);
            }
            return;
        }
        // Special case for partition around adjacent indices (for interpolation)
        if (n == 2 && k[0] + 1 == k[1]) {
            if (k[0] <= right) {
                final long x = part.partition(data, 0, right, k[0], false, null);
                // Unpack the highest sorted position
                final int s = (int) x;
                if (k[1] > s) {
                    // Unpack the closest bounding pivot
                    final int p = (int) (x >>> Integer.SIZE);
                    partitionMin(data, k[1], p - 1);
                }
            }
            return;
        }

        if (keyStrategy == KeyStrategy.PIVOT_CACHE) {
            // Non-sequential processing using a pivot cache to optimise storage.
            // Bounds checks are required for all indices.
            // This strategy does not require (k,k+1) pairs to be ordered.
            // It relies on the partition function sorting small ranges and
            // the pivot cache to handle Order(1) look-up of previous pivots
            // in the case the index has already been sorted.
            final PivotCache pivots = createPivotCacheForNextIndices(k, n);

            if (k[0] <= right) {
                part.partition(data, 0, right, k[0], false, pivots);
            }
            for (int i = 1; i < n; i++) {
                final int ki = k[i];
                int l;
                // This assumes previousPivot(ki) will be as fast as contains(ki)
                if (ki > right || (l = pivots.previousPivot(ki)) == ki) {
                    // Already sorted
                    continue;
                }
                final int r = pivots.nextPivotOrElse(ki + 1, right + 1);
                part.partition(data, l + 1, r - 1, ki, r <= right,
                    // Final index does not require storing more pivots
                    i + 1 == n ? null : pivots);
            }
        } else {
            throw new IllegalStateException("Unsupported k-partitioning: " + keyStrategy);
        }
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
     * <p>All indices are assumed to be within {@code [0, right]}.
     *
     * <p>This method performs paired partitioning: it ensures that all {@code k + 1}
     * are also correctly partitioned. This is performed at negligible extra cost
     * and simplifies code for the caller.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros
     * may be destroyed (the mixture updated during partitioning). The caller is
     * responsible for counting a mixture of signed zeros and restoring them if
     * required.
     *
     * <p>This function assumes {@code n > 0} and {@code right > 0}; otherwise
     * there is nothing to do.
     *
     * @param part Partition function.
     * @param data Values.
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Indices (may be destructively modified).
     * @param n Count of indices (assumed to be strictly positive).
     */
    private void partitionK1(K1PartitionFunction part, double[] data, int right, int[] k, int n) {
        // TODO:
        // Update all code to compute (k, k+1) for all k.
        // Update all code to ignore zero checks.
        // Compare to existing implementations.

        // Special cases
        // These must check the bounds of input indices.

        // Single point
        if (n == 1) {
            part.partition(data, 0, right, k[0], false, null);
            return;
        }

        if (keyStrategy == KeyStrategy.PIVOT_CACHE) {
            // Non-sequential processing using a pivot cache to optimise storage.
            // Bounds checks are required for all indices.
            // This strategy does not require (k,k+1) pairs to be ordered.
            // It relies on the partition function sorting small ranges and
            // the pivot cache to handle Order(1) look-up of previous pivots
            // in the case the index has already been sorted.
            final PivotCache pivots = createPivotCacheForNextIndices(k, n);

            if (k[0] <= right) {
                part.partition(data, 0, right, k[0], false, pivots);
            }
            for (int i = 1; i < n; i++) {
                final int ki = k[i];
                int l = pivots.previousPivot(ki);
                // This assumes previousPivot(ki) will be as fast as contains(ki)
                if (l == ki && pivots.contains(ki + 1)) {
                    // (k, k+1) is already sorted
                    continue;
                }
                final int r = pivots.nextPivotOrElse(ki + 1, right + 1);
                part.partition(data, l + 1, r - 1, ki, r <= right,
                    // Final index does not require storing more pivots
                    i + 1 == n ? null : pivots);
            }
        } else {
            throw new IllegalStateException("Unsupported k1-partitioning: " + keyStrategy);
        }
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
     * <p>Negative indices are treated as a pair {@code k, k+1} where {@code k} is
     * the value with the sign bit removed. This is an optimisation for partitioning
     * neighbour indices required for data interpolation.
     *
     * @param part Partition function.
     * @param data Values.
     * @param right Upper bound of data (inclusive).
     * @param k Indices (may be destructively modified).
     */
    private void partitionPaired(PairedPartitionFunction part, double[] data, int right, int... k) {
        if (k.length < 1 || right < 1) {
            return;
        }

        // Note:
        // This function never checks k+1 == nextPivot.
        // The PairedPartitionFunction must handle this.

        // Validate indices.
        // Excludes indices > right. Clips pairs (k, k+1) where k == right.
        final int n = countPairedIndices(k, right);
        if (n < 1) {
            return;
        }

        if (n == 1) {
            // Special case. Partitioning only a single index / pair.
            final int k0 = k[0] & INDEX_MASK;
            final int s0 = k[0] & SIGN_MASK;
            part.partitionPaired(data, 0, right, k0, s0, IGNORE_PIVOTS);
            return;
        }
        if (n == 2) {
            // Special case. Partitioning only requires bracketing the second range.
            final int k0 = k[0] & INDEX_MASK;
            final int k1 = k[1] & INDEX_MASK;
            final int s0 = k[0] & SIGN_MASK;
            final int s1 = k[1] & SIGN_MASK;
            final PivotCache pivots = PivotCaches.ofPairedIndex(k[1]);
            part.partitionPaired(data, 0, right, k0, s0, pivots);
            final int l = pivots.previousPivot(k1);
            if (l < k1 || s1 < 0) {
                final int r = pivots.nextPivotOrElse(k1 + 1, right + 1);
                final int flags = s1 | ((r <= right) ? RIGHT_PIVOT : 0);
                part.partitionPaired(data, l + 1, r - 1, k1, flags, IGNORE_PIVOTS);
            }
            return;
        }

        // n > 2
        // There should be at least 1 index between the min/max index

        if (keyStrategy == KeyStrategy.INDEX_SET) {
            // Create storage for all indices between [min, max]
            // Partition around min and max
            // Partition internal k
            // Note:
            // Here we use an IndexSet for storage between [min, max]. This has no
            // index checks and assumes indices are in the supported range.
            // So a PivotCache implementation is used for the first two partitions
            // that ignores values outside the [min, max] range.

            // Non-sequential processing using an index set to optimise storage
            final IndexSet sortedK = createIndexSetForPairedIndices(k, n);
            final int nm1 = n - 1;
            final int k0 = k[0] & INDEX_MASK;
            final int kn = k[nm1] & INDEX_MASK;
            final int s0 = k[0] & SIGN_MASK;
            final int sn = k[nm1] & SIGN_MASK;

            // TODO - IndexSet to implement PivotCache directly
            final PivotCache pivots = sortedK.asScanningPivotCache(k0, kn + (sn >>> EXTRACT_SIGN_BIT));

            // Partition min
            part.partitionPaired(data, 0, right, k0, s0, pivots);

            // Partition max
            int l = pivots.previousPivot(kn);
            if (l < kn || sn < 0) {
                final int r = pivots.nextPivotOrElse(kn + 1, right + 1);
                part.partitionPaired(data, l + 1, r - 1, kn,
                    sn | ((r <= right) ? RIGHT_PIVOT : 0), pivots);
            }

            // Process internal indices within [min, max]
            for (int i = 1; i < nm1; i++) {
                final int ki = k[i] & INDEX_MASK;
                final int si = k[i] & SIGN_MASK;
                // These are always within [min, max]
                l = sortedK.previousSetBit(ki);
                if (l == ki && si == 0) {
                    // ki is a pivot, no ki+1
                    continue;
                }
                final int r = sortedK.nextSetBit(ki + 1);
                // Always internal
                final int flags = si | RIGHT_PIVOT;
                part.partitionPaired(data, l + 1, r - 1, ki, flags,
                    i == nm1 - 1 ? IGNORE_PIVOTS : sortedK);
            }
        } else if (keyStrategy == KeyStrategy.PIVOT_CACHE) {
            final PivotCache pivots = createPivotCacheForPairedIndices(k, n);
            for (int i = 0; i < n; i++) {
                final int ki = k[i] & INDEX_MASK;
                final int si = k[i] & SIGN_MASK;
                final int l = pivots.previousPivot(ki);
                if (l == ki && si == 0) {
                    // ki is a pivot, no ki+1
                    continue;
                }
                final int r = pivots.nextPivotOrElse(ki + 1, right + 1);
                final int flags = si | ((r <= right) ? RIGHT_PIVOT : 0);
                part.partitionPaired(data, l + 1, r - 1, ki, flags,
                    i == n - 1 ? IGNORE_PIVOTS : pivots);
            }
        } else {
            throw new IllegalStateException("Unsupported paired-key partitioning: " + keyStrategy);
        }
    }

    /**
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]} as defined by minimum and maximum index.
     *
     * @param indices Indices.
     * @param n Count of indices (must be strictly positive).
     * @return the pivot cache
     */
    static PivotCache createPivotCacheForIndices(int[] indices, int n) {
        int min = indices[0];
        int max = min;
        for (int i = 1; i < n; i++) {
            final int k = indices[i];
            min = Math.min(min, k);
            max = Math.max(max, k);
        }
        return PivotCaches.ofFullRange(min, max);
    }

    /**
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]} as defined by minimum and maximum index
     * in {@code [1, n]}.
     *
     * @param indices Indices.
     * @param n Count of indices (must be {@code > 1})
     * @return the pivot cache
     */
    static PivotCache createPivotCacheForNextIndices(int[] indices, int n) {
        int min = indices[1];
        int max = min;
        for (int i = 2; i < n; i++) {
            final int k = indices[i];
            min = Math.min(min, k);
            max = Math.max(max, k);
        }
        return PivotCaches.ofFullRange(min, max);
    }

    /**
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]} as defined by minimum and maximum index
     * when the indices are processed from {@code 0, 1, ..., n - 1}. This
     * allows optionally ignoring the first index.
     *
     * <p>It is assumed the sign bit is a flag indicating the index is a pair
     * {@code k, k+1}.
     *
     * @param indices Indices.
     * @param n Count of indices (must be strictly positive).
     * @return the pivot cache
     */
    private static PivotCache createPivotCacheForPairedIndices(int[] indices, int n) {
        if (n == 2) {
            // Ignore first index
            return PivotCaches.ofPairedIndex(indices[1]);
        }
        // Support the entire range
        int min = indices[0] & INDEX_MASK;
        int max = min + (indices[0] >>> EXTRACT_SIGN_BIT);
        for (int i = 1; i < n; i++) {
            final int ka = indices[i] & INDEX_MASK;
            final int kb = ka + (indices[i] >>> EXTRACT_SIGN_BIT);
            min = Math.min(min, ka);
            max = Math.max(max, kb);
        }
        return PivotCaches.ofFullRange(min, max);
    }

    /**
     * Return a {@link IndexSet} implementation to support the range
     * {@code [left, right]} as defined by minimum and maximum index.
     *
     * <p>It is assumed the sign bit is a flag indicating the index is a pair
     * {@code k, k+1}.
     *
     * <p>This method rearranges the indices so {@code indices[0] == min}
     * and {@code indices[n-1] == max}.
     *
     * @param indices Indices.
     * @param n Count of indices (must be {@code > 1})
     * @return the pivot cache
     */
    static IndexSet createIndexSetForPairedIndices(int[] indices, int n) {
        int min = indices[0] & INDEX_MASK;
        int max = min + (indices[0] >>> EXTRACT_SIGN_BIT);
        int mini = 0;
        int maxi = 0;
        for (int i = 1; i < n; i++) {
            final int ka = indices[i] & INDEX_MASK;
            final int kb = ka + (indices[i] >>> EXTRACT_SIGN_BIT);
            if (ka < min) {
                min = ka;
                mini = i;
            }
            if (kb > max) {
                max = kb;
                maxi = i;
            }
        }
        final IndexSet set = IndexSet.ofRange(min, max);
        // Rearrange. Use the actual key values.
        min = indices[mini];
        max = indices[maxi];
        if (mini == maxi) {
            // All indices are either k or (k, k+1) with k a constant.
            // Write directly to the ends
            indices[0] = min;
            indices[n - 1] = max;
        } else if (maxi == 0) {
            // min != max; [0] is the max
            // Record the min; swap the end with the min; record the max
            indices[0] = min;
            indices[mini] = indices[n - 1];
            indices[n - 1] = max;
        } else {
            // min != max; [0] is not the max
            // swap [0] with the min
            int k = indices[0];
            indices[0] = min;
            indices[mini] = k;
            // swap the end with the max
            k = indices[n - 1];
            indices[n - 1] = max;
            indices[maxi] = k;
        }
        return set;
    }

    /**
     * Analysis of keys to partition. The indices k are updated in-place. The keys are
     * processed to eliminate duplicates and sorted in ascending order. Close points are
     * joined into ranges using the minimum separation. A zero or negative separation
     * prevents creating ranges.
     *
     * <p>On output the indices contain ranges or single points to partition in ascending
     * order. Single points are identified as negative values and should be bit-flipped
     * to the index value.
     *
     * <p>If compression occurs the result will contain fewer indices than {@code n}.
     * The end of the compressed range is marked using {@link Integer#MIN_VALUE}. This
     * is outside the valid range for any single index and signals to stop processing
     * the ordered indices.
     *
     * <p>A {@link PivotCache} implementation is returned for optimal bracketing
     * of indices in the range after the first target range / point.
     *
     * <p>Examples:
     *
     * <pre>{@code
     *                                                 [L, R] PivotCache
     * [3]                -> [3]                       -
     *
     * // min separation 0
     * [3, 4, 5]          -> [~3, ~4, ~5]              [4, 5]
     * [3, 4, 7, 8]       -> [~3, ~4, ~7, ~8]          [4, 8]
     *
     * // min separation 1
     * [3, 4, 5]          -> [3, 5, MIN_VALUE]         -
     * [3, 4, 5, 8]       -> [3, 5, ~8, MIN_VALUE]     [8]
     * [3, 4, 5, 6, 7, 8] -> [3, 8, MIN_VALUE, ...]    -
     * [3, 4, 7, 8]       -> [3, 4, 7, 8]              [7, 8]
     * [3, 4, 7, 8, 99]   -> [3, 4, 7, 8, ~99]         [7, 99]
     * }</pre>
     *
     * <p>The length of data to partition can be used to determine if processing is
     * required. A full sort of the data is recommended by returning
     * {@code k[0] == Integer.MIN_VALUE}. This occurs if the length is sufficiently small
     * or the first range to partition covers the entire data.
     *
     * <p>Note: The signal marker {@code Integer.MIN_VALUE} is {@code Integer.MAX_VALUE}
     * bit flipped. It this is outside the range of any valid index into an array.
     *
     * @param size Length of the data to partition.
     * @param k Indices.
     * @param n Count of indices (must be strictly positive).
     * @param minSeparation Minimum separation between points (set to zero to disable ranges).
     * @return the pivot cache
     */
    // package-private for testing
    ScanningPivotCache keyAnalysis(int size, int[] k, int n, int minSeparation) {
        // Tiny data, signal to sort it
        if (size < minQuickSelectSize) {
            k[0] = Integer.MIN_VALUE;
            return null;
        }
        // TODO - optimise this
        // Sort the keys
        final IndexSet indices = Sorting.sortUnique(Math.max(6, minQuickSelectSize), k, n);
        // Find the max index
        int right = k[n - 1];
        if (right < 0) {
            right = ~right;
        }
        // Join up close keys using the min separation distance.
        final int left = compressRange(k, n, minSeparation);
        if (left < 0) {
            // Nothing to partition after the first target.
            // Recommend full sort if the range is effectively complete.
            // A range requires n > 1 and positive indices.
            if (n != 1 && k[0] >= 0 && size - (k[1] - k[0]) < minQuickSelectSize) {
                k[0] = Integer.MIN_VALUE;
            }
            return null;
        }
        // Return an optimal PivotCache to process keys in sorted order
        if (indices != null) {
            // Reuse storage from sorting large number of indices
            return indices.asScanningPivotCache(left, right);
        }
        return IndexSet.createScanningPivotCache(left, right);
    }

    /**
     * Compress sorted indices into ranges using the minimum separation.
     * Single points are identified by bit flipping to negative. The
     * first unused position after compression is set to {@link Integer#MIN_VALUE},
     * unless this is outside the array length (i.e. no compression).
     *
     * @param k Unique indices (sorted).
     * @param n Count of indices (must be strictly positive).
     * @param minSeparation Minimum separation between points.
     * @return the first index after the initial pair / point (or -1)
     */
    private static int compressRange(int[] k, int n, int minSeparation) {
        if (n == 1) {
            // Single point, mark the end of the range
            // No optimisation for minSeparation <= 0. This is not
            // sensible when processing pairs of points for interpolation.
            // The loop code below will still generate all single points and
            // identify the pivot cache range.
            if (1 < k.length) {
                k[1] = Integer.MIN_VALUE;
            }
            return -1;
        }
        // Start of range is in k[j]; end in p2
        int j = 0;
        int p2 = k[0];
        int secondTarget = -1;
        for (int i = 0; ++i < n;) {
            if (k[i] < 0) {
                // Start of duplicate indices
                break;
            }
            if (k[i] <= p2 + minSeparation) {
                // Extend range
                p2 = k[i];
            } else {
                // Store range or point (bit flipped)
                if (k[j] == p2) {
                    k[j] = ~p2;
                } else {
                    k[++j] = p2;
                }
                j++;
                // Next range is k[j] to p2
                k[j] = p2 = k[i];
                // Set the position of the second target
                if (secondTarget < 0) {
                    secondTarget = p2;
                }
            }
        }
        // Store range or point (bit flipped)
        // Note: If there is only 1 range then the second target is -1
        if (k[j] == p2) {
            k[j] = ~p2;
        } else {
            k[++j] = p2;
        }
        j++;
        // Add a marker at the end of the compressed indices
        if (j < k.length) {
            k[j] = Integer.MIN_VALUE;
        }
        return secondTarget;
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
     * <p>The keys must have been pre-processed by {@link #keyAnalysis(int, int[], int, int)}
     * to structure them for sequential processing.
     *
     * @param part Partition function.
     * @param data Values.
     * @param k Indices (created by key analysis).
     * @param n Count of indices.
     * @param right Upper bound (inclusive).
     * @param pivots Cache of pivots (created by key analysis).
     */
    private static void partitionSequential(PartitionFunction part, double[] data, int[] k, int n,
        int right, ScanningPivotCache pivots) {
        // Sequential processing of [s, s] single points / [s, e] pairs (regions).
        // Single-points are identified as negative indices.
        // The partition algorithm must run so each [s, e] is sorted:
        // lower---se----------------s---e---------upper
        // Pivots are stored to allow lower / upper to be set for the next region:
        // lower---se-------p--------s-p-e-----p---upper
        int i = 1;
        int s = k[0];
        int e;
        if (s < 0) {
            e = s = ~s;
        } else {
            e = k[i++];
        }

        // Key analysis has configured the pivot cache correctly for the first region.
        // If there is no cache, there is only 1 region.
        if (pivots == null) {
            part.partition(data, 0, right, s, e, false, false);
            return;
        }

        part.partitionSequential(data, 0, right, s, e, false, false, pivots);

        // Process remaining regions
        while (i < n) {
            s = k[i++];
            if (s < 0) {
                e = s = ~s;
            } else {
                e = k[i++];
            }
            if (s > right) {
                // End of indices
                break;
            }
            // Cases:
            // 1. l------s-----------r  Single point (s==e)
            // 2. l------se----------r  An adjacent pair of points
            // 3. l------s------e----r  A range of points (may contain internal pivots)
            // Find bounding region of range: [l, r)
            // Left (inclusive) is always above 0 as we have partitioned upstream already.
            // Right (exclusive) may not have been searched yet so we check right bounds.
            final int l = pivots.previousPivot(s);
            int r = pivots.nextPivot(e);
            if (r < 0) {
                r = right + 1;
            }

            // Create regions:
            // Partition: l------s--p1
            // Sort:                p1-----p2
            // Partition:                  p2-----e-----r
            // Look for internal pivots.
            int p1 = -1;
            int p2 = -1;
            if (e - s > 1) {
                final int p = pivots.nextPivot(s + 1);
                if (p > s && p < e) {
                    p1 = p;
                    p2 = pivots.previousPivot(e - 1);
                    if (p2 - p1 > SORT_BETWEEN_SIZE) {
                        // Special-case: multiple internal pivots
                        // Full-sort of (p1, p2). Walk the unsorted regions:
                        // l------s--p1                               p2----e-----r
                        //             ppppp-----pppp----pppp---------
                        //                  s1-e1    s1e1    s1-----e1
                        int e1 = pivots.previousNonPivot(p2);
                        while (p1 < e1) {
                            final int s1 = pivots.previousPivot(e1);
                            part.sort(data, s1 + 1, e1, true, true);
                            e1 = pivots.previousNonPivot(s1);
                        }
                    }
                }
            }

            // Pivots are only required for the next downstream region
            int sn = right + 1;
            if (i < n) {
                sn = k[i];
                if (sn < 0) {
                    sn = ~sn;
                }
            }
            // Current implementations will signal if this is outside the support.
            // Occurs on the last region the cache was created to support (i.e. sn > right).
            final boolean unsupportedCacheRange = !pivots.moveLeft(sn);

            // Note: The partition function uses inclusive left and right bounds
            // so use +/- 1 from pivot values. If r is not a pivot it is right + 1
            // which is a valid exclusive upper bound.

            if (p1 > s) {
                // At least 1 internal pivot:
                // l <= s < p1 and p2 < e <= r
                // If l == s or r == e these calls should fully sort the respective range
                part.partition(data, l + 1, p1 - 1, s, p1 - 1, true, p1 <= right);
                if (unsupportedCacheRange) {
                    part.partition(data, p2 + 1, r - 1, p2 + 1, e, true, r <= right);
                } else {
                    part.partitionSequential(data, p2 + 1, r - 1, p2 + 1, e, true, r <= right, pivots);
                }
            } else {
                // Single range
                if (unsupportedCacheRange) {
                    part.partition(data, l + 1, r - 1, s, e, true, r <= right);
                } else {
                    part.partitionSequential(data, l + 1, r - 1, s, e, true, r <= right, pivots);
                }
            }
        }
    }

    /**
     * Sort the data.
     *
     * @param part Partition function.
     * @param data Values.
     * @param right Upper bound (inclusive).
     */
    private static void sort(PartitionFunction part, double[] data, int right) {
        if (right < 1) {
            return;
        }
        // Signal entire range
        part.sort(data, 0, right, false, false);
    }

    // TODO - Add other implementations

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
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionSBM(double[] data, int[] k, int n) {
        // Handle NaN (this does assume n > 0)
        final int right = sortNaN(data);
        partition((SPEPartitionFunction) this::partitionSBM, data, right, k, n);
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
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionKSBM(double[] data, int[] k, int n) {
        // Handle NaN (this does assume n > 0)
        final int right = sortNaN(data);
        partitionK(this::partitionKSBM, data, right, k, n);
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
     * <p>This method ensures that all {@code k + 1} are also correctly partitioned.
     * The method assumes all {@code k} are valid indices into the data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    // TODO - call this function with only k of (k, k+1)
    void partitionK1SBM(double[] data, int[] k, int count) {
        // This method does all pre-processing for NaNs and signed zeros.
        // The partition function can then assume no NaNs and can
        // use zero as any other number. In particular any value == pivotValue
        // can be moved using the pivotValue. This is true if all zeros
        // are 0.0, or all are -0.0, but not a mixture.

        // Remove NaN and count mixed signed zeros
        int c = 0;
        int cn = 0;
        int right = data.length - 1;
        for (int i = data.length; --i >= 0;) {
            final double v = data[i];
            // NaN check
            if (v != v) {
                // swap(data, i, right--)
                data[i] = data[right];
                data[right] = v;
                right--;
            } else if (v == 0) {
                c++;
                if (Double.doubleToRawLongBits(v) < 0) {
                    cn++;
                }
                // XXX: For testing (some tests) unify the zeros here
                // and downstream detection code should ignore
                // correcting zeros.
                //data[i] = 0.0;
            }
        }
        if (right < 1) {
            // No non-NaN data
            return;
        }

        // Filter invalid indices removed by NaN check
        int n = count;
        if (right < k.length - 1) {
            for (int i = n; --i >= 0;) {
                if (k[i] > right) {
                    // swap(k, i, --n)
                    int j = k[i];
                    k[i] = k[--n];
                    k[n] = j;
                }
            }
            if (n == 0) {
                // NaNs for all k
                return;
            }
        }

        partitionK1(this::partitionK1SBM, data, right, k, n);

        // Fix signed zeros when a mixture of positive and negative.
        // i.e. cp > 0 && cn > 0
        if (cn > 0 && c > cn) {
            // Count of positive zeros
            c -= cn;
            // Use the partitioned indices to bracket zero.
            // For now we just fast-forward as much as possible.
            // Assumes partitioning has not changed indices (but
            // reordering is OK).
            int j = -1;
            for (int i = 0; i < n; i++) {
                int kk = k[i];
                if (data[kk] < 0) {
                    j = Math.max(j, kk);
                }
            }
            // Fix. Assume the zeros are all present so no bounds checks
            // are used when incrementing j.
            for (;;) {
                if (data[++j] == 0) {
                    data[j] = -0.0;
                    if (--cn == 0) {
                        break;
                    }
                }
            }
            // Finish the positive zeros
            for (;;) {
                if (data[++j] == 0) {
                    data[j] = 0.0;
                    if (--c == 0) {
                        break;
                    }
                }
            }
        }
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
     * <p>Negative indices are treated as a pair {@code k, k+1} where {@code k} is
     * the value with the sign bit removed. This is an optimisation for partitioning
     * neighbour indices required for data interpolation.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     */
    void partitionPairedSBM(double[] data, int... k) {
        // Handle NaN (this does assume k.length != 0 and partitioning is required)
        final int right = sortNaN(data);
        partitionPaired(this::partitionPairedSBM, data, right, k);
    }

    /**
     * Implementation of {@link KPartitionFunction} using a
     * Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index.
     * @param rightInner Flag to indicate {@code right + 1} is a pivot.
     * @param pivots Used to store sorted regions.
     * @return the partition information [downstream pivot, highest sorted position]
     */
    private long partitionKSBM(double[] a, int left, int right, int k, boolean rightInner,
            PivotStore pivots) {
        // This method drives partitioning using a narrowing bracket
        // around the index to partition. Note: It is important for JVM
        // optimisation to have a static partition function.

        final int[] upper = {0};
        int l = left;
        int r = right;
        // Continue until small range or close to the ends
        while (k != l && k != r && r - l > minQuickSelectSize) {
            // Pick a pivot and partition
            final int k0 = partitionSBM(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int k1 = upper[0];
            // Sorted in [k0, k1]
            // Unsorted in [left, k0) and (k1, right]
            if (pivots != null) {
                pivots.add(k0, k1);
            }
            if (k < k0) {
                r = k0 - 1;
            } else if (k > k1) {
                l = k1 + 1;
            } else {
                // Sorted range contains k
                // Pack [downstream pivot, highest sorted position]
                return ((r + 1L) << Integer.SIZE) | k1;
            }
        }
        // Edge of range partitioning
        // Currently only support min/max heap partitioning of size 1

        if (k == l) {
            partitionMin(a, k, r);
            if (pivots != null) {
                pivots.add(k);
            }
            // Here we only know k is sorted
            return ((r + 1L) << Integer.SIZE) | k;
        }
        if (k == r) {
            partitionMax(a, l, k);
            if (pivots != null) {
                pivots.add(k);
            }
        } else {
            // Switch to insertion sort for small range
            Sorting.sort(a, l, r, l > 0);
            fixContinuousSignedZeros(a, l, r);
            if (pivots != null) {
                pivots.add(l, r);
            }
        }
        // From here r+1 is a pivot or the end of the data and is sorted.
        return ((r + 1L) << Integer.SIZE) | (r + 1);
    }

    /**
     * Implementation of {@link K1PartitionFunction} using a
     * Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * <p>This function does not respect the ordering of signed zeros.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index.
     * @param rightInner Flag to indicate {@code right + 1} is a pivot.
     * @param pivots Used to store sorted regions.
     */
    private void partitionK1SBM(double[] a, int left, int right, int k, boolean rightInner,
            PivotStore pivots) {
        // This method drives partitioning using a narrowing bracket
        // around the index to partition. Note: It is important for JVM
        // optimisation to have a static partition function.

        final int[] upper = {0};
        int l = left;
        int r = right;
        // Continue until small range or close to the ends
        // TODO: support minQuickSelectSize and minSortSize
        while (k != l && k != r && r - l > minQuickSelectSize) {
            // Pick a pivot and partition
            final int k0 = partitionSBMIgnoreZeros(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int k1 = upper[0];
            // Sorted in [k0, k1]
            // Unsorted in [left, k0) and (k1, right]
            if (pivots != null) {
                pivots.add(k0, k1);
            }
            if (k < k0) {
                r = k0 - 1;
            } else if (k > k1) {
                l = k1 + 1;
            } else {
                // Edge case: Sorted range contains k.
                // This is not the usual exit point.
                // Here k == pivot (or a constant range contains k).
                // Ensure k+1 is sorted.
                // Note: k1 and r+1 are pivots.
                // Only sort if k1 < k+1 <= r.
                if (k == k1) {
                    if (k < r) {
                        partitionMinIgnoreZeros(a, k + 1, r);
                    }
                    if (pivots != null) {
                        pivots.add(k + 1);
                    }
                }
                return;
            }
        }
        // Edge of range partitioning
        // Currently only support min/max heap partitioning of size 1

        if (k == l) {
            // Here we use special support to partition (k,k+1)
            partitionMin2IgnoreZeros(a, k, r);
            if (pivots != null) {
                pivots.add(k, k + 1);
            }
            return;
        }

        // Here r+1 is a pivot or the end of the data so k+1 is sorted
        if (k == r) {
            partitionMaxIgnoreZeros(a, l, k);
            if (pivots != null) {
                pivots.add(k);
            }
            return;
        }

        // Switch to insertion sort for small range.
        // This is the expected exit point of this function.
        Sorting.sort(a, l, r, l > 0);
        if (pivots != null) {
            pivots.add(l, r);
        }
    }

    /**
     * Implementation of {@link PairedPartitionFunction} using a
     * Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index.
     * @param flags Flags.
     * @param pivots Used to store sorted regions.
     */
    private void partitionPairedSBM(double[] a, int left, int right, int k,
        int flags, PivotStore pivots) {
        // Ignore invalid ranges
        // These occur when both (k, k+1) are pivots
        if (right < left) {
            return;
        }
        // This method drives partitioning using a narrowing bracket
        // around the index to partition.
        final int[] upper = {0};
        int l = left;
        int r = right;
        // Continue until small range or close to the ends
        while (k != l && k != r && r - l > minQuickSelectSize) {
            // Pick a pivot and partition
            final int k0 = partitionSBM(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int k1 = upper[0];
            // Sorted in [k0, k1]
            // Unsorted in [left, k0) and (k1, right]
            if (pivots != null) {
                pivots.add(k0, k1);
            }
            if (k < k0) {
                r = k0 - 1;
            } else if (k > k1) {
                l = k1 + 1;
            } else {
                // Edge case: This is not the usual exit point.
                // Here pivot == k (or a constant range contains k).
                // Also sort k+1 if required.
                // Note: k1 and r+1 are pivots.
                // Only sort if k1 < k+1 <= r.
                if (flags < 0 && k == k1) {
                    if (k < r) {
                        partitionMin(a, k + 1, r);
                    }
                    pivots.add(k + 1);
                }
                return;
            }
        }

        // Edge of range partitioning
        // Currently only support min/max heap partitioning of size 1

        if (k == l) {
            partitionMin(a, k, r);
            if (pivots != null) {
                pivots.add(k);
            }
            if (flags < 0) {
                partitionMin(a, k + 1, r);
                if (pivots != null) {
                    pivots.add(k);
                }
            }
        // From here r+1 is a pivot or the end of the data and k+1 sorted.
        } else if (k == r) {
            partitionMax(a, l, k);
            if (pivots != null) {
                pivots.add(k);
            }
        } else {
            // Switch to insertion sort for small range
            Sorting.sort(a, l, r, l > 0);
            fixContinuousSignedZeros(a, l, r);
            if (pivots != null) {
                pivots.add(l, r);
            }
        }
    }

    /**
     * Sort the data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * @param data Values.
     */
    void sortSBM(double[] data) {
        // Handle NaN
        final int right = sortNaN(data);
        sort((SPEPartitionFunction) this::partitionSBM, data, right);
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
     * @param upper Upper bound (inclusive) of the pivot range.
     * @param leftInner Flag to indicate {@code left - 1} is a pivot.
     * @param rightInner Flag to indicate {@code right + 1} is a pivot.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int partitionSBM(double[] data, int begin, int end, int[] upper,
        boolean leftInner, boolean rightInner) {
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
        if (end - begin <= minQuickSelectSize) {
            Sorting.sort(data, begin, end, leftInner);
            fixContinuousSignedZeros(data, begin, end);
            upper[0] = end;
            return begin;
        }

        final int l = begin;
        final int r = end;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinal value
        final int pivot = pivotingStrategy.pivotIndex(data, begin, end);
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Special case: count signed zeros
        int c = 0;
        if (v == 0) {
            c = countMixedSignedZeros(data, begin, end);
        }

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                if (c != 0) {
                    sortZero(data, begin, end);
                }
                upper[0] = end;
                return begin;
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
        final int lower = j - (p - l);
        for (int k = l; move-- > 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        upper[0] = i + (r - q) - 1;
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
            while (p <= upper[0]) {
                data[p++] = 0.0;
            }
        }

        // Equal in [lower, upper]
        return lower;
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
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionSBM(double[] data, int begin, int end, int pivot, int[] upper) {
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

        final int l = begin;
        final int r = end;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinal value
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Special case: count signed zeros
        int c = 0;
        if (v == 0) {
            c = countMixedSignedZeros(data, begin, end);
        }

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                if (c != 0) {
                    sortZero(data, begin, end);
                }
                upper[0] = end;
                return begin;
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
        final int lower = j - (p - l);
        for (int k = l; move-- > 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        upper[0] = i + (r - q) - 1;
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
            while (p <= upper[0]) {
                data[p++] = 0.0;
            }
        }

        // Equal in [lower, upper]
        return lower;
    }

    /**
     * Sort an array within the ranges identified by the {@code sortRange}.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Data array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionSBMIgnoreZeros(double[] data, int begin, int end, int pivot, int[] upper) {
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

        final int l = begin;
        final int r = end;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinal value
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                upper[0] = end;
                return begin;
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
        final int lower = j - (p - l);
        for (int k = l; move-- > 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        upper[0] = i + (r - q) - 1;
        for (int k = r; move-- > 0; i++) {
            data[--k] = data[i];
            data[i] = v;
        }

        // Equal in [lower, upper]
        return lower;
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
     * Move invalid indices to the end of the array.
     *
     * @param indices Values.
     * @param right Upper bound of data (inclusive).
     * @param count Count of indices.
     * @return count of valid indices
     */
    static int countIndices(int[] indices, int count, int right) {
        int end = count;
        // Find first valid index
        while (--end >= 0) {
            if (indices[end] <= right) {
                break;
            }
        }
        for (int i = end; --i >= 0;) {
            final int k = indices[i];
            if (k > right) {
                // swap(indices, i, end--)
                indices[i] = indices[end];
                indices[end] = k;
                end--;
            }
        }
        return end + 1;
    }

    /**
     * Move invalid indices to the end of the array.
     *
     * <p>It is assumed the sign bit is a flag indicating the index is a pair
     * {@code k, k+1}. Pairs that straddle the upper bound are truncated
     * to {@code k == right}.
     *
     * @param indices Values.
     * @param right Upper bound of data (inclusive).
     * @return count of valid indices
     */
    static int countPairedIndices(int[] indices, int right) {
        int end = indices.length;
        // Find first valid index
        while (--end >= 0) {
            if ((indices[end] & INDEX_MASK) <= right) {
                // Clip pairs (k, k+1) where k == right
                if ((indices[end] & INDEX_MASK) == right) {
                    indices[end] &= INDEX_MASK;
                }
                break;
            }
        }
        for (int i = end; --i >= 0;) {
            final int k = indices[i] & INDEX_MASK;
            if (k >= right) {
                // Clip pairs (k, k+1) where k == right
                if (k == right) {
                    indices[i] = k;
                } else {
                    // swap(indices, i, end--)
                    indices[i] = indices[end];
                    indices[end] = k;
                    end--;
                }
            }
        }
        return end + 1;
    }

    /**
     * Return an index of a zero if the range contains a mix of positive and negative zeros.
     * If all positive, or all negative then this returns -1.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @return index of a zero
     */
    static int containsMixedZeros(double[] data, int begin, int end) {
        int c = 0;
        int cn = 0;
        for (int i = begin; i <= end; i++) {
            if (data[i] == 0) {
                c++;
                if (Double.doubleToRawLongBits(data[i]) < 0) {
                    cn++;
                }
                if (c != cn) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Count the number of signed zeros (-0.0).
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @return the count
     */
    static int countSignedZeros1(double[] data, int begin, int end) {
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
     * Count the number of signed zeros (-0.0) if the range contains a mix of positive and
     * negative zeros. If all positive, or all negative then this returns 0.
     *
     * <p>This method can be used when a pivot value is zero during partitioning when the
     * method uses the pivot value to replace values matched as equal using {@code ==}.
     * This may destroy a mixture of signed zeros by overwriting them as all 0.0 or -0.0
     * depending on the pivot value.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @return the count of signed zeros if some positive zeros are also present
     */
    static int countMixedSignedZeros(double[] data, int begin, int end) {
        // Count negative zeros
        int c = 0;
        int cn = 0;
        for (int i = begin; i <= end; i++) {
            if (data[i] == 0) {
                c++;
                if (Double.doubleToRawLongBits(data[i]) < 0) {
                    cn++;
                }
            }
        }
        return c == cn ? 0 : cn;
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
    private static void fixContinuousSignedZeros(double[] data, int begin, int end) {
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

    /**
     * Detect and fix the sort order of signed zeros. Assumes the data may have been
     * partially ordered around zero.
     *
     * <p>Searches for zeros if {@code data[begin] <= 0} and {@code data[end - 1] >= 0}.
     * This function is expensive if the range is large as it must scan the range twice.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     */
    private static void fixDiscontinuousSignedZeros(double[] data, int begin, int end) {
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
            // Zeros in [i, j]
            // Count the signed zeros and rewrite all zeros as 0.0
            int c = 0;
            for (int k = i; k <= j; k++) {
                if (data[k] == 0 && Double.doubleToRawLongBits(data[k]) < 0) {
                    data[k] = 0.0;
                    c++;
                }
            }
            for (int k = i; c != 0 && k <= j; k++) {
                if (data[k] == 0) {
                    data[k] = -0.0;
                    c--;
                }
            }
        }
    }

    /**
     * Perform a stable partition of the data around zeros (all zeros are moved to
     * the centre of the data, other elements are transferred in-order to the ends).
     * Respects the order of signed zeros.
     *
     * <p>Warning: Assumes the data contains at least 1 zero.
     *
     * @param data Values.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param pivot Location of known zero.
     * @param upper Upper bound (inclusive) of the sorted range containing zero.
     * @return Lower bound (inclusive) of the sorted range containing zero.
     */
    private static int partitionZero(double[] data, int begin, int end, int pivot, int[] upper) {
        // Move values less than the partition value to the start.
        // Move values greater than than the partition value to the end.
        // Skip zeros values.

        // Count of signed zeros
        //assert data[pivot] == 0;
        int c = Double.doubleToRawLongBits(data[pivot]) < 0 ? 1 : 0;

        int lt = begin;
        int gt = end;
        for (int i = begin; i < pivot; i++) {
            final double v = data[i];
            if (v < 0) {
                data[lt++] = v;
            } else {
                // Assume v == 0.0
                // Count signed zeros
                //assert v == 0;
                if (Double.doubleToRawLongBits(v) < 0) {
                    c++;
                }
            }
        }
        for (int i = end; i > pivot; i--) {
            final double v = data[i];
            if (v > 0) {
                data[gt--] = v;
            } else {
                // Assume v == 0.0
                // Count signed zeros
                //assert v == 0;
                if (Double.doubleToRawLongBits(v) < 0) {
                    c++;
                }
            }
        }

        // zeros in [lt, gt]
        // Fill in signed zeros
        int k = lt;
        while (--c >= 0) {
            data[k++] = -0.0;
        }
        while (k <= gt) {
            data[k++] = 0.0;
        }

        upper[0] = gt;
        return lt;
    }
}
