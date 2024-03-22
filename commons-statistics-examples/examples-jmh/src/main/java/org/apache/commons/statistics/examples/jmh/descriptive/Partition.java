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
import java.util.Objects;
import java.util.function.IntConsumer;
import java.util.function.Supplier;

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
 * <p>References
 *
 * <p>Quickselect is introduced in Hoare [1]. This selects an element {@code k} from {@code n}
 * using repeat division of the data around a partition element, recursing into the
 * partition that contains {@code k}.
 *
 * <p>Introselect is introduced in Musser [2]. This detects excess recursion in quickselect
 * and reverts to heapselect to achieve an improved worst case bound on selection.
 *
 * <p>Use of dual-pivot quickselect is analysed in Wild et al [3] and shown to require
 * marginally more comparisons than single-pivot quickselect on a uniformly chosen order
 * statistic {@code k} and extremal order statistic (see table 1, page 19). This analysis
 * is reflected in the current implementation where dual-pivot quickselect is marginally
 * slower when {@code k} is close to the end of the data. However the dual-pivot quickselect
 * outperforms single-pivot quickselect when using multiple {@code k}; often significantly
 * when {@code k} or {@code n} are large.
 *
 * <ol>
 * <li>
 * Hoare (1961)
 * Algorithm 65: Find
 * <a href="https://doi.org/10.1145%2F366622.366647">Comm. ACM. 4 (7): 321–322</a>
 * <li>
 * Musser (1999)
 * Introspective Sorting and Selection Algorithms
 * <a href="https://doi.org/10.1002/(SICI)1097-024X(199708)27:8%3C983::AID-SPE117%3E3.0.CO;2-%23">
 * Software: Practice and Experience 27, 983-993.</a>
 * <li>
 * Wild, Nebel and Mahmoud (2013)
 * Analysis of Quickselect under Yaroslavskiy's Dual-Pivoting Algorithm
 * <a href="https://doi.org/10.48550/arXiv.1306.3819">arXiv:1306.3819</a>
 * <li><a href="https://en.wikipedia.org/wiki/Quickselect">Quickselect (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Introselect">Introselect (Wikipedia)</a>
 * </ol>
 *
 * @since 1.1
 */
final class Partition {
    // This class contains implementations for use in benchmarking.

    /** Default pivoting strategy. Note: Using the dynamic strategy avoids excess recursion
     * on the Bentley and McIlroy test data vs the MEDIAN_OF_3 strategy. It is possible
     * that selecting the points from within the range would improve the MEDIAN_OF_3 method;
     * currently it uses the left/right end points and the middle. */
    static final PivotingStrategy PIVOTING_STRATEGY = PivotingStrategy.DYNAMIC;
    /**
     * Default pivoting strategy. Choosing from 5 points is unbiased on random data and
     * has a lower standard deviation around the thirds than choosing 2 points
     * (Yaroslavskiy's original method, see {@link DualPivotingStrategy#MEDIANS}). It
     * performs well across various test data.
     *
     * <p>There are 3 variants using spacings of approximately 1/6, 1/7, and 1/8 computed
     * using shifts to create 0.1719, 0.1406, and 0.125; with middle thirds on large
     * lengths of 0.342, 0.28 and 0.25. The spacing using 1/7 is marginally faster when
     * performing a full sort than the others; thus favouring a smaller middle third, but
     * not too small, appears to be most performant.
     */
    static final DualPivotingStrategy DUAL_PIVOTING_STRATEGY = DualPivotingStrategy.SORT_5B;
    /** Minimum selection size for quickselect.
     * Below this switch to insertion sort rather than selection.
     * Dual-pivot quicksort used 27 in Yaroslavskiy's original paper.
     *
     * <p>This is set at a power of 2. This allows analysis of the indices saturation of
     * the range using compressed indices where compression uses a power of 2. */
    static final int MIN_QUICKSELECT_SIZE = 32;
    /** Minimum size for heapselect.
     * Below this switch to insertion sort rather than selection. This is used to avoid
     * heap select on tiny data. */
    static final int MIN_HEAPSELECT_SIZE = 5;
    /** Minimum size for sortselect.
     * Below this switch to insertion sort rather than selection. This is used to avoid
     * sort select on tiny data. */
    static final int MIN_SORTSELECT_SIZE = 4;
    /** Minimum selection size for single-pivot select (used for single k).
     * Below this switch to sort selection.
     * Changes to this value are only noticeable when the input array is small.
     * Benchmarking random data in range [96, 192] suggests a value of ~16. */
    static final int SP_QUICKSELECT_SIZE = 16;
    /** Minimum selection size for dual-pivot select (used for multiple k).
     * Below this switch to sort selection.
     * Changes to this value are only noticeable for select when the input array is small.
     * Benchmarking random data in range [162, 486] suggests a value of ~27 for n=1 and
     * increasing with higher n in the same range.
     * As keys become saturated then the partition method tends towards a full sort.
     * Dual-pivot sorting requires a value of ~120. If keys are saturated between k1 and kn
     * an increase to this threshold will gain full sort performance. */
    static final int DP_QUICKSELECT_SIZE = 27;
    /** Default length shift for heapselect. On random data this is approximately constant
     * at 6 or 7. Note that (n >>> 6) / n ~ 1/64. So heapselect will be used approximately 1.6%
     * of the time. On non-random data then the shift has to be larger. This idea is
     * captured in the heap select dynamic mask which can enable/disable dynamic determination
     * of the heap select size threshold. Note that a configurable size threshold for
     * heap select will not target the majority of cases and it is better to do quickselect
     * partitioning. This is disabled by default using the maximum shift. */
    static final int HEAPSELECT_SHIFT = 31;
    /** Default selection constant for heapselect. This is useful to pick up any indices
     * very close to the edge. Special methods handle size 1 and 2 without a heap by
     * finding the min or two smallest values in a range. The default enables the fast methods;
     * notably this will target (k, k+1) pairs of interpolation indices that may have been split
     * by a partition pivot. */
    static final int HEAPSELECT_CONSTANT = 2;
    /** Default shift for the heapselect dynamic threshold mask. Disabled by default.
     * It is a very small number of cases where heapselect is useful
     * (see {@link #heapSelectEdgeDistanceDP(int)}). */
    static final int HEAPSELECT_MASK_SHIFT = 31;
    /** Default selection constant for sortselect. Off by default as heapselect is used
     * in the default configuration for indices close to the edge. */
    static final int SORTSELECT_CONSTANT = 0;
    /** Default key strategy. */
    static final KeyStrategy KEY_STRATEGY = KeyStrategy.INDEX_SET;
    /** Default 1 or 2 key strategy. */
    static final PairedKeyStrategy PAIRED_KEY_STRATEGY = PairedKeyStrategy.SEARCHABLE_INTERVAL;
    /** Default recursion multiple. */
    static final int RECURSION_MULTIPLE = 2;
    /** Default recursion constant. */
    static final int RECURSION_CONSTANT = 0;
    /** Default compression. */
    static final int COMPRESSION_LEVEL = 1;

    /**
     * Sort select size for the the distance of a single k from the edge of the range
     * length n. Benchmarking single-pivot in range [64+32, 128+64] and dual-pivot in
     * range [81+81, 243+243] suggests a value of ~20 (or higher on some hardware). Ranges
     * are chosen based on half-interval spacing between powers of 2 for single pivot, or
     * third interval spacing between powers of 3 for dual pivot.
     *
     * <p>Sort select is faster at this small size than heap select. Note insertion into a
     * sorted array is Order(k) vs a heap which is Order(log2(k)) but has higher
     * complexity and non-local memory usage traversing the heap in jumps. At larger k the
     * heapselect is significantly faster.
     *
     * <p>On random data heap select can be used for small lengths when k ~ n / 2^6; this
     * ratio grows with length due to the log2(k) insertion cost. However on structured
     * data (ascending runs; repeat elements) quickselect can be dramatically faster
     * invalidating this relationship. Thus it is more robust on a variety of data input
     * to use quickselect until the distance from the edge is small. Heap select is
     * reserved for when quickselect fails to converge as expected.
     *
     * <p>A second advantage of sort select over heap select is that all indices closer to
     * the edge than the target index are also sorted. This allows selection of multiple
     * close indices to be performed with effectively the same speed. High density indices
     * can trigger use of sort select for small lengths to achieve a speed comparable to
     * quicksort. See {@link #dualPivotSortSelectSize(int, int, int)}.
     */
    static final int SORTSELECT_SIZE = 20;
    /** Separation distance to identify a range of indices as a single index. This allows
     * switching to an implementation optimised for a single target index. */
    private static final int MIN_SEPARATION_DISTANCE = 8;
    /** Increment used for the recursion counter. The counter will overflow to negative when
     * recursion has exceeded the maximum level. The counter is maintained in the upper bits
     * of the dual-pivot control flags. */
    private static final int RECURSION_INCREMENT = 1 << 20;
    /** Mask to extract the sort select size from the dual-pivot control flags. Currently
     * the bits below those used for the recursion counter are only used for the sort select size
     * so this can use a mask with all bits below the increment. */
    private static final int SORTSELECT_MASK = RECURSION_INCREMENT - 1;

    /** floor(log2(MIN_QUICKSELECT_SIZE / 2)). */
    private static final int LOG2_HALF_QUICKSELECT_SIZE = 4;
    /** Message for an unsupported introselect configuration. */
    private static final String UNSUPPORTED_INTROSELECT = "Unsupported introselect: ";
    /** Number of keys for saturation analysis. */
    private static final int SATURATION_ANALYSIS_COUNT = 10;

    /** Transformer factory for double data with the behaviour of a JDK sort.
     * Moves NaN to the end of the data and handles signed zeros. Works on the data in-place. */
    private static final Supplier<DoubleDataTransformer> SORT_TRANSFORMER =
        DoubleDataTransformers.createFactory(NaNPolicy.INCLUDE, false);

    /** Minimum length between 2 pivots {@code p2 - p1} that requires a full sort. */
    private static final int SORT_BETWEEN_SIZE = 2;
    /** Mask to extract the positive index from an integer. */
    private static final int INDEX_MASK = Integer.MAX_VALUE;

    // Use final for settings/objects used within partitioning functions

    /** A {@link PivotingStrategy} used for pivoting. */
    private final PivotingStrategy pivotingStrategy;
    /** A {@link DualPivotingStrategy} used for pivoting. */
    private final DualPivotingStrategy dualPivotingStrategy;

    /** Minimum size for quickselect. Below this threshold partitioning using quickselect
     * is stopped and a sort selection is performed. */
    private final int minQuickSelectSize;
    /** Length shift for heapselect. Heapselect runs when k is within d of the end of
     * length n using {@code d = (n >>> shift) + c}.
     * Not supported by all partition methods. */
    private final int heapSelectShift;
    /** Constant for heapselect. Heapselect runs when k is within d of the end of
     * length n using {@code d = (n >>> shift) + c}.
     * Not supported by all partition methods. */
    private final int heapSelectConstant;
    /** Mask used on the dynamic threshold for heapselect. The number of lower bits set in
     * this mask controls the maximum value for the dynamic heapselect threshold. If zero
     * then the dynamic threshold is ignored. If {@link Integer#MAX_VALUE} then the dynamic
     * threshold is always used. */
    private final int heapSelectDynamicMask;
    /** Constant for sortselect. Sortselect runs when k is within c of the end of
     * length n. Unlike heapselect, sortselect has no length dependent configuration
     * as insertion sort scales poorly with larger size (insertion is Order(n) vs
     * Order(log(n)) for a heap. */
    private final int sortSelectConstant;

    // Use final for settings used to configure partitioning functions

    /** Setting to indicate strategy for processing of multiple keys. */
    private KeyStrategy keyStrategy = KEY_STRATEGY;
    /** Setting to indicate strategy for processing of 1 or 2 keys. */
    private PairedKeyStrategy pairedKeyStrategy = PAIRED_KEY_STRATEGY;

    /** Multiplication factor {@code m} applied to the length based recursion factor {@code x}.
     * The recursion is set using {@code m * x + c}. */
    private double recursionMultiple = RECURSION_MULTIPLE;
    /** Constant {@code c} added to the length based recursion factor {@code x}.
     * The recursion is set using {@code m * x + c}. */
    private int recursionConstant = RECURSION_CONSTANT;
    /** Compression level for a {@link CompressedIndexSet} (in [1, 31]). */
    private int compression = COMPRESSION_LEVEL;
    /** Consumer for the recursion level reached during partitioning. Used to analyse
     * the distribution of the recursion for different input data. */
    private IntConsumer recursionConsumer = i -> { /* no-op */ };

    /**
     * Define the strategy for processing multiple keys.
     */
    enum KeyStrategy {
        /** Sort unique keys, collate ranges and process in ascending order. */
        SEQUENTIAL,
        /** Process in input order using an {@link IndexSet} to cover the entire range.
         * Introselect implementations will use a {@link SearchableInterval}. */
        INDEX_SET,
        /** Process in input order using a {@link CompressedIndexSet} to cover the entire range.
         * Introselect implementations will use a {@link SearchableInterval}. */
        COMPRESSED_INDEX_SET,
        /** Process in input order using a {@link PivotCache} to cover the minimum range. */
        PIVOT_CACHE,
        /** Sort unique keys and process using recursion with division of the keys
         * for each sub-partition. */
        ORDERED_KEYS,
        /** Sort unique keys and process using recursion with a {@link ScanningKeyInterval}. */
        SCANNING_KEY_SEARCHABLE_INTERVAL,
        /** Sort unique keys and process using recursion with a {@link BinarySearchKeyInterval}. */
        SEARCH_KEY_SEARCHABLE_INTERVAL,
        /** Sort unique keys and process using recursion with a {@link KeyIndexIterator}. */
        INDEX_ITERATOR,
        /** Process in input order using an {@link IndexIterator} of a {@link CompressedIndexSet}. */
        COMPRESSED_INDEX_ITERATOR,
        /** Process using recursion with an {@link IndexSet}-based {@link UpdatingInterval}. */
        INDEX_SET_UPDATING_INTERVAL,
        /** Sort unique keys and process using recursion with a {@link KeyUpdatingInterval}. */
        KEY_UPDATING_INTERVAL;
    }

    /**
     * Define the strategy for processing 1 or 2 keys.
     */
    enum PairedKeyStrategy {
        /** Use a dedicated single key method that returns information about (k+1). */
        PAIRED_KEYS,
        /** Use a method that accepts two keys. */
        TWO_KEYS,
        /** Use an {@link SearchableInterval} covering the keys. This will reuse a multi-key
         * strategy with keys that are a very small range. */
        SEARCHABLE_INTERVAL,
        /** Use an {@link UpdatingInterval} covering the keys. This will reuse a multi-key
         * strategy with keys that are a very small range. */
        UPDATING_INTERVAL;
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
            // This will naturally perform a full sort when ka < left and kb > right

            // Edge case for a single point
            if (ka == right) {
                selectMax(a, left, ka);
            } else if (kb == left) {
                selectMin(a, kb, right);
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
                selectMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                selectMin(a, kb, right);
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
                selectMax(a, left, ka);
                pivots.add(ka);
            } else if (kb == left) {
                selectMin(a, kb, right);
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
     * Single-pivot partition method handling equal values.
     */
    @FunctionalInterface
    interface SPEPartition {
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
         * <li>k1: upper pivot point (inclusive)
         * </ul>
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param upper Upper bound (inclusive) of the pivot range [k1].
         * @param pivot Pivot location.
         * @return Lower bound (inclusive) of the pivot range [k0].
         */
        int partition(double[] a, int left, int right, int pivot, int[] upper);
    }

    /**
     * Dual-pivot partition method handling equal values.
     */
    @FunctionalInterface
    interface DPPartition {
        /**
         * Partition an array slice around two pivots. Partitioning exchanges array
         * elements such that all elements smaller than pivot are before it and all
         * elements larger than pivot are after it.
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This method returns 4 points describing the pivot ranges of equal values.
         * <pre>{@code
         *         |k0  k1|                |k2  k3|
         * |   <P  | ==P1 |  <P1 && <P2    | ==P2 |   >P   |
         * }</pre>
         * <ul>
         * <li>k0: lower pivot1 point
         * <li>k1: upper pivot1 point (inclusive)
         * <li>k2: lower pivot2 point
         * <li>k3: upper pivot2 point (inclusive)
         * </ul>
         *
         * <p>Bounds are set so {@code i < k0},  {@code i > k3} and {@code k1 < i < k2} are
         * unsorted. When the range {@code [k0, k3]} contains fully sorted elements the result
         * is set to {@code k1 = k3; k2 == k0}. This can occur if
         * {@code P1 == P2} or there are zero or 1 value between the pivots
         * {@code P1 < v < P2}. Any sort between {@code k1 + 1} and {@code k2 - 1} must handle
         * a negative length. Any select of an index interval {@code [ka, kb]} that identifies
         * {@code k1 < kb || ka < k2} must also check {@code k2 - k1 > 1}.
         *
         * @param a Data array.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param bounds Points [k1, k2, k3].
         * @param pivot1 Pivot1 location.
         * @param pivot2 Pivot2 location.
         * @return Lower bound (inclusive) of the pivot range [k0].
         */
        int partition(double[] a, int left, int right, int pivot1, int pivot2, int[] bounds);
    }

    /**
     * Constructor with defaults.
     */
    Partition() {
        this(PIVOTING_STRATEGY, DUAL_PIVOTING_STRATEGY, MIN_QUICKSELECT_SIZE,
            HEAPSELECT_SHIFT, HEAPSELECT_CONSTANT, HEAPSELECT_MASK_SHIFT,
            SORTSELECT_CONSTANT);
    }

    /**
     * Constructor with specified quickselect size.
     *
     * <p>Used to test key analysis based on the quickselect size.
     *
     * @param minQuickSelectSize Minimum size for quickselect.
     */
    Partition(int minQuickSelectSize) {
        this(PIVOTING_STRATEGY, DUAL_PIVOTING_STRATEGY, minQuickSelectSize,
            HEAPSELECT_SHIFT, HEAPSELECT_CONSTANT, HEAPSELECT_MASK_SHIFT,
            SORTSELECT_CONSTANT);
    }

    /**
     * Constructor with specified pivoting strategy and quickselect size.
     *
     * <p>Used to test single-pivot quicksort.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     */
    Partition(PivotingStrategy pivotingStrategy, int minQuickSelectSize) {
        this(pivotingStrategy, DUAL_PIVOTING_STRATEGY, minQuickSelectSize,
            HEAPSELECT_SHIFT, HEAPSELECT_CONSTANT, HEAPSELECT_MASK_SHIFT,
            SORTSELECT_CONSTANT);
    }

    /**
     * Constructor with specified pivoting strategy and quickselect size.
     *
     * <p>Used to test dual-pivot quicksort.
     *
     * @param dualPivotingStrategy Dual pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     */
    Partition(DualPivotingStrategy dualPivotingStrategy, int minQuickSelectSize) {
        this(PIVOTING_STRATEGY, dualPivotingStrategy, minQuickSelectSize,
            HEAPSELECT_SHIFT, HEAPSELECT_CONSTANT, HEAPSELECT_MASK_SHIFT,
            SORTSELECT_CONSTANT);
    }

    /**
     * Constructor with specified pivoting strategy; quickselect size; and heapselect configuration.
     *
     * <p>Used to test single-pivot quickselect.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param heapSelectShift Length shift used for heap select distance from end threshold.
     * @param heapSelectConstant Length constant used for heap select distance from end threshold.
     * @param sortSelectConstant Length constant used for sort select distance from end threshold.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(PivotingStrategy pivotingStrategy,
        int minQuickSelectSize, int heapSelectShift,
        int heapSelectConstant, int sortSelectConstant) {
        this(pivotingStrategy, DUAL_PIVOTING_STRATEGY, minQuickSelectSize, heapSelectShift, heapSelectConstant,
            HEAPSELECT_MASK_SHIFT, sortSelectConstant);
    }

    /**
     * Constructor with specified dual-pivoting strategy; quickselect size; and heapselect configuration.
     *
     * <p>Used to test dual-pivot quickselect.
     *
     * @param dualPivotingStrategy Dual pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param heapSelectShift Length shift used for heap select distance from end threshold.
     * @param heapSelectConstant Length constant used for heap select distance from end threshold.
     * @param heapSelectMaskShift Shift applied to {@link Integer#MAX_VALUE} to mask the heap select dynamic distance from end threshold.
     * @param sortSelectConstant Length constant used for sort select distance from end threshold.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(DualPivotingStrategy dualPivotingStrategy,
        int minQuickSelectSize, int heapSelectShift,
        int heapSelectConstant, int heapSelectMaskShift,
        int sortSelectConstant) {
        this(PIVOTING_STRATEGY, dualPivotingStrategy, minQuickSelectSize, heapSelectShift,
            heapSelectConstant, heapSelectMaskShift, sortSelectConstant);
    }

    /**
     * Constructor with specified pivoting strategy; quickselect size; and heapselect configuration.
     *
     * <p>Heap select configuration is used to compute the {@code distance} from the end of the
     * current range {@code n} where heap select can be used:
     * <pre>
     * distance = (n >>> shift) + c
     * </pre>
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param dualPivotingStrategy Dual pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param heapSelectShift Length shift used for heap select distance from end threshold.
     * @param heapSelectConstant Length constant used for heap select distance from end threshold.
     * @param heapSelectMaskShift Shift applied to {@link Integer#MAX_VALUE} to mask the heap select dynamic distance from end threshold.
     * @param sortSelectConstant Length constant used for sort select distance from end threshold.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(PivotingStrategy pivotingStrategy, DualPivotingStrategy dualPivotingStrategy,
        int minQuickSelectSize, int heapSelectShift,
        int heapSelectConstant, int heapSelectMaskShift,
        int sortSelectConstant) {
        // Shift only uses lowest 5 bits. It should use [0, 31].
        // If bits outside this are set the shift is invalid.
        if ((heapSelectShift & ~31) != 0) {
            throw new IllegalArgumentException("Invalid shift: " + heapSelectShift);
        }
        this.pivotingStrategy = pivotingStrategy;
        this.dualPivotingStrategy = dualPivotingStrategy;
        this.minQuickSelectSize = minQuickSelectSize;
        this.heapSelectShift = heapSelectShift;
        this.heapSelectConstant = heapSelectConstant;
        this.heapSelectDynamicMask = Integer.MAX_VALUE >>> heapSelectMaskShift;
        this.sortSelectConstant = sortSelectConstant;
    }

    /**
     * Sets the key strategy.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setKeyStrategy(KeyStrategy v) {
        this.keyStrategy = v;
        return this;
    }

    /**
     * Sets the paired key strategy.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setPairedKeyStrategy(PairedKeyStrategy v) {
        this.pairedKeyStrategy = v;
        return this;
    }

    /**
     * Sets the recursion multiple.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setRecursionMultiple(double v) {
        this.recursionMultiple = v;
        return this;
    }

    /**
     * Sets the recursion constant.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setRecursionConstant(int v) {
        this.recursionConstant = v;
        return this;
    }

    /**
     * Sets the compression for a {@link CompressedIndexSet}.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setCompression(int v) {
        if (v < 1 || v > Integer.SIZE - 1) {
            throw new IllegalArgumentException("Bad compression: " + v);
        }
        this.compression = v;
        return this;
    }

    /**
     * Sets the recursion consumer. This is called with the value of the recursion
     * counter immediately before the introselect routine returns.
     *
     * @param v Value.
     */
    public void setRecursionConsumer(IntConsumer v) {
        this.recursionConsumer = Objects.requireNonNull(v);
    }

    /**
     * Move the minimum value to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * <p>Assumes {@code left <= right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMin(double[] data, int left, int right) {
        selectMinIgnoreZeros(data, left, right);
        // Edge-case: if min was 0.0, check for a -0.0 above and swap.
        if (data[left] == 0) {
            minZero(data, left, right);
        }
    }

    /**
     * Move the maximum value to the end of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Respects the ordering of signed zeros.
     *
     * <p>Assumes {@code left <= right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMax(double[] data, int left, int right) {
        selectMaxIgnoreZeros(data, left, right);
        // Edge-case: if max was -0.0, check for a 0.0 below and swap.
        if (data[right] == 0) {
            maxZero(data, left, right);
        }
    }

    /**
     * Place a negative signed zero at {@code left} before any positive signed zero in the range,
     * {@code -0.0 < 0.0}.
     *
     * <p>Warning: Only call when {@code data[left]} is zero.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    private static void minZero(double[] data, int left, int right) {
        // Assume data[left] is zero and check the sign bit
        if (Double.doubleToRawLongBits(data[left]) >= 0) {
            // Check for a -0.0 above and swap.
            // We only require 1 swap as this is not a full sort of zeros.
            for (int k = left; ++k <= right;) {
                if (data[k] == 0 && Double.doubleToRawLongBits(data[k]) < 0) {
                    data[k] = 0.0;
                    data[left] = -0.0;
                    break;
                }
            }
        }
    }

    /**
     * Place a positive signed zero at {@code right} after any negative signed zero in the range,
     * {@code -0.0 < 0.0}.
     *
     * <p>Warning: Only call when {@code data[right]} is zero.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    private static void maxZero(double[] data, int left, int right) {
        // Assume data[right] is zero and check the sign bit
        if (Double.doubleToRawLongBits(data[right]) < 0) {
            // Check for a 0.0 below and swap.
            // We only require 1 swap as this is not a full sort of zeros.
            for (int k = right; --k >= left;) {
                if (data[k] == 0 && Double.doubleToRawLongBits(data[k]) >= 0) {
                    data[k] = -0.0;
                    data[right] = 0.0;
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
     * <p>Assumes {@code left <= right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMinIgnoreZeros(double[] data, int left, int right) {
        // Mitigate worst case performance on descending data by backward sweep
        double min = data[left];
        for (int i = right + 1; --i > left;) {
            final double v = data[i];
            if (v < min) {
                data[i] = min;
                min = v;
            }
        }
        data[left] = min;
    }

    /**
     * Move the two smallest values to the start of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * <p>Assumes {@code left < right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMin2IgnoreZeros(double[] data, int left, int right) {
        double min1 = data[left + 1];
        if (min1 < data[left]) {
            min1 = data[left];
            data[left] = data[left + 1];
        }
        // Mitigate worst case performance on descending data by backward sweep
        for (int i = right + 1, end = left + 1; --i > end;) {
            final double v = data[i];
            if (v < min1) {
                data[i] = min1;
                if (v < data[left]) {
                    min1 = data[left];
                    data[left] = v;
                } else {
                    min1 = v;
                }
            }
        }
        data[left + 1] = min1;
    }

    /**
     * Move the maximum value to the end of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * <p>Assumes {@code left <= right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMaxIgnoreZeros(double[] data, int left, int right) {
        // Mitigate worst case performance on descending data by backward sweep
        double max = data[right];
        for (int i = left - 1; ++i < right;) {
            final double v = data[i];
            if (v > max) {
                data[i] = max;
                max = v;
            }
        }
        data[right] = max;
    }

    /**
     * Move the two largest values to the end of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respect the ordering of signed zeros.
     *
     * <p>Assumes {@code left < right}.
     *
     * @param data Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void selectMax2IgnoreZeros(double[] data, int left, int right) {
        double max1 = data[right - 1];
        if (max1 > data[right]) {
            max1 = data[right];
            data[right] = data[right - 1];
        }
        // Mitigate worst case performance on descending data by backward sweep
        for (int i = left - 1, end = right - 1; ++i < end;) {
            final double v = data[i];
            if (v > max1) {
                data[i] = max1;
                if (v > data[right]) {
                    max1 = data[right];
                    data[right] = v;
                } else {
                    max1 = v;
                }
            }
        }
        data[right - 1] = max1;
    }

    /**
     * Sort the elements using a heap sort algorithm.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void heapSort(double[] a, int left, int right) {
        // We could make a choice here
        heapSelectLeft(a, left, right, right, right - left);
        //heapSelectRight(a, left, right, left, right - left);
    }

    /**
     * Partition the elements {@code ka} and {@code kb} using a heap select algorithm. It
     * is assumed {@code left <= ka <= kb <= right}. Any range between the two elements is
     * not ensured to be sorted.
     *
     * <p>If there is no range between the two point, i.e. {@code ka == kb} or
     * {@code ka + 1 == kb}, it is preferred to use
     * {@link #heapSelectRange(double[], int, int, int, int)}. The result is the same but
     * the decision choice is simpler for the range function.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     * @see #heapSelectRange(double[], int, int, int, int)
     */
    static void heapSelectPair(double[] a, int left, int right, int ka, int kb) {
        // Avoid the overhead of heap select on tiny data (supports right <= left).
        if (right - left < MIN_HEAPSELECT_SIZE) {
            Sorting.sort(a, left, right);
            return;
        }
        // Call the appropriate heap partition function based on
        // building a heap up to 50% of the length
        // |l|-----|ka|--------|kb|------|r|
        //  ---d1----
        //                      -----d3----
        //  ---------d2----------
        //          ----------d4-----------
        final int d1 = ka - left;
        final int d2 = kb - left;
        final int d3 = right - kb;
        final int d4 = right - ka;
        if (d1 + d3 < Math.min(d2, d4)) {
            // Partition both ends.
            // Note: Not possible if ka == kb.
            // s1 + s3 == r - l and >= than the smallest
            // distance to one of the ends
            heapSelectLeft(a, left, right, ka, 0);
            // Repeat for the other side above ka
            heapSelectRight(a, ka + 1, right, kb, 0);
        } else if (d2 < d4) {
            heapSelectLeft(a, left, right, kb, kb - ka);
        } else {
            // s4
            heapSelectRight(a, left, right, ka, kb - ka);
        }
    }

    /**
     * Partition the elements between {@code ka} and {@code kb} using a heap select
     * algorithm. It is assumed {@code left <= ka <= kb <= right}.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     * @see #heapSelectPair(double[], int, int, int, int)
     */
    static void heapSelectRange(double[] a, int left, int right, int ka, int kb) {
        // Combine the test for right <= left with
        // avoiding the overhead of heap select on tiny data.
        if (right - left < MIN_HEAPSELECT_SIZE) {
            Sorting.sort(a, left, right);
            return;
        }
        // Call the appropriate heap partition function based on
        // building a heap up to 50% of the length
        // |l|-----|ka|--------|kb|------|r|
        // |---------d1-----------|
        //         |----------d2-----------|
        // Note: Optimisation for small heap size (n=1,2) is negligible.
        // The main overhead is the test for insertion against the current top of the heap
        // which grows increasingly unlikely as the range is scanned.
        if (kb - left < right - ka) {
            heapSelectLeft(a, left, right, kb, kb - ka);
        } else {
            heapSelectRight(a, left, right, ka, kb - ka);
        }
    }

    // TODO - update to heapselect a range [ka, kb]

    /**
     * Partition the minimum {@code n} elements below {@code k} where
     * {@code n = k - left + 1}. Uses a heap select algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and can be used to perform a full sort of the range below {@code k}
     * using the {@code count} parameter.
     *
     * <p>For best performance this should be called with
     * {@code k - left < right - k}, i.e.
     * to partition a value in the lower half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     * @param count Size of range to sort below k.
     */
    static void heapSelectLeft(double[] a, int left, int right, int k, int count) {
        // Create a max heap in-place in [left, k], rooted at a[left] = max
        // |l|-max-heap-|k|--------------|
        // Build the heap using Floyd's heap-construction algorithm for heap size n.
        // Start at parent of the last element in the heap (k),
        // i.e. start = parent(n-1) : parent(c) = floor((c - 1) / 2) : c = k - left
        int end = k + 1;
        for (int p = left + ((k - left - 1) >> 1); p >= left; p--) {
            maxHeapSiftDown(a, a[p], p, left, end);
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double max = a[left];
        for (int i = right + 1; --i > k;) {
            final double v = a[i];
            if (v < max) {
                a[i] = max;
                maxHeapSiftDown(a, v, left, left, end);
                max = a[left];
            }
        }

        // To partition elements k (and below) move the top of the heap to the position
        // immediately after the end of the reduced size heap; the previous end
        // of the heap [k] is placed at the top
        // |l|-max-heap-|k|--------------|
        //  |  <-swap->  |
        // The heap can be restored by sifting down the new top.

        // Always require the top 1
        a[left] = a[k];
        a[k] = max;

        if (count > 0) {
            --end;
            // Sifting limited to heap size of 2 (i.e. don't sift heap n==1)
            for (int c = Math.min(count, end - left - 1); --c >= 0;) {
                maxHeapSiftDown(a, a[left], left, left, end--);
                // Move top of heap to the sorted end
                max = a[left];
                a[left] = a[end];
                a[end] = max;
            }
        }
    }

    /**
     * Sift the element down the max heap.
     *
     * <p>Assumes {@code root <= p < end}, i.e. the max heap is above root.
     *
     * @param a Heap data.
     * @param v Value to sift.
     * @param p Start position.
     * @param root Root of the heap.
     * @param end End of the heap (exclusive).
     */
    private static void maxHeapSiftDown(double[] a, double v, int p, int root, int end) {
        // child2 = root + 2 * (parent - root) + 2
        //        = 2 * parent - root + 2
        while (true) {
            // Right child
            int c = (p << 1) - root + 2;
            if (c > end) {
                // No left child
                break;
            }
            // Use the left child if it is greater, or right doesn't exist
            if (c == end || a[c] < a[c - 1]) {
                --c;
            }
            if (v >= a[c]) {
                // Parent greater than largest child - done
                break;
            }
            // Swap and descend
            a[p] = a[c];
            p = c;
        }
        a[p] = v;
    }

    /**
     * Partition the maximum {@code n} elements above {@code k} where
     * {@code n = right - k + 1}. Uses a heap select algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and can be used to perform a full sort of the range above {@code k}
     * using the {@code count} parameter.
     *
     * <p>For best performance this should be called with
     * {@code k - left > right - k}, i.e.
     * to partition a value in the upper half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     * @param count Size of range to sort below k.
     */
    static void heapSelectRight(double[] a, int left, int right, int k, int count) {
        // Create a min heap in-place in [k, right], rooted at a[right] = min
        // |--------------|k|-min-heap-|r|
        // Build the heap using Floyd's heap-construction algorithm for heap size n.
        // Start at parent of the last element in the heap (k),
        // i.e. start = parent(n-1) : parent(c) = floor((c - 1) / 2) : c = right - k
        int end = k - 1;
        for (int p = right - ((right - k - 1) >> 1); p <= right; p++) {
            minHeapSiftDown(a, a[p], p, right, end);
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double min = a[right];
        for (int i = left - 1; ++i < k;) {
            final double v = a[i];
            if (v > min) {
                a[i] = min;
                minHeapSiftDown(a, v, right, right, end);
                min = a[right];
            }
        }

        // To partition elements k (and above) move the top of the heap to the position
        // immediately before the end of the reduced size heap; the previous end
        // of the heap [k] is placed at the top.
        // |--------------|k|-min-heap-|r|
        //                 |  <-swap->  |
        // The heap can be restored by sifting down the new top.

        // Always require the top 1
        a[right] = a[k];
        a[k] = min;

        if (count > 0) {
            ++end;
            // Sifting limited to heap size of 2 (i.e. don't sift heap n==1)
            for (int c = Math.min(count, right - end - 1); --c >= 0;) {
                minHeapSiftDown(a, a[right], right, right, end++);
                // Move top of heap to the sorted end
                min = a[right];
                a[right] = a[end];
                a[end] = min;
            }
        }
    }

    /**
     * Sift the element down the min heap.
     *
     * <p>Assumes {@code root >= p > end}, i.e. the max heap is below root.
     *
     * @param a Heap data.
     * @param v Value to sift.
     * @param p Start position.
     * @param root Root of the heap.
     * @param end End of the heap (exclusive).
     */
    private static void minHeapSiftDown(double[] a, double v, int p, int root, int end) {
        // child2 = root - 2 * (root - parent) - 2
        //        = 2 * parent - root - 2
        while (true) {
            // Right child
            int c = (p << 1) - root - 2;
            if (c < end) {
                break;
            }
            // Use the left child if it is less, or right doesn't exist
            if (c == end || a[c] > a[c + 1]) {
                ++c;
            }
            if (v <= a[c]) {
                // Parent less than smallest child - done
                break;
            }
            // Swap and descend
            a[p] = a[c];
            p = c;
        }
        a[p] = v;
    }

    /**
     * Partition the elements between {@code ka} and {@code kb} using a heap select
     * algorithm. It is assumed {@code left <= ka <= kb <= right}.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     * @see #heapSelectPair(double[], int, int, int, int)
     */
    static void heapSelectRange2(double[] a, int left, int right, int ka, int kb) {
        // Combine the test for right <= left with
        // avoiding the overhead of heap select on tiny data.
        if (right - left < MIN_HEAPSELECT_SIZE) {
            Sorting.sort(a, left, right);
            return;
        }
        // Use the smallest heap
        if (kb - left < right - ka) {
            heapSelectLeft2(a, left, right, ka, kb);
        } else {
            heapSelectRight2(a, left, right, ka, kb);
        }
    }

    /**
     * Partition the elements between {@code ka} and {@code kb} using a heap select
     * algorithm. It is assumed {@code left <= ka <= kb <= right}.
     *
     * <p>For best performance this should be called with {@code k} in the lower
     * half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     */
    static void heapSelectLeft2(double[] a, int left, int right, int ka, int kb) {
        // Create a max heap in-place in [left, k], rooted at a[left] = max
        // |l|-max-heap-|k|--------------|
        // Build the heap using Floyd's heap-construction algorithm for heap size n.
        // Start at parent of the last element in the heap (k),
        // i.e. start = parent(n-1) : parent(c) = floor((c - 1) / 2) : c = k - left
        int end = kb + 1;
        for (int p = left + ((kb - left - 1) >> 1); p >= left; p--) {
            maxHeapSiftDown(a, a[p], p, left, end);
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double max = a[left];
        for (int i = right + 1; --i > kb;) {
            final double v = a[i];
            if (v < max) {
                a[i] = max;
                maxHeapSiftDown(a, v, left, left, end);
                max = a[left];
            }
        }
        // Partition [ka, kb]
        // |l|-max-heap-|k|--------------|
        //  |  <-swap->  |   then sift down reduced size heap
        // Avoid sifting heap of size 1
        final int last = Math.max(left, ka - 1);
        while (--end > last) {
            maxHeapSiftDown(a, a[end], left, left, end);
            a[end] = max;
            max = a[left];
        }
    }

    /**
     * Partition the elements between {@code ka} and {@code kb} using a heap select
     * algorithm. It is assumed {@code left <= ka <= kb <= right}.
     *
     * <p>For best performance this should be called with {@code k} in the upper
     * half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     */
    static void heapSelectRight2(double[] a, int left, int right, int ka, int kb) {
        // Create a min heap in-place in [k, right], rooted at a[right] = min
        // |--------------|k|-min-heap-|r|
        // Build the heap using Floyd's heap-construction algorithm for heap size n.
        // Start at parent of the last element in the heap (k),
        // i.e. start = parent(n-1) : parent(c) = floor((c - 1) / 2) : c = right - k
        int end = ka - 1;
        for (int p = right - ((right - ka - 1) >> 1); p <= right; p++) {
            minHeapSiftDown(a, a[p], p, right, end);
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double min = a[right];
        for (int i = left - 1; ++i < ka;) {
            final double v = a[i];
            if (v > min) {
                a[i] = min;
                minHeapSiftDown(a, v, right, right, end);
                min = a[right];
            }
        }
        // Partition [ka, kb]
        // |--------------|k|-min-heap-|r|
        //                 |  <-swap->  |   then sift down reduced size heap
        // Avoid sifting heap of size 1
        final int last = Math.min(right, kb + 1);
        while (++end < last) {
            minHeapSiftDown(a, a[end], right, right, end);
            a[end] = min;
            min = a[right];
        }
    }

    /**
     * Partition the elements between {@code ka} and {@code kb} using a sort select
     * algorithm. It is assumed {@code left <= ka <= kb <= right}.
     *
     * <p>Note: Requires that the range contains no NaN values. Does not respects the
     * ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     */
    static void sortSelectRange(double[] a, int left, int right, int ka, int kb) {
        // Combine the test for right <= left with
        // avoiding the overhead of sort select on tiny data.
        if (right - left <= MIN_SORTSELECT_SIZE) {
            Sorting.sort(a, left, right);
            return;
        }
        // Sort the smallest side
        // |l|-----|ka|--------|kb|------|r|
        if (kb - left < right - ka) {
            sortSelectLeft(a, left, right, kb);
        } else {
            sortSelectRight(a, left, right, ka);
        }
    }

    /**
     * Partition the minimum {@code n} elements below {@code k} where
     * {@code n = k - left + 1}. Uses an insertion sort algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and performs a full sort of the range below {@code k}.
     *
     * <p>For best performance this should be called with
     * {@code k - left < right - k}, i.e.
     * to partition a value in the lower half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     */
    static void sortSelectLeft(double[] a, int left, int right, int k) {
        // Sort
        for (int i = left; ++i <= k;) {
            final double v = a[i];
            // Move preceding higher elements above (if required)
            if (v < a[i - 1]) {
                int j = i;
                while (--j >= left && v < a[j]) {
                    a[j + 1] = a[j];
                }
                a[j + 1] = v;
            }
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double m = a[k];
        for (int i = right + 1; --i > k;) {
            final double v = a[i];
            if (v < m) {
                a[i] = m;
                int j = k;
                while (--j >= left && v < a[j]) {
                    a[j + 1] = a[j];
                }
                a[j + 1] = v;
                m = a[k];
            }
        }
    }

    /**
     * Partition the maximum {@code n} elements above {@code k} where
     * {@code n = right - k + 1}. Uses an insertion sort algorithm.
     *
     * <p>Works with any {@code k} in the range {@code left <= k <= right}
     * and can be used to perform a full sort of the range above {@code k}.
     *
     * <p>For best performance this should be called with
     * {@code k - left > right - k}, i.e.
     * to partition a value in the upper half of the range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * Does not respects the ordering of signed zeros.
     *
     * @param a Data array to use to find out the K<sup>th</sup> value.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Index to select.
     */
    static void sortSelectRight(double[] a, int left, int right, int k) {
        // Sort
        for (int i = right; --i >= k;) {
            final double v = a[i];
            // Move succeeding lower elements below (if required)
            if (v > a[i + 1]) {
                int j = i;
                while (++j <= right && v > a[j]) {
                    a[j - 1] = a[j];
                }
                a[j - 1] = v;
            }
        }
        // Scan the remaining data and insert
        // Mitigate worst case performance on descending data by backward sweep
        double m = a[k];
        for (int i = left - 1; ++i < k;) {
            final double v = a[i];
            if (v > m) {
                a[i] = m;
                int j = k;
                while (++j <= right && v > a[j]) {
                    a[j - 1] = a[j];
                }
                a[j - 1] = v;
                m = a[k];
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
     * <p>Note: This is the only method in this class not based on introselect.
     * This is a legacy method containing alternatives for iterating over
     * multiple keys that are not supported by introselect, namely:
     *
     * <ul>
     * <li>{@link KeyStrategy#SEQUENTIAL}: This method concatenates close indices into
     * ranges and processes them together. It should be able to identify ranges that
     * require a full sort. Start-up cost is higher. In practice the indices do not saturate
     * the range if the length is reasonable and it is typically possible to cut between indices
     * during partitioning to create regions that do not require visiting. Thus trying to identify
     * regions for a full sort is a waste of resources.
     * <li>{@link KeyStrategy#INDEX_SET}: Uses a {@code BitSet}-type structure to store
     * pivots during call to partition. These can be used to bracket the search for the next index.
     * Storage is inefficient as it will require up to length bits of the memory of the input array
     * length even if the distribution is very sparse. Sparse ranges cannot be efficiently searched.
     * <li>{@link KeyStrategy#PIVOT_CACHE}: The {@link PivotCache} interface abstracts
     * methods from a {@code BitSet}. Indices can be stored and searched. The abstraction allows
     * the pivots to be stored efficiently. However there are no sparse implementations
     * of the interface other than 1 or 2 points. So performance is similar to the INDEX_SET
     * method. One difference is the method finds the outer indices first and then
     * only searches the internal region for the rest of the indices. This makes no difference
     * to performance.
     * </ul>
     *
     * <p>Note: In each method indices are processed independently. Thus each bracket around an
     * index to partition does not know the number of recursion steps used to obtain the start
     * pivots defining the bracket. Excess recursion cannot be efficiently tracked for each
     * partition. This is unlike introselect which tracks recursion and can switch to heapselect
     * if quickselect convergence is slow.
     *
     * <p>Benchmarking can be used to show these alternatives are slower.
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
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]} as defined by minimum and maximum index.
     *
     * @param indices Indices.
     * @param n Count of indices (must be strictly positive).
     * @return the pivot cache
     */
    private static PivotCache createPivotCacheForIndices(int[] indices, int n) {
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
     * Performs an analysis of keys to determine if they saturate the range to partition.
     * Returns {@code true} if a full sort is recommended.
     *
     * <p>This method is used to avoid the overhead of partitioning when there are so many
     * keys that they effectively cover the entire range. In this case it is far simpler
     * to use {@code Arrays.sort}. This method has to know under what conditions the
     * partition algorithm is outperformed by sorting all the data by the JDK sort
     * function.
     *
     * <p>The {@code saturation} parameter is the fraction of the range that must be
     * partitioned to recommend a full sort.
     *
     * <p>The approach is to first assume that keys could be uniformly spaced through the
     * range. If the number of keys could not cover the entire range given a minimum
     * spacing then this returns {@code false}. A small number of keys is also ignored as
     * the analysis of the saturation level consumes time and resources likely to be
     * larger than the difference between a full sort, and a full sort via the partition
     * algorithm.
     *
     * <p>If the keys could cover the range then the range of the keys is obtained
     * (min/max). If the length of the range is too small to saturate the range of the
     * data this returns {@code false}.
     *
     * <p>Otherwise keys are compressed by a power of 2 and recorded into a BitSet-type
     * structure. If the cardinality of the BitSet when decompressed is close to the
     * length of the data then the keys are identified as saturated.
     *
     * <p>The minimum spacing and compression use the same {@code compression} argument.
     *
     * <p>Example using data of size 20, 4 minimum keys, a compression level of 1 and
     * 0.9 saturation. Here compression is visualised by deleting every other index after
     *
     * <pre>
     * -------k------------   false: not enough keys: 1 &lt; 4
     *
     * ---k---kk----k------   false: keys cannot saturate the range: 4 &lt; (20 / 2)
     *
     * --kkkkkkkk----------   false: range of keys cannot saturate the range: (9 - 2 + 1) &lt; 0.9 * 20
     *
     * -k-k---k---k---k---k   false: compressed keys do not saturate the range:
     *  cc-c-c-c-c            6 * 2 &lt; 0.9 * 20
     *
     * -k-k-k-k-k-k-k-k-k-k   true: compressed keys saturate the range)
     *  cccccccccc            10 * 2 &gt; 0.9 * 20
     *
     * kkkkk-kkkkkkkkkkk-kk   true: compressed keys saturate the range)
     * cccccccccc             10 * 2 &gt; 0.9 * 20
     * </pre>
     *
     * @param size Length of the data to partition.
     * @param k Indices.
     * @param n Count of indices (must be strictly positive).
     * @param minKeys Minimum number of keys.
     * @param compression Compression level (log2 units in [1, 31]).
     * @param saturation Saturation level for a full sort.
     * @return true if the keys saturate the range
     */
    // package-private for testing
    static boolean keysAreSaturated(int size, int[] k, int n, int minKeys,
        int compression, double saturation) {
        // Check if the number of keys are small, or if they could saturated the range
        if (k.length < Math.max(minKeys, size >>> compression)) {
            return false;
        }
        // Keys could cover the entire data.
        // Set the limit on the number of indices that have to be sorted.
        final double limit = size * saturation;
        // Check the range.
        int min = k[0];
        int max = min;
        for (int i = 0; ++i < n;) {
            min = Math.min(min, k[i]);
            max = Math.max(max, k[i]);
        }
        if ((max - min + 1) < limit) {
            return false;
        }
        // Compress
        min >>>= compression;
        max >>>= compression;
        final IndexSet keys = IndexSet.ofRange(min, max);
        for (final int i : k) {
            keys.set(i >>> compression);
        }
        // Estimate number of indices to be sorted
        final long target = (long) keys.cardinality() << compression;
        return target >= limit;
    }

    /**
     * Test if the {@code keys} are saturated.
     *
     * @param keys Keys.
     * @param n Count of indices.
     * @return true if saturated
     */
    private static boolean keysAreSaturated(UpdatingInterval keys, int n) {
        // Only perform saturation analysis if is possible to saturate the range
        if (!(keys instanceof IntervalAnalysis) ||
            n < Math.max(SATURATION_ANALYSIS_COUNT, (keys.right() - keys.left()) >>> 3)) {
            return false;
        }
        return ((IntervalAnalysis) keys).saturated(3);
    }

    /**
     * Creates the {@link SearchableInterval} for the partition of data of the specified
     * {@code size}.
     *
     * <p>This method assesses the saturation of the indices given the {@code size} and
     * returns a suitable {@link SearchableInterval} for partitioning.
     *
     * <p>Returns {@code null} if the indices {@code k} saturate the {@code size}; this
     * occurs when partitioning will require visiting all regions of the data. In this
     * case a full sort of the data is recommended.
     *
     * <p>The heuristics used within this method may not always return the optimal
     * {@link SearchableInterval}. The method aims to avoid poor decisions, and recommend a
     * full sort when it is obvious that there are too many indices to efficiently
     * partition.
     *
     * <p>Partitioning
     *
     * <p>The partition algorithm should correctly sort all target indices between the
     * minimum ({@code k1}) and maximum ({@code kn}) index. During partitioning a sorted
     * point (@code pivot}) may cut the current interval. The partition algorithm then
     * requires updating the range of interest on either side of the cut point:
     *
     * <pre>{@code
     *           pivot
     *             |
     *        k1--------k2---------k3---- ... ---------kn    initial interval
     *         <--| find previous
     *    find next |-->
     *        k1        k2---------k3---- ... ---------kn    divided intervals
     * }</pre>
     *
     * <p>If a {@code pivot} is found in the interval then the smallest region of data
     * that was most recently partitioned was the length between the two flanking k. This
     * involves a full scan (and partitioning) over the data of length (k2 - k1). If the
     * {@link SearchableInterval} uses a BitSet-type structure it will require a scan over 1/64
     * of this length of data to find the next and previous index from a {@code pivot}
     * point. In practice the interval may be partitioned over a much larger length, e.g.
     * (kn - k1). Thus the length of time for the partition algorithm is expected to be at
     * least 64x the length of time for the BitSet-type scan. The disadvantage of the
     * BitSet-type structure is memory consumption. For a small number of keys a structure
     * that searches the entire set of keys is fast enough. However this requires that the
     * keys are unique and ordered.
     *
     * <p>This method will return an ordered interval of indices when {@code n} is small.
     * When {@code n} is large then a BitSet-type structure is returned. The maximum
     * memory consumption is approximately {@code size / 8} bytes.
     *
     * @param size Length of the data to partition.
     * @param k Indices.
     * @param n Count of indices (must be strictly positive).
     * @return the index interval (or {@code null} to recommend a full sort)
     */
    static SearchableInterval createSearchableInterval(int size, int[] k, int n) {
        if (size < MIN_QUICKSELECT_SIZE) {
            // Sort tiny data
            return null;
        }

        // The partition algorithm performs a full sort of data when any sub-length
        // is below 32. If keys are separated by at most 32 throughout the range
        // then the data is suitable for a full sort. However benchmarking shows that
        // partitioning with many indices performs close to the speed of a full sort
        // using the same quicksort algorithm; even when keys should be saturated,
        // e.g. 5000 random indices in length 10000. Thus switching to a full sort is
        // not performed until it is obvious the indices saturate the range.
        // Note that this method cannot know about any structure in the data.
        // Data that is structured (runs of continuous ascending/descending
        // data) will benefit from the additional algorithms within the JDK sort function
        // such as merge sort. The JDK sort function also supports parallel execution
        // which can outperform single-threaded partitioning depending on parallelism
        // and the spread of indices.

        // We check for saturation by compressing each index by 16-to-1. Any indices
        // separated by < 16 will be the same compressed index or adjacent compressed indices.
        // Any indices separated by [16, 32) may be compressed indices separated by 1 or 2.
        // If there are no gaps in the compressed indices then a full sort is recommended
        // as it is clear that all regions must be sorted. This heuristic avoids switching
        // to a full sort in all but the most obvious cases of saturation; however it may
        // choose to partition when a full sort would be faster.

        // Set the limit on the number of indices that have to be sorted.
        // Subtracting the quickselect size pads the min/max range of indices
        // at the ends.
        final int limit = size - MIN_QUICKSELECT_SIZE;

        // Use a shift with a long to avoid overflow (of an excessive number of keys !!!).
        // It is not expected to partition more than a few hundred keys.
        if (((long) n << LOG2_HALF_QUICKSELECT_SIZE) > limit) {
            // Keys could cover the entire data.
            final IndexSet keys = IndexSet.of(k, n);
            // Quick check if the range is smaller than the limit.
            if ((keys.right() - keys.left()) < limit) {
                return keys;
            }
            // Special cardinality count using 16-to-1 compression
            final int c = keys.cardinality16();
            return c < limit ? keys : null;
        }

        // This occurs when the indices cannot saturate the range.
        return IndexIntervals.createSearchableInterval(k, n);
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
            final int r = pivots.nextPivotOrElse(e, right + 1);

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
     * Sort the data by recursive partitioning (quicksort).
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

    /**
     * Sort the array using an introsort. The single-pivot partition method is provided as an argument.
     * Switches to heapsort when recursive partitioning reaches a maximum depth.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @see <a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
     */
    private void introsort(SPEPartition part, double[] a) {
        // Handle NaN / signed zeros
        final DoubleDataTransformer t = SORT_TRANSFORMER.get();
        // Assume this is in-place
        t.preProcess(a);
        final int end = t.length();
        if (end > 1) {
            introsort(part, a, 0, end - 1, createMaxDepthSinglePivot(end));
        }
        // Restore signed zeros
        t.postProcess(a);
    }

    /**
     * Sort the array.
     *
     * <p>Uses an introsort. The single-pivot partition method is provided as an argument.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros
     * may be destroyed (the mixture updated during partitioning). The caller is
     * responsible for counting a mixture of signed zeros and restoring them if
     * required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param maxDepth Maximum depth for recursion.
     * @see <a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
     */
    private void introsort(SPEPartition part, double[] a, int left, int right, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        final int l = left;
        int r = right;
        final int[] upper = {0};
        while (true) {
            // Full sort of small data
            if (r - l < minQuickSelectSize) {
                //Sorting.sort(a, l, r, l > 0);
                Sorting.sort(a, l, r);
                return;
            }
            if (maxDepth == 0) {
                // Too much recursion
                heapSort(a, l, r);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            // Recurse right side
            introsort(part, a, p1 + 1, r, --maxDepth);
            // Continue on the left side
            r = p0 - 1;
        }
    }

    /**
     * Sort the array using an introsort. The dual-pivot partition method is provided as an argument.
     * Switches to heapsort when recursive partitioning reaches a maximum depth.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @see <a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
     */
    private void introsort(DPPartition part, double[] a) {
        // Handle NaN / signed zeros
        final DoubleDataTransformer t = SORT_TRANSFORMER.get();
        // Assume this is in-place
        t.preProcess(a);
        final int end = t.length();
        if (end > 1) {
            introsort(part, a, 0, end - 1, createMaxDepthDualPivot(end));
        }
        // Restore signed zeros
        t.postProcess(a);
    }

    /**
     * Sort the array.
     *
     * <p>Uses an introsort. The dual-pivot partition method is provided as an argument.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros
     * may be destroyed (the mixture updated during partitioning). The caller is
     * responsible for counting a mixture of signed zeros and restoring them if
     * required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param maxDepth Maximum depth for recursion.
     * @see <a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
     */
    private void introsort(DPPartition part, double[] a, int left, int right, int maxDepth) {
        // Only two regions require recursion. The third region
        // can remain within this function call.
        final int l = left;
        int r = right;
        final int[] upper = {0, 0, 0};
        while (true) {
            // Full sort of small data
            if (r - l < minQuickSelectSize) {
                //Sorting.sort(a, l, r, l > 0);
                Sorting.sort(a, l, r);
                return;
            }
            if (maxDepth == 0) {
                // Too much recursion
                heapSort(a, l, r);
                return;
            }

            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            // Recurse middle and right sides
            --maxDepth;
            introsort(part, a, p3 + 1, r, maxDepth);
            introsort(part, a, p1 + 1, p2 - 1, maxDepth);
            // Continue on the left side
            r = p0 - 1;
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
     * <p>Uses an introselect variant. The dual-pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices (assumed to be strictly positive).
     */
    void introselect(SPEPartition part, double[] a, int[] k, int count) {
        // Handle NaN / signed zeros
        final DoubleDataTransformer t = SORT_TRANSFORMER.get();
        // Assume this is in-place
        t.preProcess(a);
        final int end = t.length();
        int n = count;
        if (end > 1) {
            // Filter indices invalidated by NaN check
            if (end < a.length) {
                for (int i = n; --i >= 0;) {
                    final int v = k[i];
                    if (v >= end) {
                        // swap(k, i, --n)
                        k[i] = k[--n];
                        k[n] = v;
                    }
                }
            }
            introselect(part, a, end - 1, k, n);
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
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
     * <p>Uses an introselect variant. The quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
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
     * @param a Values.
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Indices (may be destructively modified).
     * @param n Count of indices (assumed to be strictly positive).
     */
    private void introselect(SPEPartition part, double[] a, int right, int[] k, int n) {
        if (n < 1) {
            return;
        }
        final int maxDepth = createMaxDepthSinglePivot(right + 1);
        // Handle cases without multiple keys
        if (n == 1) {
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                // Dedicated method for a single key
                introselect(part, a, 0, right, k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two keys using the same key
                introselect(part, a, 0, right, k[0], k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.SEARCHABLE_INTERVAL) {
                // Reuse the IndexInterval method using the same key
                introselect(part, a, 0, right, IndexIntervals.anyIndex(), k[0], k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.UPDATING_INTERVAL) {
                // Reuse the Interval method using a single key
                introselect(part, a, 0, right, IndexIntervals.interval(k[0]), maxDepth);
            } else {
                throw new IllegalStateException(UNSUPPORTED_INTROSELECT + pairedKeyStrategy);
            }
            return;
        }
        // Special case for partition around adjacent indices (for interpolation)
        if (n == 2 && k[0] + 1 == k[1]) {
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                // Dedicated method for a single key, returns information about k+1
                final int p = introselect(part, a, 0, right, k[0], maxDepth);
                // p <= k to signal k+1 is unsorted, or p+1 is a pivot.
                // if k is sorted, and p+1 is sorted, k+1 is sorted if k+1 == p.
                if (p > k[1]) {
                    selectMinIgnoreZeros(a, k[1], p);
                }
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two keys
                // Note: This can handle keys that are not adjacent
                // e.g. keys near opposite ends without a partition step.
                final int ka = Math.min(k[0], k[1]);
                final int kb = Math.max(k[0], k[1]);
                introselect(part, a, 0, right, ka, kb, maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.SEARCHABLE_INTERVAL) {
                // Reuse the IndexInterval method using a range of two keys
                introselect(part, a, 0, right, IndexIntervals.anyIndex(), k[0], k[1], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.UPDATING_INTERVAL) {
                // Reuse the Interval method using a range of two keys
                introselect(part, a, 0, right, IndexIntervals.interval(k[0], k[1]), maxDepth);
            } else {
                throw new IllegalStateException(UNSUPPORTED_INTROSELECT + pairedKeyStrategy);
            }
            return;
        }

        // Detect possible saturated range.
        // minimum keys = 10
        // min separation = 2^3  (could use log2(minQuickSelectSize) here)
        // saturation = 0.95
        //if (keysAreSaturated(right + 1, k, n, 10, 3, 0.95)) {
        //    Arrays.sort(a, 0, right + 1);
        //    return;
        //}

        // Note: Sorting to unique keys is an overhead. This can be eliminated
        // by requesting the caller passes sorted keys (or quantiles in order).

        if (keyStrategy == KeyStrategy.ORDERED_KEYS) {
            final int unique = Sorting.sortIndices(k, n);
            introselect(part, a, 0, right, k, 0, unique - 1, maxDepth);
        } else if (keyStrategy == KeyStrategy.SCANNING_KEY_SEARCHABLE_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final SearchableInterval keys = ScanningKeyInterval.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.SEARCH_KEY_SEARCHABLE_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final SearchableInterval keys = BinarySearchKeyInterval.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.COMPRESSED_INDEX_SET) {
            // Note: Here we do not have to sort keys.
            final SearchableInterval keys = CompressedIndexSet.of(compression, k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_SET) {
            // Note: Here we do not have to sort keys.
            final SearchableInterval keys = IndexSet.of(k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.KEY_UPDATING_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final UpdatingInterval keys = KeyUpdatingInterval.of(k, unique);
            introselect(part, a, 0, right, keys, maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_SET_UPDATING_INTERVAL) {
            final UpdatingInterval keys = IndexSet.of(k, n).interval();
            introselect(part, a, 0, right, keys, maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_ITERATOR) {
            final int unique = Sorting.sortIndices(k, n);
            final IndexIterator keys = KeyIndexIterator.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.COMPRESSED_INDEX_ITERATOR) {
            final IndexIterator keys = CompressedIndexSet.iterator(compression, k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else {
            throw new IllegalStateException(UNSUPPORTED_INTROSELECT + keyStrategy);
        }
    }

    /**
     * Partition the array such that index {@code k} corresponds to its
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>Returns information {@code p} on whether {@code k+1} is sorted.
     * If {@code p <= k} then {@code k+1} is sorted.
     * If {@code p > k} then {@code p+1} is a pivot.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Index.
     * @param maxDepth Maximum depth for recursion.
     * @return the index {@code p}
     */
    private int introselect(SPEPartition part, double[] a, int left, int right,
        int k, int maxDepth) {
        int l = left;
        int r = right;
        final int[] upper = {0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d2----
            final int d1 = k - l;
            final int d2 = r - k;
            if (maxDepth == 0 || Math.min(d1, d2) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                // Too much recursion, or k is close to the end
                heapSelectPair(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            if (n < minQuickSelectSize || Math.min(d1, d2) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            maxDepth--;
            if (k < p0) {
                // The element is in the left partition
                r = p0 - 1;
            } else if (k > p1) {
                // The element is in the right partition
                l = p1 + 1;
            } else {
                // The range contains the element we wanted.
                // Signal if k+1 is sorted.
                // This can be true if the pivot was a range [p0, p1]
                return k < p1 ? k : r;
            }
        }
    }

    /**
     * Partition the array such that indices {@code ka} and {@code kb} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>Note: Requires {@code ka <= kb}. The use of two indices is to support processing
     * of pairs of indices {@code (k, k+1)}. However the indices are treated independently
     * and partitioned by recursion. They may be equal, neighbours or well separated.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka Index.
     * @param kb Index.
     * @param maxDepth Maximum depth for recursion.
     */
    private void introselect(SPEPartition part, double[] a, int left, int right,
        int ka, int kb, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        int l = left;
        int r = right;
        int ka1 = ka;
        int kb1 = kb;
        final int[] upper = {0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka1 and kb1 are close to the ends
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---d1----
            //                       -----d3----
            //  ---------d2-----------
            //          ----------d4-----------
            final int d1 = ka1 - l;
            final int d2 = kb1 - l;
            final int d3 = r - kb1;
            final int d4 = r - ka1;
            if (maxDepth == 0 ||
                Math.min(d1 + d3, Math.min(d2, d4)) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                // Too much recursion, or ka1 and kb1 are both close to the ends
                heapSelectPair(a, l, r, ka1, kb1);
                return;
            }

            if (n < minQuickSelectSize || Math.min(d2, d4) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                //Sorting.sort(a, l, r, l > 0);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the pair
            // and the recursion proceeds with a single point.
            maxDepth--;
            // Recurse left side if required
            if (ka1 < p0) {
                if (kb1 <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    kb1 = r < kb1 ? ka1 : kb1;
                    continue;
                }
                introselect(part, a, l, p0 - 1, ka1, ka1, maxDepth);
                ka1 = kb1;
            }
            if (kb1 <= p1) {
                // No right side
                return;
            }
            // Continue on the right side
            l = p1 + 1;
            ka1 = ka1 < l ? kb1 : ka1;
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts an ordered array of indices {@code k} and pointers
     * to the first and last positions in {@code k} that define the range indices
     * to partition.
     *
     * <pre>{@code
     * left <= k[ia] <= k[ib] <= right  : ia <= ib
     * }</pre>
     *
     * <p>A binary search is used to search for keys in {@code [ia, ib]}
     * to create {@code [ia, ib1]} and {@code [ia1, ib]} if partitioning splits the range.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Indices to partition (ordered).
     * @param ia Index of first key.
     * @param ib Index of last key.
     * @param maxDepth Maximum depth for recursion.
     */
    private void introselect(SPEPartition part, double[] a, int left, int right,
        int[] k, int ia, int ib, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        int l = left;
        int r = right;
        int ia1 = ia;
        int ib1 = ib;
        final int[] upper = {0};
        while (true) {
            // Switch to paired key implementation if possible.
            // Note: adjacent indices can refer to well separated keys.
            // This is the major difference between this implementation
            // and an implementation using an IndexInterval (which does not
            // have a fast way to determine if there are any keys within the range).
            if (ib1 - ia1 <= 1) {
                introselect(part, a, l, r, k[ia1], k[ib1], maxDepth);
                return;
            }

            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            int ka = k[ia1];
            final int kb = k[ib1];
            if (maxDepth == 0 ||
                Math.min(kb - l, r - ka) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                // Too much recursion, or ka and kb are both close to the same end
                heapSelectRange(a, l, r, ka, kb);
                return;
            }

            if (n < minQuickSelectSize || Math.min(kb - l, r - ka) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                //Sorting.sort(a, l, r, l > 0);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set on either side.
            //                   p0 p1
            // |l|--|ka|--k----k--|P|------k--|kb|------|r|
            //       ia1       iba  |      ia1  ib1
            // Search less/greater is bounded at ia1/ib1
            maxDepth--;
            // Recurse left side if required
            if (ka < p0) {
                if (kb <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb) {
                        ib1 = searchLessOrEqual(k, ia1, ib1, r);
                    }
                    continue;
                }
                // Require a split here
                introselect(part, a, l, p0 - 1, k, ia1, searchLessOrEqual(k, ia1, ib1, p0 - 1), maxDepth);
                ia1 = searchGreaterOrEqual(k, ia1, ib1, l);
                ka = k[ia1];
            }
            if (kb <= p1) {
                // No right side
                recursionConsumer.accept(maxDepth);
                return;
            }
            // Continue on the right side
            l = p1 + 1;
            if (ka < l) {
                ia1 = searchGreaterOrEqual(k, ia1, ib1, l);
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts a {@link SearchableInterval} of indices {@code k} and the
     * first index {@code ka} and last index {@code kb} that define the range of indices
     * to partition. The {@link SearchableInterval} is used to search for keys in {@code [ka, kb]}
     * to create {@code [ka, kb1]} and {@code [ka1, kb]} if partitioning splits the range.
     *
     * <pre>{@code
     * left <= ka <= kb <= right
     * }</pre>
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param ka First key.
     * @param kb Last key.
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(SPEPartition part, double[] a, int left, int right,
        SearchableInterval k, int ka, int kb, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        int l = left;
        int r = right;
        int ka1 = ka;
        int kb1 = kb;
        final int[] upper = {0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka and kb1 are close to the same end
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (maxDepth == 0 ||
                Math.min(kb1 - l, r - ka1) < Math.max((n >>> heapSelectShift) + heapSelectConstant,
                    heapSelectDynamicMask & heapSelectEdgeDistanceSP(n))) {
                // Too much recursion, or ka1 and kb1 are both close to the same end
                heapSelectRange(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (n < minQuickSelectSize || Math.min(kb1 - l, r - ka1) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                //Sorting.sort(a, l, r, l > 0);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set on either side.
            //                    p0 p1
            // |l|--|ka1|--k----k--|P|------k--|kb1|------|r|
            //                 kb1  |      ka1
            // Search previous/next is bounded at ka1/kb1
            maxDepth--;
            // Recurse left side if required
            if (ka1 < p0) {
                if (kb1 <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb1) {
                        kb1 = k.previousIndex(r);
                    }
                    continue;
                }
                introselect(part, a, l, p0 - 1, k, ka1, k.split(p0, p1, upper), maxDepth);
                ka1 = upper[0];
            }
            if (kb1 <= p1) {
                // No right side
                recursionConsumer.accept(maxDepth);
                return;
            }
            // Continue on the right side
            l = p1 + 1;
            if (ka1 < l) {
                ka1 = k.nextIndex(l);
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts a {@link UpdatingInterval} of indices {@code k} that define the
     * range of indices to partition. The {@link UpdatingInterval} can be narrowed or split as
     * partitioning divides the range.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(SPEPartition part, double[] a, int left, int right,
        UpdatingInterval k, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        int l = left;
        int r = right;
        int ka = k.left();
        int kb = k.right();
        final int[] upper = {0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (maxDepth == 0 ||
                Math.min(kb - l, r - ka) < Math.max((n >>> heapSelectShift) + heapSelectConstant,
                    heapSelectDynamicMask & heapSelectEdgeDistanceSP(n))) {
                // Too much recursion, or ka and kb are both close to the same end
                heapSelectRange(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (n < minQuickSelectSize || Math.min(kb - l, r - ka) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                //Sorting.sort(a, l, r, l > 0);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r),
                upper);
            final int p1 = upper[0];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set on either side.
            //                   p0 p1
            // |l|--|ka|--k----k--|P|------k--|kb|------|r|
            //                 kb  |       ka
            maxDepth--;
            // Recurse left side if required
            if (ka < p0) {
                if (kb <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb) {
                        kb = k.updateRight(r);
                    }
                    continue;
                }
                introselect(part, a, l, p0 - 1, k.splitLeft(p0, p1), maxDepth);
                ka = k.left();
            }
            if (kb <= p1) {
                // No right side
                recursionConsumer.accept(maxDepth);
                return;
            }
            // Continue on the right side
            l = p1 + 1;
            if (ka < l) {
                ka = k.updateLeft(l);
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts an {@link IndexIterator} of indices {@code k}; for
     * convenience the lower and upper indices of the current interval are passed as the
     * first index {@code ka} and last index {@code kb} of the closed interval of indices
     * to partition. These may be within the lower and upper indices if the interval was
     * split during recursion: {@code lower <= ka <= kb <= upper}.
     *
     * <p>The data is recursively partitioned using left-most ordering. When the current
     * interval has been partitioned the {@link IndexIterator} is used to advance to the
     * next interval to partition.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param ka First key.
     * @param kb Last key.
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(SPEPartition part, double[] a, int left, int right,
        IndexIterator k, int ka, int kb, int maxDepth) {
        // Left side requires recursion; right side remains within this function
        // When this function returns all indices in [left, right] must be processed.
        int l = left;
        int lo = ka;
        int hi = kb;
        final int[] upper = {0};
        while (true) {
            if (maxDepth == 0) {
                // Too much recursion.
                // Advance the iterator to the end of the current range.
                // Note: heapSelectRange handles hi > right.
                // Single API method: advanceBeyond(right): return hi <= right
                while (hi < right && k.next()) {
                    hi = k.right();
                }
                heapSelectRange(a, l, right, lo, hi);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // length - 1
            final int n = right - l;

            // If interval is close to one end then heapselect.
            // Only heapselect left if there are no further indices in the range.
            // |l|-----|lo|--------|hi|------|right|
            //  ---------d1----------
            //          --------------d2-----------
            if (Math.min(hi - l, right - lo) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                if (hi - l > right - lo) {
                    // Right end
                    heapSelectRight(a, l, right, lo, right - lo);
                    recursionConsumer.accept(maxDepth);
                    return;
                } else if (k.nextAfter(right)) {
                    // Left end
                    // Only if no further indices in the range.
                    // If false this branch will continue to be triggered until
                    // a partition is made to separate the next indices.
                    heapSelectLeft(a, l, right, hi, hi - l);
                    recursionConsumer.accept(maxDepth);
                    // Advance iterator
                    l = hi + 1;
                    if (!k.positionAfter(hi) || Math.max(k.left(), l) > right) {
                        // No more keys, or keys beyond the current bounds
                        return;
                    }
                    lo = Math.max(k.left(), l);
                    hi = Math.min(right, k.right());
                    // Continue right (allows a second heap select for the right side)
                    continue;
                }
            }

            // If interval is close to both ends then full sort
            // |l|-----|lo|--------|hi|------|right|
            //  ---d1----
            //                       ----d2--------
            // (lo - l) + (right - hi) == (right - l) - (hi - lo)
            if (n - (hi - lo) < minQuickSelectSize) {
                // Handle small data. This is done as the JDK sort will
                // use insertion sort for small data. For double data it
                // will also pre-process the data for NaN and signed
                // zeros which is an overhead to avoid.
                if (n < minQuickSelectSize) {
                    // Must not use sortSelectRange in [lo, hi] as the iterator
                    // has not been advanced to check after hi
                    sortSelectRight(a, l, right, lo);
                    //Sorting.sort(a, l, right, l > 0);
                } else {
                    // Note: This disregards the current level of recursion
                    // but can exploit the JDK's more advanced sort algorithm.
                    Arrays.sort(a, l, right + 1);
                }
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Here: l <= lo <= hi <= right
            // Pick a pivot and partition
            final int p0 = part.partition(a, l, right,
                pivotingStrategy.pivotIndex(a, l, right),
                upper);
            final int p1 = upper[0];

            maxDepth--;
            // Recursion left
            if (lo < p0) {
                introselect(part, a, l, p0 - 1, k, lo, Math.min(hi, p0 - 1), maxDepth);
                // Advance iterator
                // Single API method: fastForwardAndLeftWithin(p1, right)
                if (!k.positionAfter(p1) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }
            if (hi <= p1) {
                // Advance iterator
                if (!k.positionAfter(p1) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }
            // Continue right
            l = p1 + 1;
            lo = Math.max(lo, l);
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
     * <p>Uses an introselect variant. The dual pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices (assumed to be strictly positive).
     */
    void introselect(DPPartition part, double[] a, int[] k, int count) {
        // Handle NaN / signed zeros
        final DoubleDataTransformer t = SORT_TRANSFORMER.get();
        // Assume this is in-place
        t.preProcess(a);
        final int end = t.length();
        int n = count;
        if (end > 1) {
            // Filter indices invalidated by NaN check
            if (end < a.length) {
                for (int i = n; --i >= 0;) {
                    final int v = k[i];
                    if (v >= end) {
                        // swap(k, i, --n)
                        k[i] = k[--n];
                        k[n] = v;
                    }
                }
            }
            introselect(part, a, end - 1, k, n);
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
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
     * <p>Uses an introselect variant. The dual pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
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
     * @param a Values.
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Indices (may be destructively modified).
     * @param n Count of indices (assumed to be strictly positive).
     */
    private void introselect(DPPartition part, double[] a, int right, int[] k, int n) {
        if (n < 1) {
            return;
        }
        final int maxDepth = createMaxDepthDualPivot(right + 1);
        // Handle cases without multiple keys
        if (n == 1) {
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                // Dedicated method for a single key
                introselect(part, a, 0, right, k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two keys using the same key
                introselect(part, a, 0, right, k[0], k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.SEARCHABLE_INTERVAL) {
                // Reuse the IndexInterval method using the same key
                introselect(part, a, 0, right, IndexIntervals.anyIndex(), k[0], k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.UPDATING_INTERVAL) {
                // Reuse the Interval method using a single key
                introselect(part, a, 0, right, IndexIntervals.interval(k[0]), maxDepth);
            } else {
                throw new IllegalStateException(UNSUPPORTED_INTROSELECT + pairedKeyStrategy);
            }
            return;
        }
        // Special case for partition around adjacent indices (for interpolation)
        if (n == 2 && k[0] + 1 == k[1]) {
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                // Dedicated method for a single key, returns information about k+1
                final int p = introselect(part, a, 0, right, k[0], maxDepth);
                // p <= k to signal k+1 is unsorted, or p+1 is a pivot.
                // if k is sorted, and p+1 is sorted, k+1 is sorted if k+1 == p.
                if (p > k[1]) {
                    selectMinIgnoreZeros(a, k[1], p);
                }
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two keys
                // Note: This can handle keys that are not adjacent
                // e.g. keys near opposite ends without a partition step.
                final int ka = Math.min(k[0], k[1]);
                final int kb = Math.max(k[0], k[1]);
                introselect(part, a, 0, right, ka, kb, maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.SEARCHABLE_INTERVAL) {
                // Reuse the IndexInterval method using a range of two keys
                introselect(part, a, 0, right, IndexIntervals.anyIndex(), k[0], k[1], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.UPDATING_INTERVAL) {
                // Reuse the Interval method using a range of two keys
                introselect(part, a, 0, right, IndexIntervals.interval(k[0], k[1]), maxDepth);
            } else {
                throw new IllegalStateException(UNSUPPORTED_INTROSELECT + pairedKeyStrategy);
            }
            return;
        }

        // Detect possible saturated range.
        // minimum keys = 10
        // min separation = 2^3  (could use log2(minQuickSelectSize) here)
        // saturation = 0.95
        //if (keysAreSaturated(right + 1, k, n, 10, 3, 0.95)) {
        //    Arrays.sort(a, 0, right + 1);
        //    return;
        //}

        // Note: Sorting to unique keys is an overhead. This can be eliminated
        // by requesting the caller passes sorted keys (or quantiles in order).

        if (keyStrategy == KeyStrategy.ORDERED_KEYS) {
            // DP does not offer ORDERED_KEYS implementation but we include the branch
            // for completeness.
            throw new IllegalStateException(UNSUPPORTED_INTROSELECT + keyStrategy);
        } else if (keyStrategy == KeyStrategy.SCANNING_KEY_SEARCHABLE_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final SearchableInterval keys = ScanningKeyInterval.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.SEARCH_KEY_SEARCHABLE_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final SearchableInterval keys = BinarySearchKeyInterval.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.COMPRESSED_INDEX_SET) {
            // Note: Here we do not have to sort keys.
            final SearchableInterval keys = CompressedIndexSet.of(compression, k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_SET) {
            // Note: Here we do not have to sort keys.
            final SearchableInterval keys = IndexSet.of(k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.KEY_UPDATING_INTERVAL) {
            final int unique = Sorting.sortIndices(k, n);
            final UpdatingInterval keys = KeyUpdatingInterval.of(k, unique);
            introselect(part, a, 0, right, keys, maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_SET_UPDATING_INTERVAL) {
            final UpdatingInterval keys = IndexSet.of(k, n).interval();
            introselect(part, a, 0, right, keys, maxDepth);
        } else if (keyStrategy == KeyStrategy.INDEX_ITERATOR) {
            final int unique = Sorting.sortIndices(k, n);
            final IndexIterator keys = KeyIndexIterator.of(k, unique);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else if (keyStrategy == KeyStrategy.COMPRESSED_INDEX_ITERATOR) {
            final IndexIterator keys = CompressedIndexSet.iterator(compression, k, n);
            introselect(part, a, 0, right, keys, keys.left(), keys.right(), maxDepth);
        } else {
            throw new IllegalStateException(UNSUPPORTED_INTROSELECT + keyStrategy);
        }
    }

    /**
     * Partition the array such that index {@code k} corresponds to its
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>Returns information {@code p} on whether {@code k+1} is sorted.
     * If {@code p <= k} then {@code k+1} is sorted.
     * If {@code p > k} then {@code p+1} is a pivot.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Index.
     * @param maxDepth Maximum depth for recursion.
     * @return the index {@code p}
     */
    private int introselect(DPPartition part, double[] a, int left, int right,
        int k, int maxDepth) {
        int l = left;
        int r = right;
        final int[] upper = {0, 0, 0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d3----
            final int d1 = k - l;
            final int d3 = r - k;
            if (maxDepth == 0 || Math.min(d1, d3) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                // Too much recursion, or k is close to the end
                heapSelectPair(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            if (n < minQuickSelectSize || Math.min(d1, d3) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, k, k);
                //Sorting.sort(a, l, r, l > 0);
                // Since r+1 is a pivot, then k+1 is sorted
                return l;
            }

            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            maxDepth--;
            if (k < p0) {
                // The element is in the left partition
                r = p0 - 1;
                continue;
            } else if (k > p3) {
                // The element is in the right partition
                l = p3 + 1;
                continue;
            }
            // Check the interval overlaps the middle; and the middle exists.
            //                    p0 p1                p2 p3
            // |l|-----------------|P|------------------|P|----|r|
            // Eliminate:     ----kb1                    ka1----
            if (k <= p1 || p2 <= k || p2 - p1 <= 2) {
                // Signal if k+1 is sorted.
                // This can be true if the pivots were ranges [p0, p1] or [p2, p3]
                // This check will match *most* sorted k for the 3 eliminated cases.
                // It will not identify p2 - p1 <= 2 when k == p1. In this case
                // k+1 is sorted and a min-select for k+1 is a fast scan up to r.
                return k != p1 && k < p3 ? k : r;
            }
            // Continue in the middle partition
            l = p1 + 1;
            r = p2 - 1;
        }
    }

    /**
     * Partition the array such that indices {@code ka} and {@code kb} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>Note: Requires {@code ka <= kb}. The use of two indices is to support processing
     * of pairs of indices {@code (k, k+1)}. However the indices are treated independently
     * and partitioned by recursion. They may be equal, neighbours or well separated.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka Index.
     * @param kb Index.
     * @param maxDepth Maximum depth for recursion.
     */
    private void introselect(DPPartition part, double[] a, int left, int right,
        int ka, int kb, int maxDepth) {
        // Only one side requires recursion. The other side
        // can remain within this function call.
        int l = left;
        int r = right;
        int ka1 = ka;
        int kb1 = kb;
        final int[] upper = {0, 0, 0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka1 and kb1 are close to the ends
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---s1----
            //                       -----s3----
            //  ---------s2-----------
            //          ----------s4-----------
            final int s1 = ka1 - l;
            final int s2 = kb1 - l;
            final int s3 = r - kb1;
            final int s4 = r - ka1;
            if (maxDepth == 0 ||
                Math.min(s1 + s3, Math.min(s2, s4)) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                // Too much recursion, or ka1 and kb1 are both close to the ends
                heapSelectPair(a, l, r, ka1, kb1);
                return;
            }

            if (n < minQuickSelectSize || Math.min(s2, s4) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                //Sorting.sort(a, l, r, l > 0);
                return;
            }

            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the pair
            // and the recursion proceeds with a single point.
            maxDepth--;
            // Recurse left side if required
            if (ka1 < p0) {
                if (kb1 <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    kb1 = r < kb1 ? ka1 : kb1;
                    continue;
                }
                introselect(part, a, l, p0 - 1, ka1, ka1, maxDepth);
                // Here we must process middle and possibly right
                ka1 = kb1;
            }
            // Recurse middle if required
            // Check the either k is in the range (p1, p2)
            //                    p0 p1                p2 p3
            // |l|-----------------|P|------------------|P|----|r|
            if (ka1 < p2 && ka1 > p1 || kb1 < p2 && kb1 > p1) {
                // Advance lower bound
                l = p1 + 1;
                ka1 = ka1 < l ? kb1 : ka1;
                if (kb1 <= p3) {
                    // Entirely in middle
                    r = p2 - 1;
                    kb1 = r < kb1 ? ka1 : kb1;
                    continue;
                }
                introselect(part, a, l, p2 - 1, ka1, ka1, maxDepth);
                // Here we must process right
                ka1 = kb1;
            }
            if (kb1 <= p3) {
                // No right side
                return;
            }
            // Continue right
            l = p3 + 1;
            ka1 = ka1 < l ? kb1 : ka1;
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts a {@link SearchableInterval} of indices {@code k} and the
     * first index {@code ka} and last index {@code kb} that define the range of indices
     * to partition. The {@link SearchableInterval} is used to search for keys in {@code [ka, kb]}
     * to create {@code [ka, kb1]} and {@code [ka1, kb]} if partitioning splits the range.
     *
     * <pre>{@code
     * left <= ka <= kb <= right
     * }</pre>
     *
     * <p>Uses an introselect variant. The dual pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param ka First key.
     * @param kb Last key.
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(DPPartition part, double[] a, int left, int right,
        SearchableInterval k, int ka, int kb, int maxDepth) {
        // If partitioning splits the interval then recursion is used for left and/or
        // right sides and the middle remains within this function. If partitioning does
        // not split the interval then it remains within this function.
        int l = left;
        int r = right;
        int ka1 = ka;
        int kb1 = kb;
        final int[] upper = {0, 0, 0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka1 and kb1 are close to the same end
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---------s2-----------
            //          ----------s4-----------
            // Note: The overhead of dynamic heap select distance computation is negligible.
            // This allows various strategies for heapselect to be tested.
            if (maxDepth == 0 ||
                Math.min(kb1 - l, r - ka1) < Math.max((n >>> heapSelectShift) + heapSelectConstant,
                    heapSelectDynamicMask & heapSelectEdgeDistanceDP(n))) {
                // Too much recursion, or ka1 and kb1 are both close to the same end
                heapSelectRange(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }
//            // Note limit heap select to 2 when a full sort is possible
//            // (since heapselect is slow on tiny size).
//            if (maxDepth == 0 ||
//                Math.min(kb1 - l, r - ka1) < Math.min(n < minQuickSelectSize ? 2 : Integer.MAX_VALUE,
//                    Math.max((n >>> heapSelectShift) + heapSelectConstant,
//                        heapSelectDynamicMask & heapSelectEdgeDistance(n)))) {
//                // Too much recursion, or ka1 and kb1 are both close to the same end
//                heapSelectRange(a, l, r, ka1, kb1);
//                recursionConsumer.accept(maxDepth);
//                return;
//            }

            if (n < minQuickSelectSize || Math.min(kb1 - l, r - ka1) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                //Sorting.sort(a, l, r, l > 0);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            // Recursion to max depth
            // Note: Here we possibly branch left, middle and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set in each region.
            //                    p0 p1                p2 p3
            // |l|--|ka1|--k----k--|P|------k--|kb1|----|P|----|r|
            //                 kb1  |      ka1
            // Search previous/next is bounded at ka1/kb1
            maxDepth--;
            // Recurse left side if required
            if (ka1 < p0) {
                if (kb1 <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb1) {
                        kb1 = k.previousIndex(r);
                    }
                    continue;
                }
                introselect(part, a, l, p0 - 1, k, ka1, k.split(p0, p1, upper), maxDepth);
                ka1 = upper[0];
            }
            // Recurse right side if required
            if (kb1 > p3) {
                if (ka1 >= p2) {
                    // Entirely on right-side
                    l = p3 + 1;
                    if (ka1 < l) {
                        ka1 = k.nextIndex(l);
                    }
                    continue;
                }
                final int lo = k.split(p2, p3, upper);
                introselect(part, a, p3 + 1, r, k, upper[0], kb1, maxDepth);
                kb1 = lo;
            }
            // Check the interval overlaps the middle; and the middle exists.
            //                    p0 p1                p2 p3
            // |l|-----------------|P|------------------|P|----|r|
            // Eliminate:     ----kb1                    ka1----
            if (kb1 <= p1 || p2 <= ka1 || p2 - p1 <= 2) {
                // No middle
                recursionConsumer.accept(maxDepth);
                return;
            }
            l = p1 + 1;
            r = p2 - 1;
            // Interval [ka1, kb1] overlaps the middle but there may be nothing in the interval.
            // |l|-----------------|P|------------------|P|----|r|
            // Eliminate:          ka1                  kb1
            // Detect this if ka1 is advanced too far.
            if (ka1 < l) {
                ka1 = k.nextIndex(l);
                if (ka1 > r) {
                    // No middle
                    recursionConsumer.accept(maxDepth);
                    return;
                }
            }
            if (r < kb1) {
                kb1 = k.previousIndex(r);
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts a {@link UpdatingInterval} of indices {@code k} that define the
     * range of indices to partition. The {@link UpdatingInterval} can be narrowed or split as
     * partitioning divides the range.
     *
     * <p>Uses an introselect variant. The dual pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(DPPartition part, double[] a, int left, int right,
        UpdatingInterval k, int maxDepth) {
        // If partitioning splits the interval then recursion is used for left and/or
        // right sides and the middle remains within this function. If partitioning does
        // not split the interval then it remains within this function.
        int l = left;
        int r = right;
        int ka = k.left();
        int kb = k.right();
        final int[] upper = {0, 0, 0};
        while (true) {
            // length - 1
            final int n = r - l;

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2-----------
            //          ----------s4-----------
            // Note: The overhead of dynamic heap select distance computation is negligible.
            // This allows various strategies for heapselect to be tested.
            if (maxDepth == 0 ||
                Math.min(kb - l, r - ka) < Math.max((n >>> heapSelectShift) + heapSelectConstant,
                    heapSelectDynamicMask & heapSelectEdgeDistanceDP(n))) {
                // Too much recursion, or ka and kb are both close to the same end
                heapSelectRange(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (n < minQuickSelectSize || Math.min(kb - l, r - ka) < sortSelectConstant) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                //Sorting.sort(a, l, r, l > 0);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            // Recursion to max depth
            // Note: Here we possibly branch left, middle and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set in each region.
            //                   p0 p1               p2 p3
            // |l|--|ka|--k----k--|P|------k--|kb|----|P|----|r|
            //                 kb  |      ka
            // Search previous/next is bounded at ka/kb
            maxDepth--;
            // Recurse left side if required
            if (ka < p0) {
                if (kb <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb) {
                        kb = k.updateRight(r);
                    }
                    continue;
                }
                introselect(part, a, l, p0 - 1, k.splitLeft(p0, p1), maxDepth);
                ka = k.left();
            }
            // Recurse middle if required
            // Check the interval overlaps the middle; and the middle exists.
            //                    p0 p1                p2 p3
            // |l|-----------------|P|------------------|P|----|r|
            // Eliminate:      ----kb                    ka----
            if (ka < p2 && kb > p1 && p2 - p1 > 1) {
                // Advance lower bound
                l = p1 + 1;
                // Interval [ka, kb] overlaps the middle but there may be nothing in the interval.
                // |l|-----------------|P|------------------|P|----|r|
                // Eliminate:          ka1                  kb1
                // Detect this if ka must be advanced and passes p2.
                if (ka >= l || (ka = k.updateLeft(l)) < p2) {
                    if (kb <= p3) {
                        // Entirely in middle
                        r = p2 - 1;
                        if (r < kb) {
                            kb = k.updateRight(r);
                        }
                        continue;
                    }
                    introselect(part, a, l, p2 - 1, k.splitLeft(p2, p3), maxDepth);
                    // Here we must process right
                    ka = k.left();
                }
            }
            if (kb <= p3) {
                // No right side
                return;
            }
            // Continue right
            l = p3 + 1;
            if (ka < l) {
                ka = k.updateLeft(l);
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     *
     * <p>This function accepts an {@link IndexIterator} of indices {@code k}; for
     * convenience the lower and upper indices of the current interval are passed as the
     * first index {@code ka} and last index {@code kb} of the closed interval of indices
     * to partition. These may be within the lower and upper indices if the interval was
     * split during recursion: {@code lower <= ka <= kb <= upper}.
     *
     * <p>The data is recursively partitioned using left-most ordering. When the current
     * interval has been partitioned the {@link IndexIterator} is used to advance to the
     * next interval to partition.
     *
     * <p>Uses an introselect variant. The dual pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param ka First key.
     * @param kb Last key.
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    void introselect(DPPartition part, double[] a, int left, int right,
        IndexIterator k, int ka, int kb, int maxDepth) {
        // If partitioning splits the interval then recursion is used for left and/or
        // right sides and the middle remains within this function. If partitioning does
        // not split the interval then it remains within this function.
        int l = left;
        final int r = right;
        int lo = ka;
        int hi = kb;
        final int[] upper = {0, 0, 0};
        while (true) {
            if (maxDepth == 0) {
                // Too much recursion.
                // Advance the iterator to the end of the current range.
                // Note: heapSelectRange handles hi > right.
                // Single API method: advanceBeyond(right): return hi <= right
                while (hi < right && k.next()) {
                    hi = k.right();
                }
                heapSelectRange(a, l, right, lo, hi);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // length - 1
            final int n = right - l;

            // If interval is close to one end then heapselect.
            // Only heapselect left if there are no further indices in the range.
            // |l|-----|lo|--------|hi|------|right|
            //  ---------d1----------
            //          --------------d2-----------
            if (Math.min(hi - l, right - lo) < ((n >>> heapSelectShift) + heapSelectConstant)) {
                if (hi - l > right - lo) {
                    // Right end
                    heapSelectRight(a, l, right, lo, right - lo);
                    recursionConsumer.accept(maxDepth);
                    return;
                } else if (k.nextAfter(right)) {
                    // Left end
                    // Only if no further indices in the range.
                    // If false this branch will continue to be triggered until
                    // a partition is made to separate the next indices.
                    heapSelectLeft(a, l, right, hi, hi - l);
                    recursionConsumer.accept(maxDepth);
                    // Advance iterator
                    l = hi + 1;
                    if (!k.positionAfter(hi) || Math.max(k.left(), l) > right) {
                        // No more keys, or keys beyond the current bounds
                        return;
                    }
                    lo = Math.max(k.left(), l);
                    hi = Math.min(right, k.right());
                    // Continue right (allows a second heap select for the right side)
                    continue;
                }
            }

            // If interval is close to both ends then sort
            // |l|-----|lo|--------|hi|------|right|
            //  ---d1----
            //                       ----d2--------
            // (lo - l) + (right - hi) == (right - l) - (hi - lo)
            if (n - (hi - lo) < minQuickSelectSize) {
                // Handle small data. This is done as the JDK sort will
                // use insertion sort for small data. For double data it
                // will also pre-process the data for NaN and signed
                // zeros which is an overhead to avoid.
                if (n < minQuickSelectSize) {
                    Sorting.sort(a, l, right, l > 0);
                } else {
                    // Note: This disregards the current level of recursion
                    // but can exploit the JDK's more advanced sort algorithm.
                    Arrays.sort(a, l, right + 1);
                }
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Here: l <= lo <= hi <= right
            // Pick 2 pivots and partition
            int p0 = dualPivotingStrategy.pivotIndex(a, l, r, upper);
            p0 = part.partition(a, l, r, p0, upper[0], upper);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            maxDepth--;
            // Recursion left
            if (lo < p0) {
                introselect(part, a, l, p0 - 1, k, lo, Math.min(hi, p0 - 1), maxDepth);
                // Advance iterator
                if (!k.positionAfter(p1) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }
            if (hi <= p1) {
                // Advance iterator
                if (!k.positionAfter(p1) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }

            // Recursion middle
            l = p1 + 1;
            lo = Math.max(lo, l);
            if (lo < p2) {
                introselect(part, a, l, p2 - 1, k, lo, Math.min(hi, p2 - 1), maxDepth);
                // Advance iterator
                if (!k.positionAfter(p3) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }
            if (hi <= p3) {
                // Advance iterator
                if (!k.positionAfter(p3) || k.left() > right) {
                    // No more keys, or keys beyond the current bounds
                    return;
                }
                lo = k.left();
                hi = Math.min(right, k.right());
            }

            // Continue right
            l = p3 + 1;
            lo = Math.max(lo, l);
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
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionSBM(double[] data, int[] k, int n) {
        // Handle NaN (this does assume n > 0)
        final int right = sortNaN(data);
        partition((SPEPartitionFunction) this::partitionSBMWithZeros, data, right, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses an introselect variant. The quickselect is a single-pivot partition method;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISP(double[] data, int[] k, int n) {
        introselect(Partition::partitionSP, data, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy quicksort
     * partition method; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIBM(double[] data, int[] k, int n) {
        introselect(Partition::partitionBM, data, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy quicksort
     * partition method by Sedgewick; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISBM(double[] data, int[] k, int n) {
        introselect(Partition::partitionSBM, data, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is a Dutch-National-Flag
     * partition method; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIDNF(double[] data, int length, int[] k, int n) {
        introselect(Partition::partitionDNF3, data, length - 1, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses an introselect variant. The quickselect is a Dutch-National-Flag
     * partition method; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIDNF(double[] data, int[] k, int n) {
        introselect(Partition::partitionDNF3, data, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is a single-pivot partition method;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISP(double[] data, int length, int[] k, int n) {
        introselect(Partition::partitionSP, data, length - 1, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy quicksort;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIBM(double[] data, int length, int[] k, int n) {
        introselect(Partition::partitionBM, data, length - 1, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy quicksort
     * partition method by Sedgewick; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISBM(double[] data, int length, int[] k, int n) {
        introselect(Partition::partitionSBM, data, length - 1, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses an introselect variant. The quickselect is a dual-pivot quicksort
     * partition method by Vladimir Yaroslavskiy; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIDP(double[] data, int[] k, int n) {
        introselect((DPPartition) Partition::partitionDP, data, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is a dual-pivot quicksort
     * partition method by Vladimir Yaroslavskiy; the fall-back on poor convergence of the quickselect
     * is a heapselect.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionIDP(double[] data, int length, int[] k, int n) {
        introselect((DPPartition) Partition::partitionDP, data, length - 1, k, n);
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
     * <p>Note: This method does not use any configuration. It is built using the
     * components that perform well across a benchmarking for: single keys; a pair of keys;
     * multiple keys; all using a range of data input.
     *
     * <p>Note: This function pre/post-processes data to handles NaN and signed zeros.
     * It is used for testing. The {@link Quantile} and {@link Median} implementations
     * use a NaN policy and already perform floating-point data processing. This method should
     * only be called by benchmarking/testing functions that may create any type
     * of floating-point data.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices (assumed to be strictly positive).
     */
    static void select(double[] a, int[] k, int count) {
        // Handle NaN / signed zeros
        final DoubleDataTransformer t = SORT_TRANSFORMER.get();
        // Assume this is in-place
        t.preProcess(a);
        final int end = t.length();
        int n = count;
        if (end > 1) {
            // Filter indices invalidated by NaN check
            if (end < a.length) {
                for (int i = n; --i >= 0;) {
                    final int v = k[i];
                    if (v >= end) {
                        // swap(k, i, --n)
                        k[i] = k[--n];
                        k[n] = v;
                    }
                }
            }
            // select accepts an exclusive end
            select(a, end, k, n);
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses an introselect variant. The quickselect is either a single-pivot
     * Bentley-McIlroy partition method by Sedgewick, or a dual-pivot partition method by
     * Yaroslavskiy; the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Note: This method does not use any configuration. It is built using the
     * components that perform well across a benchmarking for: single keys; a pair of keys;
     * multiple keys; all using a range of data input.
     *
     * @param a Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    static void select(double[] a, int length, int[] k, int n) {
        if (n < 1) {
            return;
        }

        // Note:
        // Single-pivot quickselect is faster by up to 7% when searching for single keys;
        // the difference is most apparent when searching in the central region.
        // Dual-pivot quickselect is faster by 2% when searching for two keys and up to 8%
        // for multiple keys.
        // If the keys saturate the range then partitioning -> sorting.

        if (n == 1) {
            select(a, 0, length - 1, k[0], k[0], singlePivotMaxDepth(length));
            return;
        }

//        if (n == 2 && Math.abs(k[0] - k[1]) < MIN_SEPARATION_DISTANCE) {
//            final int k1 = Math.min(k[0], k[1]);
//            final int kn = Math.max(k[0], k[1]);
//            select(a, 0, length - 1, k1, kn, singlePivotMaxDepth(length));
//            return;
//        }

        final UpdatingInterval keys = IndexIntervals.createUpdatingInterval(k, n);
        final int k1 = keys.left();
        final int kn = keys.right();
        // If the keys are not separated then they are effectively a single key.
        if (kn - k1 < MIN_SEPARATION_DISTANCE) {
            select(a, 0, length - 1, k1, kn, singlePivotMaxDepth(length));
            return;
        }

        // Dual-pivot mode with small range sort length configured using index density
        //select(a, 0, length - 1, keys, dualPivotMaxDepth(length),
        //    dualPivotSortSelectSize(k1, kn, n));
        select(a, 0, length - 1, keys, dualPivotFlags(0, length - 1, k1, kn, n));
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [k1, kn]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k1] <= data[k1] <= data[kn] <= data[kn < i]
     * }</pre>
     *
     * <p>This function accepts a indices {@code [k1, kn]} that define the
     * range of indices to partition. The interval can be narrowed or split as
     * partitioning divides the range.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy quicksort
     * partition method by Sedgewick; switching to heapselect when close to the edge or
     * poor convergence of quickselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k1 First key of interest.
     * @param kn Last key of interest.
     * @param maxDepth Maximum depth for recursion.
     */
    // package-private for benchmarking
    static void select(double[] a, int left, int right, int k1, int kn, int maxDepth) {
        // If partitioning splits the interval then recursion is used for the left-most side(s)
        // and the right-most side remains within this function. If partitioning does
        // not split the interval then it remains within this function.
        int l = left;
        int r = right;
        int ka = k1;
        int kb = kn;
        final int[] upper = {0};
        while (true) {
            // select when ka and kb are close to the same end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            if (Math.min(kb - l, r - ka) < SORTSELECT_SIZE) {
                sortSelectRange(a, l, r, ka, kb);
                return;
            }
            if (maxDepth == 0) {
                // Excess recursion, switch to heap select
                heapSelectRange2(a, l, r, ka, kb);
                return;
            }

            // Single-pivot partitioning handling equal values
            final int p0 = partitionSBM(a, l, r, upper);
            final int p1 = upper[0];

            // Recursion to max depth
            // Note: Here we possibly branch left and right with multiple keys.
            // It is possible that the partition has split the range
            // and the recursion proceeds with a reduced range in each region.
            //                   p0 p1
            // |l|--|ka|kkkkkkkkkk|P|kkkkkkkkk|kb|------|r|
            //                  kb | ka
            maxDepth--;
            // Recurse left side if required
            if (ka < p0) {
                if (kb <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    kb = kb < r ? kb : r;
                    continue;
                }
                select(a, l, p0 - 1, ka, kb < r ? kb : r, maxDepth);
            }
            if (kb <= p1) {
                // No right side
                return;
            }
            // Continue on the right side
            l = p1 + 1;
            ka = ka > l ? ka : l;
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code k} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < k] <= data[k] <= data[k < i]
     * }</pre>
     *
     * <p>This function accepts a {@link UpdatingInterval} of indices {@code k} that define the
     * range of indices to partition. The {@link UpdatingInterval} can be narrowed or split as
     * partitioning divides the range.
     *
     * <p>Uses an introselect variant. The quickselect is a dual-pivot quicksort
     * partition method by Vladimir Yaroslavskiy; the fall-back on poor convergence of
     * the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>The control {@code flags} contain the the current recursion count and the configured
     * length threshold for {@code r - l} to perform sort select. The count is in the upper
     * bits and the threshold is in the lower bits.
     *
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Interval of indices to partition (ordered).
     * @param flags Control flags.
     */
    // package-private for benchmarking
    static void select(double[] a, int left, int right, UpdatingInterval k, int flags) {
        //int maxDepth, int ss) {
        // Inline code using the defaults.
        // Branching uses left/middle/right.
        // This allows branch prediction to track that after a split then the next section
        // should execute (since a split is used when there are indices after a pivot).

        // If partitioning splits the interval then recursion is used for the left-most side(s)
        // and the right-most side remains within this function. If partitioning does
        // not split the interval then it remains within this function.
        int l = left;
        int r = right;
        int ka = k.left();
        int kb = k.right();
        final int[] upper = {0, 0, 0};
        while (true) {
            // TODO: dynamic ss threshold is not working. It is too high.
            // Q. How to determine what the threshold should be...

            // select when ka and kb are close to the same end,
            // or the entire range is small
            // |l|-----|ka|--------|kb|------|r|
            if (r - l < (flags & SORTSELECT_MASK) || Math.min(kb - l, r - ka) < SORTSELECT_SIZE) {
                sortSelectRange(a, l, r, ka, kb);
                return;
            }
            if (flags < 0) {
                // Excess recursion, switch to heap select
                heapSelectRange2(a, l, r, ka, kb);
                return;
            }
//            // TODO:
//            // Switch between sort and heapselect to finish the range [ka, kb].
//            // Dynamically adapt the sort select size.
//            //ss = kb - ka < MIN_SEPARATION_DISTANCE ? SORTSELECT_SIZE : ss;
//            //ss = kb - ka <= 1 ? 0 : ss;
//
//            //if (r - l < ss) {
//
//            // TODO - test this condition...
//            // sortselect when [ka, kb] is a significant part of [l, r], and the size is small
//            // Approximate speed: heapselect(n/4) == sort(n)
//            if (r - l < ((kb - ka) << 2 > (r - l) ? ss : 0)) {
//                // Switch to a sort of small data to avoid partition overhead
//                //Sorting.sort(a, l, r, l > 0);
//                //Sorting.sort(a, l, r);
//                sortSelectRange(a, l, r, ka, kb);
//                return;
//            }
//
//            // heapselect when ka and kb are close to the same end, or too much recursion
//            // |l|-----|ka|--------|kb|------|r|
//            if (maxDepth == 0 ||
//                Math.min(kb - l, r - ka) < HEAPSELECT_SIZE) {
//                heapSelectRange(a, l, r, ka, kb);
//                return;
//            }

            // Dual-pivot partitioning
            final int p0 = partitionDP(a, l, r, upper, ka, kb);
            final int p1 = upper[0];
            final int p2 = upper[1];
            final int p3 = upper[2];

            // Recursion to max depth
            // Note: Here we possibly branch left, middle and right with multiple keys.
            // It is possible that the partition has split the keys
            // and the recursion proceeds with a reduced set in each region.
            //                   p0 p1               p2 p3
            // |l|--|ka|--k----k--|P|------k--|kb|----|P|----|r|
            //                 kb  |      ka
            flags += RECURSION_INCREMENT;
            // Recurse left side if required
            if (ka < p0) {
                if (kb <= p1) {
                    // Entirely on left side
                    r = p0 - 1;
                    if (r < kb) {
                        kb = k.updateRight(r);
                    }
                    continue;
                }
                select(a, l, p0 - 1, k.splitLeft(p0, p1), flags);
                // Here we must process middle and possibly right
                ka = k.left();
            }
            // Recurse middle if required
            // Check the interval overlaps the middle; and the middle exists.
            //                    p0 p1                p2 p3
            // |l|-----------------|P|------------------|P|----|r|
            // Eliminate:      ----kb                    ka----
            if (ka < p2 && kb > p1 && p2 > p1) {
                // Advance lower bound
                l = p1 + 1;
                // Interval [ka, kb] overlaps the middle but there may be nothing in the interval.
                // |l|-----------------|P|------------------|P|----|r|
                // Eliminate:          ka1                  kb1
                // Detect this if ka must be advanced and passes p2.
                if (ka >= l || (ka = k.updateLeft(l)) < p2) {
                    if (kb <= p3) {
                        // Entirely in middle
                        r = p2 - 1;
                        if (r < kb) {
                            kb = k.updateRight(r);
                        }
                        continue;
                    }
                    select(a, l, p2 - 1, k.splitLeft(p2, p3), flags);
                    // Here we must process right
                    ka = k.left();
                }
            }
            if (kb <= p3) {
                // No right side
                return;
            }
            // Continue right
            l = p3 + 1;
            if (ka < l) {
                ka = k.updateLeft(l);
            }

//            // Recurse right side if required
//            if (kb > p3) {
//                if (ka >= p2) {
//                    // Entirely on right-side
//                    l = p3 + 1;
//                    if (ka < l) {
//                        ka = k.updateLeft(l);
//                    }
//                    continue;
//                }
//                select(a, p3 + 1, r, k.splitRight(p2, p3), maxDepth);
//                // Here we must process middle
//                kb = k.right();
//            }
//            // Check the interval overlaps the middle; and the middle exists.
//            //                    p0 p1                p2 p3
//            // |l|-----------------|P|------------------|P|----|r|
//            // Eliminate:      ----kb                    ka----
//            if (kb <= p1 || p2 <= ka || p2 <= p1) {
//                // No middle
//                return;
//            }
//            l = p1 + 1;
//            r = p2 - 1;
//            // Interval [ka, kb] overlaps the middle but there may be nothing in the interval.
//            // Detect this if ka is advanced too far.
//            if (ka < l && (ka = k.updateLeft(l)) > r) {
//                // No middle
//                return;
//            }
//            if (r < kb) {
//                kb = k.updateRight(r);
//            }
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
        sort((SPEPartitionFunction) this::partitionSBMWithZeros, data, right);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a single-pivot quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortISP(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionSP, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a Bentley-McIlroy quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortIBM(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionBM, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a Bentley-McIlroy quicksort method by Sedgewick; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortISBM(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionSBM, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a Dutch-National-Flag quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortIDNF1(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionDNF1, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a Dutch-National-Flag quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortIDNF2(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionDNF2, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a Dutch-National-Flag quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortIDNF3(double[] data) {
        // NaN processing is done in the introsort method
        introsort(Partition::partitionDNF3, data);
    }

    /**
     * Sort the data using an intrasort.
     *
     * <p>Uses a dual-pivot quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortIDP(double[] data) {
        // NaN processing is done in the introsort method
        introsort((DPPartition) Partition::partitionDP, data);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Implements {@link SPEPartitionFunction}. This method is not static as the
     * pivot strategy and minimum quick select size are used within the method.
     *
     * <p>Note: Requires that the range contains no NaN values. Handles signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range.
     * @param leftInner Flag to indicate {@code left - 1} is a pivot.
     * @param rightInner Flag to indicate {@code right + 1} is a pivot.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int partitionSBMWithZeros(double[] data, int left, int right, int[] upper,
        boolean leftInner, boolean rightInner) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys (Sedgewick's algorithm).
        //
        // Partition data using pivot P into less-than, greater-than or equal.
        // P is placed at the end to act as a sentinel.
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
        if (right - left <= minQuickSelectSize) {
            Sorting.sort(data, left, right, leftInner);
            fixContinuousSignedZeros(data, left, right);
            upper[0] = right;
            return left;
        }

        final int l = left;
        final int r = right;

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinel value
        final int pivot = pivotingStrategy.pivotIndex(data, left, right);
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Special case: count signed zeros
        int c = 0;
        if (v == 0) {
            c = countMixedSignedZeros(data, left, right);
        }

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                if (c != 0) {
                    sortZero(data, left, right);
                }
                upper[0] = right;
                return left;
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
                // Cannot use j == i in the event that i == q (already passed j)
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
                    data[i] = data[p];
                    data[p] = v;
                    i++;
                    p++;
                }
                break;
            }
            //swap(data, i, j)
            final double vi = data[j];
            final double vj = data[i];
            data[i] = vi;
            data[j] = vj;
            // Move the equal values to the ends
            if (vi == v) {
                //swap(data, i, p++)
                data[i] = data[p];
                data[p] = v;
                p++;
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
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a single pivot partition method. This method does not handle equal values
     * at the pivot location: {@code lower == upper}. The method conforms to the
     * {@link SPEPartition} interface to allow use with the single-pivot introselect method.
     *
     * @param data Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionSP(double[] data, int l, int r, int pivot, int[] upper) {
        // Partition data using pivot P into less-than or greater-than.
        // P is placed at the end to act as a sentinel.
        // k traverses the unknown region ??? and values moved if less or greater:
        //
        // left            i            j              right
        // |          <P   |     ???    |   >P           |P|
        //
        // At the end P is swapped back to the centre.
        //
        // |         <P          |P|             >P        |

        // Use the pivot index to set the upper sentinel value
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        int i = l - 1;
        int j = r;

        for (;;) {
            // Cannot pass upper sentinal
            do {
                ++i;
            } while (data[i] < v);
            while (v < data[--j]) {
                // Cannot use i in the event that i == r
                if (j == l) {
                    break;
                }
            }
            if (i >= j) {
                break;
            }
            //swap(data, i, j)
            final double tmp = data[j];
            data[j] = data[i];
            data[i] = tmp;
        }

        // data[i] >= P
        // Move pivot value to correct location
        data[r] = data[i];
        data[i] = v;

        upper[0] = i;
        return i;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * @param data Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param pivot Initial index of the pivot.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionBM(double[] data, int l, int r, int pivot, int[] upper) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys.
        //
        // Adapted from program 7 in Bentley-McIlroy (1993)
        // Engineering a sort function
        // SOFTWARE—PRACTICE AND EXPERIENCE, VOL.23(11), 1249–1265
        //
        // 3-way partition of the data using a pivot value into
        // less-than, equal or greater-than.
        //
        // First partition data into 4 reqions by scanning the unknown region from
        // left (i) and right (j) and moving equal values to the ends:
        //                  i->       <-j
        // l        p       |           |         q       r
        // | equal  | less  |  unknown  | greater | equal |
        //
        //                    <-j
        // l        p             i               q       r
        // | equal  | less        |       greater | equal |
        //
        // Then the equal values are copied from the ends to the centre:
        // | less        |        equal      |    greater |

        int i = l;
        int j = r;
        int p = l;
        int q = r;

        final double v = data[pivot];

        for (;;) {
            while (i <= j && data[i] <= v) {
                if (data[i] == v) {
                    //swap(data, i, p++)
                    data[i] = data[p];
                    data[p] = v;
                    p++;
                }
                i++;
            }
            while (j >= i && data[j] >= v) {
                if (v == data[j]) {
                    //swap(data, j, q--)
                    data[j] = data[q];
                    data[q] = v;
                    q--;
                }
                j--;
            }
            if (i > j) {
                break;
            }
            //swap(data, i++, j--)
            final double tmp = data[j];
            data[j] = data[i];
            data[i] = tmp;
        }

        // Move equal regions to the centre.
        int s = Math.min(p - l, i - p);
        for (int k = l; s > 0; k++, s--) {
            //swap(data, k, i - s)
            data[k] = data[i - s];
            data[i - s] = v;
        }
        s = Math.min(q - j, r - q);
        for (int k = i; --s >= 0; k++) {
            //swap(data, r - s, k)
            data[r - s] = data[k];
            data[k] = v;
        }

        // Set output range
        i = i - p + l;
        j = j - q + r;
        upper[0] = j;

        return i;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * @param data Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    static int partitionSBM(double[] data, int l, int r, int pivot, int[] upper) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys (Sedgewick's algorithm).
        //
        // Partition data using pivot P into less-than, greater-than or equal.
        // P is placed at the end to act as a sentinel.
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
        // Note: The difference between this and the original BM partition is the use of
        // < or > rather than <= and >=. This allows the pivot to act as a sentinal and removes
        // the requirement for checks on i; and j can be checked against an unlikely condition.
        // This method will swap runs of equal values.
        //
        // The algorithm has been changed so that:
        // - A pivot point must be provided.
        // - An edge case where the search meets in the middle is handled.
        // - Equal value data is not swapped to the end. Since the value is fixed then
        //   only the less than / greater than value must be moved from the end inwards.
        //   The end is then assumed to be the equal value. This would not work with
        //   object references. Equivalent swap calls are commented.
        // - Added a fast-forward over initial range containing the pivot.
        // - Changed the final move to perform the minimum moves.

        int p = l;
        int q = r;

        // Use the pivot index to set the upper sentinel value
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                upper[0] = r;
                return l;
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
                // Cannot use j == i in the event that i == q (already passed j)
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
                    data[i] = data[p];
                    data[p] = v;
                    i++;
                    p++;
                }
                break;
            }
            //swap(data, i, j)
            final double vi = data[j];
            final double vj = data[i];
            data[i] = vi;
            data[j] = vj;
            // Move the equal values to the ends
            if (vi == v) {
                //swap(data, i, p++)
                data[i] = data[p];
                data[p] = v;
                p++;
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
        //   for k = l; k < p; k++
        //     swap(data, k, --j)
        // greater-equal:
        //   for k = r; k-- > q; i++
        //     swap(data, k, i)

        // Move the minimum of less-equal or less-than
        int move = Math.min(p - l, j - p);
        final int lower = j - (p - l);
        for (int k = l; --move >= 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        upper[0] = i + (r - q) - 1;
        for (int k = r; --move >= 0; i++) {
            data[--k] = data[i];
            data[i] = v;
        }

        // Equal in [lower, upper]
        return lower;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Sedgewick.
     *
     * <p>This method is similar to {@link #partitionSBM(double[], int, int, int, int[])}
     * with the following changes:
     * <ul>
     * <li>Selection of the pivots is performed in this method.
     * </ul>
     *
     * @param data Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    static int partitionSBM(double[] data, int l, int r, int[] upper) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys (Sedgewick's algorithm).
        //
        // Partition data using pivot P into less-than, greater-than or equal.
        // P is placed at the end to act as a sentinel.
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
        // Note: The difference between this and the original BM partition is the use of
        // < or > rather than <= and >=. This allows the pivot to act as a sentinal and removes
        // the requirement for checks on i; and j can be checked against an unlikely condition.
        // This method will swap runs of equal values.
        //
        // The algorithm has been changed so that:
        // - A pivot point must be provided.
        // - An edge case where the search meets in the middle is handled.
        // - Equal value data is not swapped to the end. Since the value is fixed then
        //   only the less than / greater than value must be moved from the end inwards.
        //   The end is then assumed to be the equal value. This would not work with
        //   object references. Equivalent swap calls are commented.
        // - Added a fast-forward over initial range containing the pivot.
        // - Changed the final move to perform the minimum moves.

        final int pivot = pivotIndex(data, l, r);

        // Use the pivot index to set the upper sentinel value
        final double v = data[pivot];
        data[pivot] = data[r];
        data[r] = v;

        int p = l;
        int q = r;

        // Fast-forward over equal regions to reduce swaps
        while (data[p] == v) {
            if (++p == q) {
                // Edge-case: constant value
                upper[0] = r;
                return l;
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
                // Cannot use j == i in the event that i == q (already passed j)
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
                    data[i] = data[p];
                    data[p] = v;
                    i++;
                    p++;
                }
                break;
            }
            //swap(data, i, j)
            final double vi = data[j];
            final double vj = data[i];
            data[i] = vi;
            data[j] = vj;
            // Move the equal values to the ends
            if (vi == v) {
                //swap(data, i, p++)
                data[i] = data[p];
                data[p] = v;
                p++;
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
        //   for k = l; k < p; k++
        //     swap(data, k, --j)
        // greater-equal:
        //   for k = r; k-- > q; i++
        //     swap(data, k, i)

        // Move the minimum of less-equal or less-than
        int move = Math.min(p - l, j - p);
        final int lower = j - (p - l);
        for (int k = l; --move >= 0; k++) {
            data[k] = data[--j];
            data[j] = v;
        }
        // Move the minimum of greater-equal or greater-than
        move = Math.min(r - q, q - i);
        upper[0] = i + (r - q) - 1;
        for (int k = r; --move >= 0; i++) {
            data[--k] = data[i];
            data[i] = v;
        }

        // Equal in [lower, upper]
        return lower;
    }

    /**
     * Find a pivot index of the array so that partitioning into 2-regions can be made.
     *
     * <pre>{@code
     * left <= p <= right
     * }</pre>
     *
     * @param data Array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @return pivot
     */
    private static int pivotIndex(double[] data, int l, int r) {
        // Median of 9 pivot selection using the median of 3 medians:
        // 1 4 7
        // x y z --> med(x,y,z) identifies pivot as 4th - 6th of 9 sorted values
        // 3 6 9
        // Bentley and McIlroy (1993) switch to median of 3 below size 40.
        // Heapselect edge distance is 15 so partitioning is limited to size 30
        // and we choose to always use this pivot selection.
        final int s = (r - l) >>> 3;
        final int m = (l + r) >>> 1;
        final int x = med3(data, l, l + s, l + (s << 1));
        final double a = data[x];
        final int y = med3(data, m - s, m, m + s);
        final double b = data[y];
        final int z = med3(data, r - (s << 1), r - s, r);
        return med3(a, b, data[z], x, y, z);
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
    private static int med3(double[] data, int i, int j, int k) {
        return med3(data[i], data[j], data[k], i, j, k);
    }

    /**
     * Find the median index of 3 values.
     *
     * @param a Value.
     * @param b Value.
     * @param c Value.
     * @param ia Index of a.
     * @param ib Index of b.
     * @param ic Index of c.
     * @return the median index
     */
    private static int med3(double a, double b, double c, int ia, int ib, int ic) {
        if (a < b) {
            if (b < c) {
                return ib;
            }
            return a < c ? ic : ia;
        }
        if (b > c) {
            return ib;
        }
        return a > c ? ic : ia;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Dutch-National-Flag method handling equal keys (version 1).
     *
     * @param data Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionDNF1(double[] data, int left, int right, int pivot, int[] upper) {
        // Dutch National Flag partitioning:
        // https://www.baeldung.com/java-sorting-arrays-with-repeated-entries
        // https://en.wikipedia.org/wiki/Dutch_national_flag_problem

        // Partition data using pivot P into less-than, greater-than or equal.
        // i traverses the unknown region ??? and values moved to the correct end.
        //
        // left    lt      i            gt       right
        // |  < P  |   P   |     ???    |   > P  |
        //
        // We can delay filling in [lt, gt) with P until the end and only
        // move values in the wrong place.

        final double value = data[pivot];

        // Fast-forward initial less-than region
        int lt = left;
        while (data[lt] < value) {
            lt++;
        }

        // Pointers positioned to use pre-increment/decrement: ++x / --x
        lt--;
        int gt = right + 1;

        // DNF partitioning which inspects one position per loop iteration
        for (int i = lt; ++i < gt;) {
            final double v = data[i];
            if (v < value) {
                data[++lt] = v;
            } else if (v > value) {
                data[i] = data[--gt];
                data[gt] = v;
                // Ensure data[i] is inspected next time
                i--;
            }
            // else v == value and is in the central region to fill at the end
        }

        // Equal in (lt, gt) so adjust to [lt, gt]
        ++lt;
        upper[0] = --gt;

        // Fill the equal values gap
        for (int i = lt; i <= gt; i++) {
            data[i] = value;
        }

        return lt;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Dutch-National-Flag method handling equal keys (version 2).
     *
     * @param data Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionDNF2(double[] data, int left, int right, int pivot, int[] upper) {
        // Dutch National Flag partitioning:
        // https://www.baeldung.com/java-sorting-arrays-with-repeated-entries
        // https://en.wikipedia.org/wiki/Dutch_national_flag_problem

        // Partition data using pivot P into less-than, greater-than or equal.
        // i traverses the unknown region ??? and values moved to the correct end.
        //
        // left    lt      i            gt       right
        // |  < P  |   P   |     ???    |   > P  |
        //
        // We can delay filling in [lt, gt) with P until the end and only
        // move values in the wrong place.

        final double value = data[pivot];

        // Fast-forward initial less-than region
        int lt = left;
        while (data[lt] < value) {
            lt++;
        }

        // Pointers positioned to use pre-increment/decrement: ++x / --x
        lt--;
        int gt = right + 1;

        // Modified DNF partitioning with fast-forward of the greater-than
        // pointer. Note the fast-forward must check bounds.
        for (int i = lt; ++i < gt;) {
            final double v = data[i];
            if (v < value) {
                data[++lt] = v;
            } else if (v > value) {
                // Fast-forward here:
                do {
                    --gt;
                } while (gt > i && data[gt] > value);
                // here data[gt] <= value
                // if data[gt] == value we can skip over it
                if (data[gt] < value) {
                    data[++lt] = data[gt];
                }
                // Move v to the >P side
                data[gt] = v;
            }
            // else v == value and is in the central region to fill at the end
        }

        // Equal in (lt, gt) so adjust to [lt, gt]
        ++lt;
        upper[0] = --gt;

        // Fill the equal values gap
        for (int i = lt; i <= gt; i++) {
            data[i] = value;
        }

        return lt;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Dutch-National-Flag method handling equal keys (version 3).
     *
     * @param data Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private static int partitionDNF3(double[] data, int left, int right, int pivot, int[] upper) {
        // Dutch National Flag partitioning:
        // https://www.baeldung.com/java-sorting-arrays-with-repeated-entries
        // https://en.wikipedia.org/wiki/Dutch_national_flag_problem

        // Partition data using pivot P into less-than, greater-than or equal.
        // i traverses the unknown region ??? and values moved to the correct end.
        //
        // left    lt      i            gt       right
        // |  < P  |   P   |     ???    |   > P  |
        //
        // This version writes in the value of P as it traverses. Any subsequent
        // less-than values will overwrite P values trailing behind i.

        final double value = data[pivot];

        // Fast-forward initial less-than region
        int lt = left;
        while (data[lt] < value) {
            lt++;
        }

        // Pointers positioned to use pre-increment/decrement: ++x / --x
        lt--;
        int gt = right + 1;

        // Note:
        // This benchmarks as faster than DNF1 and equal to DNF2 on random data.
        // On data with (many) repeat values it is faster than DNF2.
        // Both DNF2 & 3 have fast-forward of the gt pointer.

        // Modified DNF partitioning with fast-forward of the greater-than
        // pointer. Here we write in the pivot value at i during the sweep.
        // This acts as a sentinel when fast-forwarding greater-than.
        // It is over-written by any future <P value.
        // [begin, lt] < pivot
        // (lt, i)    == pivot
        // [i, gt)    == ???
        // [gt, end)   > pivot
        for (int i = lt; ++i < gt;) {
            final double v = data[i];
            if (v != value) {
                // Overwrite with the pivot value
                data[i] = value;
                if (v < value) {
                    // Move v to the <P side
                    data[++lt] = v;
                } else {
                    // Fast-forward here cannot pass sentinel
                    // while (data[--gt] > value)
                    do {
                        --gt;
                    } while (data[gt] > value);
                    // Now data[gt] <= value
                    // if data[gt] == value we can skip over it
                    if (data[gt] < value) {
                        data[++lt] = data[gt];
                    }
                    // Move v to the >P side
                    data[gt] = v;
                }
            }
        }

        // Equal in (lt, gt) so adjust to [lt, gt]
        ++lt;
        upper[0] = --gt;

        // In contrast to version 1 and 2 there is no requirement to fill the central
        // region with the pivot value as it was filled during the sweep

        return lt;
    }

    /**
     * Partition an array slice around 2 pivots. Partitioning exchanges array elements
     * such that all elements smaller than pivot are before it and all elements larger
     * than pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values. This does not respect the
     * ordering of signed zeros.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * <p>This method assumes {@code a[pivot1] <= a[pivot2]}.
     * If {@code pivot1 == pivot2} this triggers a switch to a single-pivot method.
     * It is assumed this indicates that choosing two pivots failed due to many equal
     * values. In this case the single-pivot method uses a Dutch National Flag algorithm
     * suitable for many equal values.
     *
     * <p>This method returns 4 points describing the pivot ranges of equal values.
     *
     * <pre>{@code
     *         |k0  k1|                |k2  k3|
     * |   <P  | ==P1 |  <P1 && <P2    | ==P2 |   >P   |
     * }</pre>
     *
     * <ul>
     * <li>k0: lower pivot1 point
     * <li>k1: upper pivot1 point (inclusive)
     * <li>k2: lower pivot2 point
     * <li>k3: upper pivot2 point (inclusive)
     * </ul>
     *
     * <p>Bounds are set so {@code i < k0},  {@code i > k3} and {@code k1 < i < k2} are
     * unsorted. When the range {@code [k0, k3]} contains fully sorted elements the result
     * is set to {@code k1 = k3; k2 == k0}. This can occur if
     * {@code P1 == P2} or there are zero or 1 value between the pivots
     * {@code P1 < v < P2}. Any sort between {@code k1 + 1} and {@code k2 - 1} must handle
     * a negative length. Any select of an index interval {@code [ka, kb]} that identifies
     * {@code k1 < kb || ka < k2} must also check {@code k2 - k1 > 1}.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param bounds Points [k1, k2, k3].
     * @param pivot1 Pivot1 location.
     * @param pivot2 Pivot2 location.
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    static int partitionDP(double[] a, int left, int right, int pivot1, int pivot2, int[] bounds) {
        // Allow caller to choose a single-pivot
        if (pivot1 == pivot2) {
            // Switch to a single pivot sort. This is used when there are
            // estimated to be many equal values so use the fastest equal
            // value single pivot method.
            final int lower = partitionDNF3(a, left, right, pivot1, bounds);
            // Set dual pivot range
            bounds[2] = bounds[0];
            // No unsorted internal region (set k1 = k3; k2 = k0)
            // Note: It is extra work for the caller to detect that this region can be skipped.
            bounds[1] = lower;
            return lower;
        }

        // Dual-pivot quicksort method by Vladimir Yaroslavskiy.
        //
        // Partition data using pivots P1 and P2 into less-than, greater-than or between.
        // Pivot values P1 & P2 are placed at the end. If P1 < P2, P2 acts as a sentinel.
        // k traverses the unknown region ??? and values moved if less-than (lt) or
        // greater-than (gt):
        //
        // left        lt                k           gt        right
        // |P1|  <P1   |   P1 <= & <= P2 |    ???    |    >P2   |P2|
        //
        // <P1           (left, lt)
        // P1<= & <= P2  [lt, k)
        // >P2           (gt, right)
        //
        // At the end pivots are swapped back to behind the lt and gt pointers.
        //
        // |  <P1        |P1|     P1<= & <= P2    |P2|      >P2    |
        //
        // Adapted from Yaroslavskiy
        // http://codeblab.com/wp-content/uploads/2009/09/DualPivotQuicksort.pdf
        //
        // Modified to allow partial sorting (partitioning):
        // - Allow the caller to supply the pivot indices
        // - Ignore insertion sort for tiny array (handled by calling code)
        // - Ignore recursive calls for a full sort (handled by calling code)
        // - Change to fast-forward over initial ascending / descending runs
        // - Change to a single-pivot partition method if the pivots are equal
        // - Change to fast-forward great when v > v2 and either break the sorting
        //   loop, or move a[great] direct to the correct location.
        // - Change to remove the 'div' parameter used to control the pivot selection
        //   using the medians method (div initialises as 3 for 1/3 and 2/3 and increments
        //   when the central region is too large).
        // - Identify a large central region using ~5/8 of the length.

        final double v1 = a[pivot1];
        final double v2 = a[pivot2];

        // Swap ends to the pivot locations.
        a[pivot1] = a[left];
        a[pivot2] = a[right];
        a[left] = v1;
        a[right] = v2;

        // pointers
        int less = left;
        int great = right;

        // Fast-forward ascending / descending runs to reduce swaps.
        // Cannot overrun as end pivots (v1 <= v2) act as sentinels.
        do {
            ++less;
        } while (a[less] < v1);
        do {
            --great;
        } while (a[great] > v2);

        // a[less - 1] < P1 : a[great + 1] > P2
        // unvisited in [less, great]
        SORTING:
        for (int k = less - 1; ++k <= great;) {
            final double v = a[k];
            if (v < v1) {
                // swap(a, k, less++)
                a[k] = a[less];
                a[less] = v;
                less++;
            } else if (v > v2) {
                // while k < great and a[great] > v2:
                //   great--
                while (a[great] > v2) {
                    if (great-- == k) {
                        // Done
                        break SORTING;
                    }
                }
                // swap(a, k, great--)
                // if a[k] < v1:
                //   swap(a, k, less++)
                final double w = a[great];
                a[great] = v;
                great--;
                // delay a[k] = w
                if (w < v1) {
                    a[k] = a[less];
                    a[less] = w;
                    less++;
                } else {
                    a[k] = w;
                }
            }
        }

        // Change to inclusive ends : a[less] < P1 : a[great] > P2
        less--;
        great++;
        // Move the pivots to correct locations
        a[left] = a[less];
        a[less] = v1;
        a[right] = a[great];
        a[great] = v2;

        // Record the pivot locations
        final int lower = less;
        bounds[2] = great;

        // equal elements
        // Original paper: If middle partition is bigger than a threshold
        // then check for equal elements.

        // Note: This is extra work. When performing partitioning the region of interest
        // may be entirely above or below the central region and this can be skipped.

        // Here we look for equal elements if the centre is more than 5/8 the length.
        // 5/8 = 1/2 + 1/8. Pivots must be different.
        if ((great - less) > ((right - left) >>> 1) + ((right - left) >>> 3) && v1 != v2) {

            // Fast-forward to reduce swaps. Changes inclusive ends to exclusive ends.
            // Since v1 != v2 these act as sentinels to prevent overrun.
            do {
                ++less;
            } while (a[less] == v1);
            do {
                --great;
            } while (a[great] == v2);

            // This copies the logic in the sorting loop using == comparisons
            EQUAL:
            for (int k = less - 1; ++k <= great;) {
                final double v = a[k];
                if (v == v1) {
                    a[k] = a[less];
                    a[less] = v;
                    less++;
                } else if (v == v2) {
                    while (a[great] == v2) {
                        if (great-- == k) {
                            // Done
                            break EQUAL;
                        }
                    }
                    final double w = a[great];
                    a[great] = v;
                    great--;
                    if (w == v1) {
                        a[k] = a[less];
                        a[less] = w;
                        less++;
                    } else {
                        a[k] = w;
                    }
                }
            }

            // Change to inclusive ends
            less--;
            great++;
        }

        // Between pivots in (less, great)
        if (v1 < v2 && less < great - 1) {
            // Record the pivot end points
            bounds[0] = less;
            bounds[1] = great;
        } else {
            // No unsorted internal region (set k1 = k3; k2 = k0)
            bounds[0] = bounds[2];
            bounds[1] = lower;
        }

        return lower;
    }

    /**
     * Partition an array slice around 2 pivots. Partitioning exchanges array elements
     * such that all elements smaller than pivot are before it and all elements larger
     * than pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values. This does not respect the
     * ordering of signed zeros.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * <p>This method returns 4 points describing the pivot ranges of equal values.
     *
     * <pre>{@code
     *         |k0  k1|                |k2  k3|
     * |   <P  | ==P1 |  <P1 && <P2    | ==P2 |   >P   |
     * }</pre>
     *
     * <ul>
     * <li>k0: lower pivot1 point
     * <li>k1: upper pivot1 point (inclusive)
     * <li>k2: lower pivot2 point
     * <li>k3: upper pivot2 point (inclusive)
     * </ul>
     *
     * <p>Bounds are set so {@code i < k0},  {@code i > k3} and {@code k1 < i < k2} are
     * unsorted. When the range {@code [k0, k3]} contains fully sorted elements the result
     * is set to {@code k1 = k3; k2 == k0}. This can occur if
     * {@code P1 == P2} or there are zero or 1 value between the pivots
     * {@code P1 < v < P2}. Any sort between {@code k1 + 1} and {@code k2 - 1} must handle
     * a negative length. Any select of an index interval {@code [ka, kb]} that identifies
     * {@code k1 < kb || ka < k2} must also check {@code k2 - k1 > 1}.
     *
     * <p>This method is similar to {@link #partitionDP(double[], int, int, int, int, int[])}
     * with the following changes:
     * <ul>
     * <li>Selection of the pivots is performed in this method.
     * <li>The first {@code k1} and last {@code kn} indices of interest are passed. These
     * are used to determine if the central region should be processed. Benchmarking
     * fails to show this is noticeable.
     * </ul>
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param bounds Points [k1, k2, k3].
     * @param k1 First key of interest.
     * @param kn Last key of interest.
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    static int partitionDP(double[] a, int left, int right, int[] bounds,
            int k1, int kn) {

        // Pick 2 pivots from 5 approximately uniform through the range.
        // Spacing is ~ 1/7 made using shifts. Other strategies are equal or much
        // worse. 1/7 = 5/35 ~ 1/8 + 1/64 : 0.1429 ~ 0.1406
        // Ensure the value is above zero to choose different points!
        final int n = right - left;
        final int step = 1 + (n >>> 3) + (n >>> 6);
        final int i3 = left + (n >>> 1);
        final int i2 = i3 - step;
        final int i1 = i2 - step;
        final int i4 = i3 + step;
        final int i5 = i4 + step;
        Sorting.sort5(a, i1, i2, i3, i4, i5);

//        // Sort the 5 points. This includes a detection for already sorted data.
//        if (Sorting.sort5a(a, i1, i2, i3, i4, i5) && Sorting.isAscending(a, left, right)) {
//            // k1 = k3; k2 == k0
//            bounds[0] = bounds[2] = right;
//            bounds[1] = left;
//            return left;
//        }

        // Possible switch to single pivot partition here ...
//        if (a[i2] == a[i4]) {
//            // Switch to a single pivot sort. This is used when there are
//            // estimated to be many equal values so use the fastest equal
//            // value single pivot method.
//            final int lower = partitionDNF3(a, left, right, i3, bounds);
//            // Set dual pivot range
//            bounds[2] = bounds[0];
//            // No unsorted internal region (set k1 = k3; k2 = k0)
//            // Note: It is extra work for the caller to detect that this region can be skipped.
//            bounds[1] = lower;
//            return lower;
//        }

        // Dual-pivot quicksort method by Vladimir Yaroslavskiy.
        //
        // Partition data using pivots P1 and P2 into less-than, greater-than or between.
        // Pivot values P1 & P2 are placed at the end. If P1 < P2, P2 acts as a sentinel.
        // k traverses the unknown region ??? and values moved if less-than (lt) or
        // greater-than (gt):
        //
        // left        lt                k           gt        right
        // |P1|  <P1   |   P1 <= & <= P2 |    ???    |    >P2   |P2|
        //
        // <P1           (left, lt)
        // P1<= & <= P2  [lt, k)
        // >P2           (gt, right)
        //
        // At the end pivots are swapped back to behind the lt and gt pointers.
        //
        // |  <P1        |P1|     P1<= & <= P2    |P2|      >P2    |
        //
        // Adapted from Yaroslavskiy
        // http://codeblab.com/wp-content/uploads/2009/09/DualPivotQuicksort.pdf
        //
        // Modified to allow partial sorting (partitioning):
        // - Allow the caller to supply the pivot indices
        // - Ignore insertion sort for tiny array (handled by calling code)
        // - Ignore recursive calls for a full sort (handled by calling code)
        // - Change to fast-forward over initial ascending / descending runs
        // - Change to a single-pivot partition method if the pivots are equal
        // - Change to fast-forward great when v > v2 and either break the sorting
        //   loop, or move a[great] direct to the correct location.
        // - Change to remove the 'div' parameter used to control the pivot selection
        //   using the medians method (div initialises as 3 for 1/3 and 2/3 and increments
        //   when the central region is too large).
        // - Identify a large central region using ~5/8 of the length.

        // Swap ends to the pivot locations.
        final double v1 = a[i2];
        a[i2] = a[left];
        a[left] = v1;
        final double v2 = a[i4];
        a[i4] = a[right];
        a[right] = v2;

        // pointers
        int less = left;
        int great = right;

        // Fast-forward ascending / descending runs to reduce swaps.
        // Cannot overrun as end pivots (v1 <= v2) act as sentinels.
        do {
            ++less;
        } while (a[less] < v1);
        do {
            --great;
        } while (a[great] > v2);

        // a[less - 1] < P1 : a[great + 1] > P2
        // unvisited in [less, great]
        SORTING:
        for (int k = less - 1; ++k <= great;) {
            final double v = a[k];
            if (v < v1) {
                // swap(a, k, less++)
                a[k] = a[less];
                a[less] = v;
                less++;
            } else if (v > v2) {
                // while k < great and a[great] > v2:
                //   great--
                while (a[great] > v2) {
                    if (great-- == k) {
                        // Done
                        break SORTING;
                    }
                }
                // swap(a, k, great--)
                // if a[k] < v1:
                //   swap(a, k, less++)
                final double w = a[great];
                a[great] = v;
                great--;
                // delay a[k] = w
                if (w < v1) {
                    a[k] = a[less];
                    a[less] = w;
                    less++;
                } else {
                    a[k] = w;
                }
            }
        }

        // Change to inclusive ends : a[less] < P1 : a[great] > P2
        less--;
        great++;
        // Move the pivots to correct locations
        a[left] = a[less];
        a[less] = v1;
        a[right] = a[great];
        a[great] = v2;

        // Record the pivot locations
        final int lower = less;
        bounds[2] = great;

        // equal elements
        // Original paper: If middle partition is bigger than a threshold
        // then check for equal elements.

        // Note: This is extra work. When performing partitioning the region of interest
        // may be entirely above or below the central region and this can be skipped.

        // Here we look for equal elements if the centre is more than 5/8 the length.
        // 5/8 = 1/2 + 1/8. Pivots must be different and the central region must be
        // of interest.
        if ((great - less) > (n >>> 1) + (n >>> 3) &&
            v1 != v2 && kn > less && k1 < great) {

            // Fast-forward to reduce swaps. Changes inclusive ends to exclusive ends.
            // Since v1 != v2 these act as sentinels to prevent overrun.
            do {
                ++less;
            } while (a[less] == v1);
            do {
                --great;
            } while (a[great] == v2);

            // This copies the logic in the sorting loop using == comparisons
            EQUAL:
            for (int k = less - 1; ++k <= great;) {
                final double v = a[k];
                if (v == v1) {
                    a[k] = a[less];
                    a[less] = v;
                    less++;
                } else if (v == v2) {
                    while (a[great] == v2) {
                        if (great-- == k) {
                            // Done
                            break EQUAL;
                        }
                    }
                    final double w = a[great];
                    a[great] = v;
                    great--;
                    if (w == v1) {
                        a[k] = a[less];
                        a[less] = w;
                        less++;
                    } else {
                        a[k] = w;
                    }
                }
            }

            // Change to inclusive ends
            less--;
            great++;
        }

        // Between pivots in (less, great)
        if (v1 != v2 && less < great - 1) {
            // Record the pivot end points
            bounds[0] = less;
            bounds[1] = great;
        } else {
            // No unsorted internal region (set k1 = k3; k2 = k0)
            bounds[0] = bounds[2];
            bounds[1] = lower;
        }

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
     * Move NaN values to the end of the array. Count signed zeros {@code -0.0}.
     * Any signed zero is replaced with {@code 0.0}.
     *
     * <p>Returns the end of the data and the count of signed zeros packed as a long:
     * <pre>
     * long x = ...
     * int end = (int) x;
     * int count = (int) (x >>> 0);
     * </pre>
     *
     * @param data Values.
     * @return {signedZeroCount, end of data}
     */
    static long sortNaNandCountSignedZeros(double[] data) {
        int end = data.length;
        int cn = 0;
        for (int i = end; i > 0;) {
            final double v = data[--i];
            // Count negative zeros using a sign bit check.
            // This requires a performance test. If the conversion to raw bits
            // is natively supported this is faster than using the == check.
            //if (v == 0.0 && Double.doubleToRawLongBits(v) < 0) {
            if (Double.doubleToRawLongBits(v) == Long.MIN_VALUE) {
                cn++;
                // Change to positive zero.
                // The later pass then only has to restore negatives.
                data[i] = 0.0;
            } else if (v != v) {
                // Move NaN to end
                data[i] = data[--end];
                data[end] = v;
            }
        }
        // Pack result
        return (((long) cn) << Integer.SIZE) | end;
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
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return index of a zero
     */
    static int containsMixedZeros(double[] data, int left, int right) {
        int c = 0;
        int cn = 0;
        for (int i = left; i <= right; i++) {
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
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the count
     */
    static int countSignedZeros1(double[] data, int left, int right) {
        // Count negative zeros
        int c = 0;
        for (int i = left; i <= right; i++) {
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
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the count of signed zeros if some positive zeros are also present
     */
    static int countMixedSignedZeros(double[] data, int left, int right) {
        // Count negative zeros
        int c = 0;
        int cn = 0;
        for (int i = left; i <= right; i++) {
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
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    static void sortZero(double[] data, int left, int right) {
        // Count negative zeros
        int c = 0;
        for (int i = left; i <= right; i++) {
            if (Double.doubleToRawLongBits(data[i]) < 0) {
                c++;
            }
        }
        // Replace
        if (c != 0) {
            int i = left;
            while (c-- > 0) {
                data[i++] = -0.0;
            }
            while (i <= right) {
                data[i++] = 0.0;
            }
        }
    }

    /**
     * Detect and fix the sort order of signed zeros. Assumes the data may have been
     * partially ordered around zero.
     *
     * <p>Searches for zeros if {@code data[left] <= 0} and {@code data[right] >= 0}.
     * If zeros are discovered in the range then they are assumed to be continuous.
     *
     * @param data Values.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     */
    private static void fixContinuousSignedZeros(double[] data, int left, int right) {
        int j;
        if (data[left] <= 0 && data[right] >= 0) {
            int i = left;
            while (data[i] < 0) {
                i++;
            }
            j = right;
            while (data[j] > 0) {
                j--;
            }
            sortZero(data, i, j);
        }
    }

    /**
     * Creates the maximum recursion depth for single-pivot quickselect recursion.
     *
     * <p>Warning: A length of zero will create a negative recursion depth.
     * In practice this does not matter as the sort / partition of a length
     * zero array should ignore the data.
     *
     * @param n Length of data (must be strictly positive).
     * @return the maximum recursion depth
     */
    private int createMaxDepthSinglePivot(int n) {
        // Ideal single pivot recursion will take log2(n) steps as data is
        // divided into length (n/2) at each iteration.
        final int maxDepth = floorLog2(n);
        // This factor should be tuned for practical performance
        return (int) Math.floor(maxDepth * recursionMultiple) + recursionConstant;
    }

    /**
     * Compute {@code floor(log 2 (x))}. This is valid for all strictly positive {@code x}.
     *
     * <p>Returns -1 for {@code x = 0} in place of -infinity.
     *
     * @param x Value.
     * @return {@code floor(log 2 (x))}
     */
    static int floorLog2(int x) {
        return 31 - Integer.numberOfLeadingZeros(x);
    }

    /**
     * Creates the maximum recursion depth for dual-pivot quickselect recursion.
     *
     * <p>Warning: A length of zero will create a high recursion depth.
     * In practice this does not matter as the sort / partition of a length
     * zero array should ignore the data.
     *
     * @param n Length of data (must be strictly positive).
     * @return the maximum recursion depth
     */
    private int createMaxDepthDualPivot(int n) {
        // Ideal dual pivot recursion will take log3(n) steps as data is
        // divided into length (n/3) at each iteration.
        final int maxDepth = log3(n);
        // This factor should be tuned for practical performance
        return (int) Math.floor(maxDepth * recursionMultiple) + recursionConstant;
    }

    /**
     * Compute an approximation to {@code log3 (x)}.
     *
     * <p>The result is between {@code floor(log3(x))} and {@code ceil(log3(x))}.
     * The result is correctly rounded when {@code x +/- 1} is a power of 3.
     *
     * @param x Value.
     * @return {@code log3(x))}
     */
    static int log3(int x) {
        // log3(2) ~ 1.5849625
        // log3(x) ~ log2(x) * 0.630929753... ~ log2(x) * 323 / 512 (0.630859375)
        // Use (floor(log2(x))+1) * 323 / 512
        // This result is always between floor(log3(x)) and ceil(log3(x)).
        // It is correctly rounded when x +/- 1 is a power of 3.
        return ((32 - Integer.numberOfLeadingZeros(x)) * 323) >>> 9;
    }

    /**
     * Compute the maximum recursion depth for single pivot recursion.
     * Uses {@code 2 * floor(log2 (x))}.
     *
     * @param x Value.
     * @return {@code log3(x))}
     */
    static int singlePivotMaxDepth(int x) {
        return (31 - Integer.numberOfLeadingZeros(x)) << 1;
    }

    /**
     * Compute the maximum recursion depth for dual pivot recursion.
     * This is an approximation to {@code 2 * log3 (x)}.
     *
     * <p>The result is between {@code floor(log3(x))} and {@code ceil(log3(x))}.
     * The result is correctly rounded when {@code x +/- 1} is a power of 3.
     *
     * @param x Value.
     * @return {@code log3(x))}
     */
    static int dualPivotMaxDepth(int x) {
        // log3(2) ~ 1.5849625
        // log3(x) ~ log2(x) * 0.630929753... ~ log2(x) * 323 / 512 (0.630859375)
        // Use (floor(log2(x))+1) * 323 / 256
        // This result is always between 2 * floor(log3(x)) and 2 * ceil(log3(x)).
        // It is correctly rounded when x +/- 1 is a power of 3.
        return ((32 - Integer.numberOfLeadingZeros(x)) * 323) >>> 8;
    }

    /**
     * Configure the sort select size for dual pivot partitioning.
     *
     * @param k1 First key of interest.
     * @param kn Last key of interest.
     * @param n Number of indices (must be above 1).
     * @return the sort select size.
     */
    static int dualPivotSortSelectSize(int k1, int kn, int n) {
        // Configure the sort select size based on the index density
        // l---k1---k---k-----k--k------kn----r
        //
        // For a full sort the dual-pivot quicksort can switch to insertion sort
        // when the length is small. The optimum value is dependent on the
        // hardware and the insertion sort implementation. Benchmarks show that
        // insertion sort can be used at length 80-120.
        //
        // During selection the SORTSELECT_SIZE specifies the distance from the edge
        // to use sort select. When keys are not dense there may be a small length
        // that is ignored by sort select due to the presence of another key.
        // Diagram of k-l = SORTSELECT_SIZE and r-k < SORTSELECT_SIZE where a second
        // key b is blocking the use of sort select. The key b is closest it can be to the right
        // key to enable blocking; it could be further away (up to k = left).
        //
        // |--SORTSELECT_SIZE--|
        //    |--SORTSELECT_SIZE--|
        // l--b----------------k--r
        // l----b--------------k----r
        // l------b------------k------r
        // l--------b----------k--------r
        // l----------b--------k----------r
        // l------------b------k------------r
        // l--------------b----k--------------r
        // l----------------b--k----------------r
        // l------------------bk------------------r
        //                    |--SORTSELECT_SIZE--|
        //
        // For all these cases the partitioning method would have to run. Assuming ideal
        // dual-pivot partitioning into thirds, and that the left key is randomly positioned
        // in [left, k) it is more likely that after partitioning 2 partitions will have to
        // be processed rather than 1 partition. In this case the options are:
        // - split the range using partitioning; sort select next iteration
        // - use sort select with a edge distance above the optimum length for single k selection
        //
        // Contrast with a longer length:
        // |--SORTSELECT_SIZE--|
        // l-------------------k-----k-------k-------------------r
        //                                   |--SORTSELECT_SIZE--|
        // Here partitioning has to run and 1, 2, or 3 partitions processed. But all k can
        // be found with a sort. In this case sort select could be used with a much higher
        // length (e.g. 80 - 120).
        //
        // When keys are extremely sparse (never within SORTSELECT_SIZE) then no switch
        // to sort select based on length is *required*. It may still be beneficial to avoid
        // partitioning if the length is very small due to the overhead of partitioning.
        //
        // Benchmarking with different lengths for a switch to sort select show inconsistent
        // behaviour across platforms due to the variable speed of insertion sort at longer
        // lengths. Attempts to transition the length based on various ramps schemes can
        // be incorrect and result is a slowdown rather than speed-up (if the correct threshold
        // is not chosen).
        //
        // Here we use a much simpler scheme based on these observations:
        // - If the average separation is very low then no length will collect extra indices
        // from a sort select over the current trigger of using the distance from the end. But
        // using a length dependence will not effect the work done by sort select as it only
        // performs the minimum sorting required.
        // - If the average separation is within the SORTSELECT_SIZE then a round of
        // partitioning will create multiple regions that all require a sort selection.
        // Thus a partitioning round can be avoided if the length is small.
        // -If the keys are at the end with nothing in between then partitioning will be able
        // to split them and a sort will have to sort the entire range:
        // lk-------------------------------kr
        // After partitioning starts the chance of keys being at the ends is low as keys
        // should be random within the divided range.
        // - Extremely high density keys is rare. It is only expected to saturate the range
        // with short lengths, e.g. 100 quantiles for length 1000 = separation 10 (high density)
        // but for length 10000 = separation 100 (low density).
        // - The density of (non-uniform) keys is hard to predict without complex analysis.
        //
        // Benchmarking using random keys at various density show no performance loss from
        // using a fixed size for the length dependence of sort select, if the size is small.
        // A large length can impact performance with low density keys, and on machines
        // where insertion sort is slower. Extreme performance gains occur when the average
        // separation of random keys is below 8-16, or of uniform keys around 32, by using a
        // sort at lengths up to 90. But this threshold shows performance loss greater than
        // the gains with separation of 64-128 on random keys, and on machines with slow
        // insertion sort. The transition to using an insertion sort of a longer length
        // is difficult to predict for all situations.

        // Let partitioning run on small lengths.
        // Use kn - k1 as a proxy for the length. If length is actually very large then
        // the final selection is insignificant. This avoids slowdown for small lengths
        // where the keys may only be at the ends. Note ideal dual-pivot partitioning
        // creates thirds so 1 iteration on SORTSELECT_SIZE * 3 should create
        // SORTSELECT_SIZE partitions.
        if (kn - k1 < SORTSELECT_SIZE * 3) {
            return 0;
        }
        // Here partitioning will run at least once.
        // Stable performance across platforms using a modest length dependence.
        return SORTSELECT_SIZE * 2;
    }

    /**
     * Configure the dual-pivot control flags. This packs the maximum recursion depth and
     * sort select size into a single integer.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k1 First key of interest.
     * @param kn Last key of interest.
     * @param n Number of indices (must be above 1).
     * @return the flags
     */
    static int dualPivotFlags(int left, int right, int k1, int kn, int n) {
        final int maxDepth = dualPivotMaxDepth(right - left);
        final int ss = dualPivotSortSelectSize(k1, kn, n);
        // The flags are packed using the upper bits to count back from -1 in
        // step sizes. The lower bits pack the sort select size.
        int flags = Integer.MIN_VALUE - maxDepth * RECURSION_INCREMENT;
        flags &= ~SORTSELECT_MASK;
        return flags | ss;
    }

    /**
     * Determine a threshold for the distance of a partition index {@code k} from the end
     * of the length to switch to heapselect. Applies to single-pivot partitioning.
     * For convenience this method accepts length {@code n = r - l} and not the correct
     * {@code n = r - l + 1}.
     *
     * <p>This method is used to estimate when heapselect will be faster than repeat
     * partitioning using quickselect. Heapselect will use a heap of size {@code k}. When
     * {@code k} is very small, relative to the length of the data, it is faster to
     * perform a single pass heapselect.
     *
     * <p>This method requires some more analysis of a suitable switch point.
     *
     * @param n Length.
     * @return distance
     */
    static int heapSelectEdgeDistanceSP(int n) {
        // Note:
        // 1. Heapselect is slow compared to insertion sort on small data.
        // 2. Heapselect can be much faster on large data.
        // Benchmarking shows that heapselect 1 is useful in SP partitioning
        // even at small lengths.
        // Here we set a conservative threshold using floor(log2(n)) but we
        // shift n to avoid heapselect on tiny lengths.
        // n   threshold
        // 8   1
        // 16  2
        // 32  3
        return floorLog2(n >> 2);
    }

    /**
     * Determine a threshold for the distance of a partition index {@code k} from the end
     * of the length to switch to heapselect. Applies to dual-pivot partitioning.
     * For convenience this method accepts length {@code n = r - l} and not the correct
     * {@code n = r - l + 1}.
     *
     * <p>This method is used to estimate when heapselect will be faster than repeat
     * partitioning using quickselect. Heapselect will use a heap of size {@code k}. When
     * {@code k} is very small, relative to the length of the data, it is faster to
     * perform a single pass heapselect.
     *
     * <p>Measurements on random data of variable length observed that the heapselect and
     * quickselect run times were approximately equal when {@code k == (n >> 6)}.
     *
     * <p>However quickselect performs faster on structured data introducing an additional
     * length dependence. This is observed to be approximately
     * {@code k == (n >> log3(n))}. This threshold may be too conservative for some data
     * (e.g. random data) and too high for other data. Note that this only occurs when the
     * distance from the end is very small:
     *
     * <pre>{@code
     *           n  (n >> log3(n)) / (double) n
     *           8     0.250000
     *          64    0.0625000
     *         512    0.0156250
     *        4096   0.00390625
     *       32768  0.000976563
     *      262144  0.000488281
     *     2097152  0.000122070
     *    16777216  3.05176e-05
     *   134217728  7.62939e-06
     *  1073741824  1.90735e-06
     * }</pre>
     *
     * <p>As such it is very infrequent that heapselect is useful; it can be used to quickly collect
     * a key at the end of the range. For example if a partition index cut a pair of target indices
     * (k, k+1) leaving k or k+1 at the edge.
     *
     * @param n Length.
     * @return distance
     */
    static int heapSelectEdgeDistanceDP(int n) {
        // Note:
        // 1. Heapselect is slow compared to insertion sort on small data.
        // 2. Heapselect can be much faster on large data.
        // Benchmarking shows that heapselect 1 is useful in DP partitioning
        // when length is ~21.
        // Here we set a conservative threshold using floor(log2(n)) but we
        // shift n to avoid heapselect on tiny lengths.
        // n   threshold
        // 8   0
        // 16  1
        // 32  2
        //return floorLog2(n >> 3);

        // Ideally this should be monotonic.
        // Applying a shift to n is not monotonic when log3(n+1) = log3(n) + 1.
        // Thus we use a power of 2:
        // n >> log3(n) ~ n / 2^log3(n) ~ 2^(log2(n) - log3(n))
        // == 1 << (log2(n) - log3(n))

        // compute log2(n) as (floor(log2(x)))
        // compute log3(n) as (floor(log2(x))+1) * 323 / 512
        //final int log2p1 = 32 - Integer.numberOfLeadingZeros(n);
        //return 1 << (log2p1 - 1 - ((log2p1 * 323) >> 9));

        // log2(n) - log3(n) = log2(n) - log2(n) / log2(3)
        // log2(3) ~ 512/323
        // => log2(n) - log2(n) * 323/512 = log2(n) * 189/512
        // Too high for small n
        // n   threshold
        // 1   1
        // 8   2
        // 64  4
        // May be better as a look-up table based on highest 1 bit
        return 1 << ((floorLog2(n & ~0x7) * 189) >>> 9);
    }

    /**
     * Search the data for the largest index {@code i} where {@code a[i]} is
     * less-than-or-equal to the {@code key}; else return {@code left - 1}.
     * <pre>
     * a[i] <= k    :   left <= i <= right, or (left - 1)
     * </pre>
     *
     * <p>The data is assumed to be in ascending order, otherwise the behaviour is undefined.
     * If the range contains multiple elements with the {@code key} value, the result index
     * may be any that match.
     *
     * <p>This is similar to using {@link java.util.Arrays#binarySearch(int[], int, int, int)
     * Arrays.binarySearch}. The method differs in:
     * <ul>
     * <li>use of an inclusive upper bound;
     * <li>returning the closest index with a value below {@code key} if no match was not found;
     * <li>performing no range checks: it is assumed {@code left <= right} and they are valid
     * indices into the array.
     * </ul>
     *
     * <p>An equivalent use of binary search is:
     * <pre>{@code
     * int i = Arrays.binarySearch(a, left, right + 1, k);
     * if (i < 0) {
     *     i = ~i - 1;
     * }
     * }</pre>
     *
     * <p>This specialisation avoids the caller checking the binary search result for the use
     * case when the presence or absence of a key is not important; only that the returned
     * index for an absence of a key is the largest index. When used on unique keys this
     * method can be used to update an upper index so all keys are known to be below a key:
     *
     * <pre>{@code
     * int[] keys = ...
     * // [i0, i1] contains all keys
     * int i0 = 0;
     * int i1 = keys.length - 1;
     * // Update: [i0, i1] contains all keys <= k
     * i1 = searchLessOrEqual(keys, i0, i1, k);
     * }</pre>
     *
     * @param a Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Key.
     * @return largest index {@code i} such that {@code a[i] <= k}, or {@code left - 1} if no
     * such index exists
     */
    static int searchLessOrEqual(int[] a, int left, int right, int k) {
        int l = left;
        int r = right;
        while (l <= r) {
            // Middle value
            final int m = (l + r) >>> 1;
            final int v = a[m];
            // Test:
            // l------m------r
            //        v  k      update left
            //     k  v         update right

            // Full binary search
            // Run time is up to log2(n) (fast exit on a match) but has more comparisons
            if (v < k) {
                l = m + 1;
            } else if (v > k) {
                r = m - 1;
            } else {
                // Equal
                return m;
            }

            // Modified search that does not expect a match
            // Run time is log2(n). Benchmarks as the same speed.
            //if (v > k) {
            //    r = m - 1;
            //} else {
            //    l = m + 1;
            //}
        }
        // Return largest known value below:
        // r is always moved downward when a middle index value is too high
        return r;
    }

    /**
     * Search the data for the smallest index {@code i} where {@code a[i]} is
     * greater-than-or-equal to the {@code key}; else return {@code right + 1}.
     * <pre>
     * a[i] >= k      :   left <= i <= right, or (right + 1)
     * </pre>
     *
     * <p>The data is assumed to be in ascending order, otherwise the behaviour is undefined.
     * If the range contains multiple elements with the {@code key} value, the result index
     * may be any that match.
     *
     * <p>This is similar to using {@link java.util.Arrays#binarySearch(int[], int, int, int)
     * Arrays.binarySearch}. The method differs in:
     * <ul>
     * <li>use of an inclusive upper bound;
     * <li>returning the closest index with a value above {@code key} if no match was not found;
     * <li>performing no range checks: it is assumed {@code left <= right} and they are valid
     * indices into the array.
     * </ul>
     *
     * <p>An equivalent use of binary search is:
     * <pre>{@code
     * int i = Arrays.binarySearch(a, left, right + 1, k);
     * if (i < 0) {
     *     i = ~i;
     * }
     * }</pre>
     *
     * <p>This specialisation avoids the caller checking the binary search result for the use
     * case when the presence or absence of a key is not important; only that the returned
     * index for an absence of a key is the smallest index. When used on unique keys this
     * method can be used to update a lower index so all keys are known to be above a key:
     *
     * <pre>{@code
     * int[] keys = ...
     * // [i0, i1] contains all keys
     * int i0 = 0;
     * int i1 = keys.length - 1;
     * // Update: [i0, i1] contains all keys >= k
     * i0 = searchGreaterOrEqual(keys, i0, i1, k);
     * }</pre>
     *
     * @param a Data.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Key.
     * @return largest index {@code i} such that {@code a[i] >= k}, or {@code right + 1} if no
     * such index exists
     */
    static int searchGreaterOrEqual(int[] a, int left, int right, int k) {
        int l = left;
        int r = right;
        while (l <= r) {
            // Middle value
            final int m = (l + r) >>> 1;
            final int v = a[m];
            // Test:
            // l------m------r
            //        v  k      update left
            //     k  v         update right

            // Full binary search
            // Run time is up to log2(n) (fast exit on a match) but has more comparisons
            if (v < k) {
                l = m + 1;
            } else if (v > k) {
                r = m - 1;
            } else {
                // Equal
                return m;
            }

            // Modified search that does not expect a match
            // Run time is log2(n). Benchmarks as the same speed.
            //if (v < k) {
            //    l = m + 1;
            //} else {
            //    r = m - 1;
            //}
        }
        // Smallest known value above
        // l is always moved upward when a middle index value is too low
        return l;
    }
}
