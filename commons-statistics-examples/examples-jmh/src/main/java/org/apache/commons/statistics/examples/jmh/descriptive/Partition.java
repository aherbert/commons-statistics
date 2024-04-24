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
import java.util.SplittableRandom;
import java.util.function.IntConsumer;
import java.util.function.IntUnaryOperator;
import java.util.function.Supplier;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;

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
 * <p>Use of sampling to identify a pivot that places {@code k} in the smaller partition is
 * performed in the SELECT algorithm of Floyd and Rivest [4]. The original algorithm partitions
 * on a single pivot. This was extended by Kiwiel to partition using two pivots either side
 * of {@code k} with high probability [5].
 *
 * <p>Confidence bounds for the number of iterations to reduce a partition length by 2<sup>-x</sup>
 * are provided in Valois [6].
 *
 * <p>A worst-case linear time algorithm PICK is described in Blum et al [7]. This uses the median
 * of medians as a partition element for selection which ensures a minimum fraction of the
 * elements are eliminated per iteration. This was extended to use an asymmetric pivot choice
 * with efficient reuse of the medians sample in the QuickselectAdpative algorithm of
 * Alexandrescu [8].
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
 * <li>Floyd and Rivest (1975)
 * Algorithm 489: The Algorithm SELECT—for Finding the ith Smallest of n elements.
 * Comm. ACM. 18 (3): 173.
 * <li>Kiwiel (2005)
 * On Floyd and Rivest's SELECT algorithm.
 * <a href="https://doi.org/10.1016/j.tcs.2005.06.032">
 * Theoretical Computer Science 347, 214-238</a>.
 * <li>Valois (2000)
 * Introspective sorting and selection revisited
 * <a href="https://doi.org/10.1002/(SICI)1097-024X(200005)30:6%3C617::AID-SPE311%3E3.0.CO;2-A">
 * Software: Practice and Experience 30, 617-638.</a>
 * <li>Blum, Floyd, Pratt, Rivest, and Tarjan (1973)
 * Time bounds for selection.
 * <a href="https://doi.org/10.1016%2FS0022-0000%2873%2980033-9">
 * Journal of Computer and System Sciences. 7 (4): 448–461</a>.
 * <li>Alexandrescu (2016)
 * Fast Deterministic Selection
 * <a href="https://arxiv.org/abs/1606.00484">arXiv:1606.00484</a>.
 * <li><a href="https://en.wikipedia.org/wiki/Quickselect">Quickselect (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Introsort">Introsort (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Introselect">Introselect (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Median_of_medians">Median of medians (Wikipedia)</a>
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
    /** Minimum selection size for quickselect/quicksort.
     * Below this switch to sortselect/insertion sort rather than selection.
     * Dual-pivot quicksort used 27 in Yaroslavskiy's original paper.
     * Changes to this value are only noticeable when the input array is small.
     *
     * <p>This is a legacy setting from when insertion sort was used as the stopper.
     * This has been replaced by edge selection functions. Using insertion sort
     * is slower as n must be sorted compared to an edge select that only sorts
     * up to n/2 from the edge. It is disabled by default but can be used for
     * benchmarking.
     *
     * <p>If using insertion sort as the stopper for quickselect:
     * <ul>
     * <li>Single-pivot: Benchmarking random data in range [96, 192] suggests a value of ~16 for n=1.
     * <li>Dual-pivot: Benchmarking random data in range [162, 486] suggests a value of ~27 for n=1
     * and increasing with higher n in the same range.
     * Dual-pivot sorting requires a value of ~120. If keys are saturated between k1 and kn
     * an increase to this threshold will gain full sort performance.
     * </ul> */
    static final int MIN_QUICKSELECT_SIZE = 0;
    /** Minimum size for heapselect.
     * Below this switch to insertion sort rather than selection. This is used to avoid
     * heap select on tiny data. */
    static final int MIN_HEAPSELECT_SIZE = 5;
    /** Minimum size for sortselect.
     * Below this switch to insertion sort rather than selection. This is used to avoid
     * sort select on tiny data. */
    static final int MIN_SORTSELECT_SIZE = 4;
    /** Default selection constant for edgeselect. */
    static final int EDGESELECT_CONSTANT = 20;
    /** Default sort selection constant for linearselect. Note that linear select variants
     * recursively call quickselect so very small lengths are included with an initial
     * small length. Using lengths of 1023-5 and 2043-53 indicate optimum performance around
     * 80 for median-of-medians when palcing the sample on the left. Adaptive linear methods
     * are faster and so this value is reduced. Quickselect adaptive has a value around 20-30. */
    static final int LINEAR_SORTSELECT_SIZE = 26;
    /** Default sub-sampling size to identify a single pivot. Off by default.
     * The SELECT algorithm of Floyd-Rivest uses 600. */
    static final int SUBSAMPLING_SIZE = Integer.MAX_VALUE;
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
    /** Default control flags. */
    static final int CONTROL_FLAGS = 0;
    /** Default single-pivot partition strategy. */
    static final SPStrategy SP_STRATEGY = SPStrategy.KBM;
    /** Default expand partition strategy. */
    static final ExpandStrategy EXPAND_STRATEGY = ExpandStrategy.T2;
    /** Default single-pivot linear select strategy. */
    static final LinearStrategy LINEAR_STRATEGY = LinearStrategy.RSA;
    /** Default edge select strategy. */
    static final EdgeSelectStrategy EDGE_STRATEGY = EdgeSelectStrategy.ESS;
    /** Default single-pivot stopper strategy. */
    static final StopperStrategy STOPPER_STRATEGY = StopperStrategy.SQA;

    /** Control flag for random sampling. */
    static final int FLAG_RANDOM_SAMPLING = 0x2;
    /** Control flag for vector swap of the sample. */
    static final int FLAG_MOVE_SAMPLE = 0x4;
    /** Control flag for random subset sampling. This creates the sample at the end
     * of the data and requires moving regions to reposition around the target k. */
    static final int FLAG_SUBSET_SAMPLING = 0x8;
    /** Control flag for biased nextInt(n) RNG. */
    static final int FLAG_BIASED_RANDOM = 0x10;
    /** Control flag for SplittableRandom RNG. */
    static final int FLAG_SPLITTABLE_RANDOM = 0x20;
    /** Control flag for MSWS RNG. */
    static final int FLAG_MSWS = 0x40;
    /** Control flag for quickselect adaptive to not use sampling mode. */
    static final int FLAG_QA_NO_SAMPLING = 0x1;
    /** Control flag for quickselect adaptive to propagate the no sampling mode recursively. */
    static final int FLAG_QA_PROPAGATE = 0x2;

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
     * invalidating this relationship as it has best case Order(n) performance.
     * Thus it is more robust on a variety of data input to use quickselect until the
     * distance from the edge is small and avoid heuristics to choose heap select.
     *
     * <p>A second advantage of sort select over heap select is that all indices closer to
     * the edge than the target index are also sorted. This allows selection of multiple
     * close indices to be performed with effectively the same speed. High density indices
     * can trigger use of sort select for small lengths to achieve a speed comparable to
     * quicksort. See {@link #dualPivotSortSelectSize(int, int, int)}.
     */
    static final int SORTSELECT_SIZE = 20;
    /** Threshold to use sub-sampling of the range to identify the single pivot.
     * Sub-sampling uses the Floyd-Rivest algorithm to partition a sample of the data to
     * identify a pivot so that the target element is in the smaller set after partitioning.
     * The original FR paper used 600 otherwise reverted to the target index as the pivot.
     * This implementation uses a sample to identify a median pivot which increases robustness
     * at small size on a variety of data and allows raising the original FR threshold. */
    static final int SELECT_SUB_SAMPLING_SIZE = 1200;
    /** Threshold to use a random sub-sample for the Floyd-Rivest algorithm.
     * Note: Random sampling is a redundant overhead on fully random data and will part
     * destroy sorted data. On data that is structured with repeat patterns, the
     * shuffle removes side-effects of patterns and stabilises performance where the
     * standard Floyd-Rivest algorithm (with a non-random local sample) will recurse excessively
     * and trigger a switch to heapselect. The threshold has been chosen at a level
     * where average performance over a variety of data distributions shows no performance loss.
     * Individual distributions may be better or worse at different thresholds. On random
     * data the impact is minimal; on sorted data the impact is approximately 10%. On data with
     * patterns that trigger excess recursion this can increase performance by an order of
     * magnitude. Note that heapselect will still be used to avoid worst-case quickselect
     * performance if this threshold is not appropriate for the input data. */
    static final int RANDOM_SUB_SAMPLING_SIZE = 25000;
    /** Increment used for the recursion counter. The counter will overflow to negative when
     * recursion has exceeded the maximum level. The counter is maintained in the upper bits
     * of the dual-pivot control flags. */
    private static final int RECURSION_INCREMENT = 1 << 20;
    /** Mask to extract the sort select size from the dual-pivot control flags. Currently
     * the bits below those used for the recursion counter are only used for the sort select size
     * so this can use a mask with all bits below the increment. */
    private static final int SORTSELECT_MASK = RECURSION_INCREMENT - 1;

    /** Message for an unsupported introselect configuration. */
    private static final String UNSUPPORTED_INTROSELECT = "Unsupported introselect: ";

    /** Transformer factory for double data with the behaviour of a JDK sort.
     * Moves NaN to the end of the data and handles signed zeros. Works on the data in-place. */
    private static final Supplier<DoubleDataTransformer> SORT_TRANSFORMER =
        DoubleDataTransformers.createFactory(NaNPolicy.INCLUDE, false);

    /** Minimum length between 2 pivots {@code p2 - p1} that requires a full sort. */
    private static final int SORT_BETWEEN_SIZE = 2;
    /** Mask to extract the positive index from an integer. */
    private static final int INDEX_MASK = Integer.MAX_VALUE;
    /** log2(e). Used for conversions: log2(x) = ln(x) * log2(e) */
    private static final double LOG2_E = 1.4426950408889634;

    /** Threshold to use repeated step left: 7 / 16. */
    private static final double STEP_LEFT = 0.4375;
    /** Threshold to use repeated step right: 9 / 16. */
    private static final double STEP_RIGHT = 0.5625;
    /** Threshold to use repeated step far-left: 1 / 12. */
    private static final double STEP_FAR_LEFT = 0.08333333333333333;
    /** Threshold to use repeated step far-right: 11 / 12. */
    private static final double STEP_FAR_RIGHT = 0.9166666666666666;

    /** Default instance. */
    private static final Partition DEFAULT = new Partition();

    // Use final for settings/objects used within partitioning functions

    /** A {@link PivotingStrategy} used for pivoting. */
    private final PivotingStrategy pivotingStrategy;
    /** A {@link DualPivotingStrategy} used for pivoting. */
    private final DualPivotingStrategy dualPivotingStrategy;

    /** Minimum size for quickselect when partitioning multiple keys.
     * Below this threshold partitioning using quickselect is stopped and a sort selection
     * is performed.
     *
     * <p>This threshold is also used in the sort methods to switch to insertion sort;
     * and in legacy partition methods which do not use edge selection. These may perform
     * key analysis using this value to determine saturation. */
    private final int minQuickSelectSize;
    /** Constant for edgeselect. */
    private final int edgeSelectConstant;
    /** Size for sortselect in the linearselect function. Optimal value for this is much higher
     * than for regular quickselect as the median-of-medians pivot strategy is expensive. */
    private int linearSortSelectSize = LINEAR_SORTSELECT_SIZE;
    /** Threshold to use sub-sampling of the range to identify the single pivot.
     * Sub-sampling uses the Floyd-Rivest algorithm to partition a sample of the data. This
     * identifies a pivot so that the target element is in the smaller set after partitioning.
     * The algorithm applies to searching for a single k.
     * Not all single-pivot {@link PairedKeyStrategy} methods support sub-sampling. It is
     * available to test in {@link #introselect(SPEPartition, double[], int, int, int, int)}.
     *
     * <p>Sub-sampling can provide up to a 2-fold performance gain on large random data.
     * It can have a 2-fold slowdown on some structured data (e.g. large shuffle data from
     * the Bentley and McIlroy test data). Large shuffle data also observes a larger performance
     * drop when using the SBM/BM/DNF partition methods (collect equal values) verses a
     * simple SP method ignoring equal values. Here large ~500,000; the behaviour
     * is observed at smaller sizes and becomes increasingly obvious at larger sizes.
     *
     * <p>The algorithm relies on partitioning of a subset to be representative of partitioning
     * of the entire data. Values in a small range partitioned around a pivot P
     * should create P in a similar location to its position in the entire fully sorted array,
     * i.e. ordering around P in [ll, rr] will be similar to P's order in [l, r]:
     * <pre>
     * target:                       k
     * subset:                  ll---P-------rr
     * sorted: l----------------------P-------------------------------------------r
     *                                Good pivot
     * </pre>
     *
     * <p>If the data in [ll, rr] is not representative then pivot selection based on a
     * subset creates bad pivot choices and performance is worse than using a
     * {@link PivotingStrategy}.
     * <pre>
     * target:                       k
     * subset:                 ll----P-------rr
     * sorted: l------------------------------------------P----------------------r
     *                                                    Bad pivot
     * </pre>
     *
     * <p>Use of the Floyd-Rivest subset sampling is not always an improvement and is data
     * dependent. The type of data cannot be known by the partition algorithm before processing.
     * Thus the Floyd-Rivest subset sampling is more suitable as an option to be enabled by
     * user settings.
     *
     * <p>See <a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">
     * Floyd-Rivest Algorithm (Wikipedia)</a>.
     *
     * <pre>
     * Floyd and Rivest (1975)
     * Algorithm 489: The Algorithm SELECT—for Finding the ith Smallest of n elements.
     * Comm. ACM. 18 (3): 173.
     * </pre> */
    private final int subSamplingSize;

    // Use final for settings used to configure partitioning functions

    /** Setting to indicate strategy for processing of multiple keys. */
    private KeyStrategy keyStrategy = KEY_STRATEGY;
    /** Setting to indicate strategy for processing of 1 or 2 keys. */
    private PairedKeyStrategy pairedKeyStrategy = PAIRED_KEY_STRATEGY;

    /** Multiplication factor {@code m} applied to the length based recursion factor {@code x}.
     * The recursion is set using {@code m * x + c}.
     * Also used for the multiple of the original length to check the sum of the partition length
     * for poor quickselect partitions. */
    private double recursionMultiple = RECURSION_MULTIPLE;
    /** Constant {@code c} added to the length based recursion factor {@code x}.
     * The recursion is set using {@code m * x + c}.
     * Also used for the number of iterations before checking the partition length has been
     * reduced by a given factor, e.g. half. */
    private int recursionConstant = RECURSION_CONSTANT;
    /** Compression level for a {@link CompressedIndexSet} (in [1, 31]). */
    private int compression = COMPRESSION_LEVEL;
    /** Control flags level for Floyd-Rivest sub-sampling. */
    private int controlFlags = CONTROL_FLAGS;
    /** Consumer for the recursion level reached during partitioning. Used to analyse
     * the distribution of the recursion for different input data. */
    private IntConsumer recursionConsumer = i -> { /* no-op */ };

    /** The single-pivot partition function. */
    private SPEPartition spFunction;
    /** The expand partition function. */
    private ExpandPartition expandFunction;
    /** The single-pivot linear partition function. */
    private SPEPartition linearSpFunction;
    /** Selection function used when {@code k} is close to the edge of the range. */
    private SelectFunction edgeSelection;
    /** Selection function used when quickselect progress is poor. */
    private SelectFunction stopperSelection;
    /** Mask applied to the quickselect adaptive flags before mutual recursion.
     * If the sign-bit is not masked out then no sampling mode is enabled through
     * the call stack. */
    private int qaFlagMask;

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
        /** Use a dedicated single key method that returns information about (k+1).
         * Use recursion depth to trigger the stopper select. */
        PAIRED_KEYS,
        /** Use a dedicated single key method that returns information about (k+1).
         * Recursion is monitored by checking the partition is reduced by 2<sup>-x</sup> after
         * {@code c} iterations where {@code x} is the
         * {@link #setRecursionConstant(int) recursion constant} and {@code c} is the
         * {@link #setRecursionMultiple(double) recursion multiple} */
        PAIRED_KEYS_2,
        /** Use a dedicated single key method that returns information about (k+1).
         * Use a multiple of the sum of the length of all partitions to trigger the stopper select. */
        PAIRED_KEYS_LEN,
        /** Use a method that accepts two separate keys. */
        TWO_KEYS,
        /** Use a method that accepts two keys to define a range.
         * Recursion is monitored by checking the partition is reduced by 2<sup>-x</sup> after
         * {@code c} iterations where {@code x} is the
         * {@link #setRecursionConstant(int) recursion constant} and {@code c} is the
         * {@link #setRecursionMultiple(double) recursion multiple} */
        KEY_RANGE,
        /** Use an {@link SearchableInterval} covering the keys. This will reuse a multi-key
         * strategy with keys that are a very small range. */
        SEARCHABLE_INTERVAL,
        /** Use an {@link UpdatingInterval} covering the keys. This will reuse a multi-key
         * strategy with keys that are a very small range. */
        UPDATING_INTERVAL;
    }

    /**
     * Define the strategy for single-pivot partitioning. Partitioning may be binary
     * ({@code <, >}), or ternary ({@code <, ==, >}) by collecting values equal to the
     * pivot value. Typically partitioning will use two pointers i and j to traverse the
     * sequence from either end; or a single pointer i for a single pass.
     *
     * <p>Binary partitioning will be faster for quickselect when no equal elements are
     * present. As duplicates become increasingly likely a ternary partition will be
     * faster for quickselect to avoid repeat processing of values (that matched the
     * previous pivot) on the next iteration. The type of ternary partition with the best
     * performance depends on the number of duplicates. In the extreme case of 1 or 2
     * unique elements it is more likely to match the {@code ==, !=} comparison to the
     * pivot than {@code <, >} (see {@link #DNF3}). An ideal ternary scheme should have
     * little impact on data with no repeats, and significantly improve performance as the
     * number of repeat elements increases.
     *
     * <p>Binary partitioning will skip over values already {@code <, >}, or
     * {@code <=, =>} to the pivot value; otherwise values at the pointers i and j are
     * swapped. If using {@code <, >} then values can be placed at either end of the
     * sequence that are {@code >=, <=} respectively to act as sentinels during the scan.
     * This is always possible in binary partitioning as the pivot can be one sentinel;
     * any other value will be either {@code <=, =>} to the pivot and so can be used at
     * one or the other end as appropriate. Note: Many schemes omit using sentinels. Modern
     * processor branch prediction nullifies the cost of checking indices remain within
     * the {@code [left, right]} bounds. However placing sentinels is a negligible cost
     * and at least simplifies the code for the region traversal.
     *
     * <p>Bentley-McIlroy ternary partitioning schemes move equal values to the ends
     * during the traversal, these are moved to the centre after the pass. This may use
     * minimal swaps based on region sizes. Note that values already {@code <, >} are not
     * moved during traversal allowing moves to be minimised.
     *
     * <p>Dutch National Flag schemes move non-equal values to either end and finish with
     * the equal value region in the middle. This requires that every element is moved
     * during traversal, even if already {@code <, >}. This can be mitigated by fast-forward
     * of pointers at the current {@code <, >} end points until the condition is not true.
     *
     * @see SPEPartition
     */
    enum SPStrategy {
        /**
         * Single-pivot partitioning. Uses a method adapted from Floyd and Rivest (1975)
         * which uses sentinels to avoid bounds checks on the i and j pointers.
         * This is a baseline for the maximum speed when no equal elements are present.
         */
        SP,
        /**
         * Bentley-McIlroy ternary partitioning. Requires bounds checks on the i and j
         * pointers during traversal. Comparisons to the pivot use {@code <=, =>} and a
         * second check for {@code ==} if the first is true.
         */
        BM,
        /**
         * Sedgewick's Bentley-McIlroy ternary partitioning. Requires bounds checks on the
         * j pointer during traversal. Comparisons to the pivot use {@code <, >} and a
         * second check for {@code ==} when both i and j have stopped.
         */
        SBM,
        /**
         * Kiwiel's Bentley-McIlroy ternary partitioning. Similar to Sedgewick's BM but
         * avoids bounds checks on both pointers during traversal using sentinels.
         * Comparisons to the pivot use {@code <, >} and a second check for {@code ==}
         * when both i and j have stopped. Handles i and j meeting at the pivot without a
         * swap.
         */
        KBM,
        /**
         * Dutch National Flag partitioning. Single pointer iteration using {@code <, >}
         * comparisons to move elements to the edges. Fast-forwards any initial {@code <}
         * region. The {@code ==} region is filled with the pivot after region traversal.
         */
        DNF1,
        /**
         * Dutch National Flag partitioning. Single pointer iteration using {@code <, >}
         * comparisons to move elements to the edges. Fast-forwards any initial {@code <}
         * region. The {@code >} region uses fast-forward to reduce swaps. The {@code ==}
         * region is filled with the pivot after region traversal.
         */
        DNF2,
        /**
         * Dutch National Flag partitioning. Single pointer iteration using {@code !=}
         * comparison to identify elements to move to the edges, then {@code <, >}
         * comparisons. Fast-forwards any initial {@code <} region. The {@code >} region
         * uses fast-forward to reduce swaps. The {@code ==} region is filled during
         * traversal.
         */
        DNF3;
    }

    /**
     * Define the strategy for expanding a partition. This function is used when
     * partitioning has used a sample located within the range to find the pivot.
     * The remaining range below and above the sample can be partitioned without
     * re-processing the sample.
     *
     * <p>Schemes may be binary ({@code <, >}), or ternary ({@code <, ==, >}) by
     * collecting values equal to the pivot value. Schemes may process the
     * unpartitioned range below and above the partitioned middle using a sweep
     * outwards towards the ends; or start at the ends and sweep inwards towards
     * the partitioned middle.
     *
     * @see ExpandPartition
     */
    enum ExpandStrategy {
        /** Use the current {@link SPStrategy} parition method. This will not expand
         * the partition but will Partition the Entire Range (PER). This can be used
         * to test if the implementation of expand are efficient. */
        PER,
        /** Ternary partition method 1. Sweeps outwards and uses sentinels at the ends
         * to avoid pointer range checks. Equal values are move directly into the
         * central pivot range. */
        T1,
        /** Ternary partition method 2. Sweeps inwards and uses sentinels at the ends
         * to avoid pointer range checks. Equal values are move to the outer edges;
         * these are swapped to the pivot region in the final step using minimum moves.
         * In the event of no equal values this requires no additional swaps. */
        T2,
        /** Binary partition method 1. Sweeps outwards and uses sentinels at the ends
         * to avoid pointer range checks. */
        B1,
        /** Binary partition method 2. Sweeps inwards and uses sentinels at the ends
         * to avoid pointer range checks. */
        B2,
    }

    /**
     * Define the strategy for the linear select single-pivot partition function.
     * Linear select functions use a deterministic sample to find a pivot value
     * that will eliminate at least a set fraction of the range. After the sample
     * has been processed to find a pivot the entire range is partitioned. This
     * can be done by re-processing the entire range, or expanding the partition.
     *
     * @see SPStrategy
     * @see ExpandStrategy
     * @see SPEPartition
     * @see ExpandPartition
     */
    enum LinearStrategy {
        /** Uses the Blum, Floyd, Pratt, Rivest, and Tarjan (BFPRT) median-of-medians algorithm
         * with medians of 5. This is the baseline version that creates the median sample
         * at the left end and repartitions the entire range using the pivot. */
        BFPRT,
        /** Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
         * with medians of 3. This is the baseline version that creates the median sample
         * at the left end and repartitions the entire range using the pivot. */
        RS,
        /** Uses the Blum, Floyd, Pratt, Rivest, and Tarjan (BFPRT) median-of-medians algorithm
         * with medians of 5. This is the improved version that creates the median sample
         * in the centre and expands the partition around the pivot sample. */
        BFPRT_IM,
        /** Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
         * with medians of 3. This is the improved version that creates the median sample
         * in the centre and expands the partition around the pivot sample. */
        RS_IM,
        /** Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
         * with medians of 3. This is the adaptive version that creates the median sample
         * in the centre and expands the partition around the pivot sample; the adaption
         * is to use k to define the pivot in the sample instead of using the median. */
        RSA;
    }

    /**
     * Define the strategy for selecting {@code k} close to the edge.
     * <p>These are named to allow regex identification for dynamic configuration
     * in benchmarking using the name.
     */
    enum EdgeSelectStrategy {
        /** Use heapselect version 1. Selects {@code k} and an additional
         * {@code c} elements closer to the edge than {@code k} using a heap
         * structure. */
        ESH,
        /** Use heapselect version 2. Differs from {@link #ESH} in the
         * final unwinding of the heap to sort the range {@code [ka, kb]};
         * the heap construction is identical. */
        ESH2,
        /** Use sortselect which uses an insertion sort to maintain {@code k}
         * and all elements closer to the edge as sorted. */
        ESS;
    }

    /**
     * Define the strategy for selecting {@code k} when quickselect progress is poor
     * (worst case is quadratic). This should be a method providing good worst-case
     * performance.
     * <p>These are named to allow regex identification for dynamic configuration
     * in benchmarking using the name.
     */
    enum StopperStrategy {
        /** Use heapselect version 1. Selects {@code k} and an additional
         * {@code c} elements closer to the edge than {@code k}. Heapselect
         * provides increasingly slower performance with distance from the edge.
         * It has better worst-case performance than quickselect. */
        SSH,
        /** Use heapselect version 2. Differs from {@link #SSH} in the
         * final unwinding of the heap to sort the range {@code [ka, kb]};
         * the heap construction is identical. */
        SSH2,
        /** Use a linear selection algorithm with Order(n) worst-case performance.
         * This is a median-of-medians using medians of size 5 */
        SLS,
        /** Use the quickselect adaptive algorithm with Order(n) worst-case performance. */
        SQA;
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
     * Select function.
     *
     * <p>Used to define the function to call when {@code k} is close
     * to the edge; or when quickselect progress is poor. This allows
     * the edge-select or stopper-function to be configured using parameters.
     */
    @FunctionalInterface
    interface SelectFunction {
        /**
         * Partition the elements between {@code ka} and {@code kb}.
         * It is assumed {@code left <= ka <= kb <= right}.
         *
         * @param a Data array to use to find out the K<sup>th</sup> value.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower index to select.
         * @param kb Upper index to select.
         */
        void partition(double[] a, int left, int right, int ka, int kb);
    }

    /**
     * Single-pivot partition method handling a pre-partitioned range in the centre.
     */
    @FunctionalInterface
    interface ExpandPartition {
        /**
         * Expand a partition around a single pivot. Partitioning exchanges array
         * elements such that all elements smaller than pivot are before it and all
         * elements larger than pivot are after it. The central region is already
         * partitioned.
         *
         * <pre>{@code
         * |l             |s   |p0 p1|   e|                r|
         * |    ???       | <P | ==P | >P |        ???      |
         * }</pre>
         *
         * <p>Note: Requires that the range contains no NaN values.
         *
         * <p>This method returns 2 points describing the pivot range of equal values.
         * <pre>{@code
         * |l                  |k0 k1|                     r|
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
         * @param start Start of the partition range (inclusive).
         * @param end End of the partitioned range (inclusive).
         * @param pivot0 Lower pivot location (inclusive).
         * @param pivot1 Upper pivot location (inclusive).
         * @param upper Upper bound (inclusive) of the pivot range [k1].
         * @return Lower bound (inclusive) of the pivot range [k0].
         */
        int partition(double[] a, int left, int right, int start, int end,
            int pivot0, int pivot1, int[] upper);
    }

    /**
     * Constructor with defaults.
     */
    Partition() {
        this(PIVOTING_STRATEGY, DUAL_PIVOTING_STRATEGY, MIN_QUICKSELECT_SIZE,
            EDGESELECT_CONSTANT, SUBSAMPLING_SIZE);
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
            EDGESELECT_CONSTANT, SUBSAMPLING_SIZE);
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
            EDGESELECT_CONSTANT, SUBSAMPLING_SIZE);
    }

    /**
     * Constructor with specified pivoting strategy; quickselect size; and heapselect configuration.
     *
     * <p>Used to test single-pivot quickselect.
     *
     * @param pivotingStrategy Pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param edgeSelectConstant Length constant used for heap select distance from end threshold.
     * @param subSamplingSize Size threshold to use sub-sampling for single-pivot selection.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(PivotingStrategy pivotingStrategy,
        int minQuickSelectSize, int edgeSelectConstant, int subSamplingSize) {
        this(pivotingStrategy, DUAL_PIVOTING_STRATEGY, minQuickSelectSize, edgeSelectConstant,
            subSamplingSize);
    }

    /**
     * Constructor with specified dual-pivoting strategy; quickselect size; and heapselect configuration.
     *
     * <p>Used to test dual-pivot quickselect.
     *
     * @param dualPivotingStrategy Dual pivoting strategy to use.
     * @param minQuickSelectSize Minimum size for quickselect.
     * @param edgeSelectConstant Length constant used for heap select distance from end threshold.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(DualPivotingStrategy dualPivotingStrategy,
        int minQuickSelectSize, int edgeSelectConstant) {
        this(PIVOTING_STRATEGY, dualPivotingStrategy, minQuickSelectSize,
            edgeSelectConstant, SUBSAMPLING_SIZE);
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
     * @param edgeSelectConstant Length constant used for distance from end threshold.
     * @param subSamplingSize Size threshold to use sub-sampling for single-pivot selection.
     * @throws IllegalArgumentException If the shift is not in {@code [0, 31]}.
     */
    Partition(PivotingStrategy pivotingStrategy, DualPivotingStrategy dualPivotingStrategy,
        int minQuickSelectSize, int edgeSelectConstant, int subSamplingSize) {
        this.pivotingStrategy = pivotingStrategy;
        this.dualPivotingStrategy = dualPivotingStrategy;
        this.minQuickSelectSize = minQuickSelectSize;
        this.edgeSelectConstant = edgeSelectConstant;
        this.subSamplingSize = subSamplingSize;
        // Default strategies
        setSPStrategy(SP_STRATEGY);
        setEdgeSelectStrategy(EDGE_STRATEGY);
        setStopperStrategy(STOPPER_STRATEGY);
        setExpandStrategy(EXPAND_STRATEGY);
        setLinearStrategy(LINEAR_STRATEGY);
    }

    /**
     * Sets the single-pivot partition strategy.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setSPStrategy(SPStrategy v) {
        switch (v) {
        case BM:
            spFunction = Partition::partitionBM;
            break;
        case DNF1:
            spFunction = Partition::partitionDNF1;
            break;
        case DNF2:
            spFunction = Partition::partitionDNF2;
            break;
        case DNF3:
            spFunction = Partition::partitionDNF3;
            break;
        case KBM:
            spFunction = Partition::partitionKBM;
            break;
        case SBM:
            spFunction = Partition::partitionSBM;
            break;
        case SP:
            spFunction = Partition::partitionSP;
            break;
        default:
            throw new IllegalArgumentException("Unknown single-pivot strategy: " + v);
        }
        return this;
    }

    /**
     * Sets the single-pivot partition expansion strategy.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setExpandStrategy(ExpandStrategy v) {
        switch (v) {
        case PER:
            // Partition the entire range
            expandFunction = (a, left, right, start, end, pivot0, pivot1, upper) ->
                spFunction.partition(a, left, right, (pivot0 + pivot1) >>> 1, upper);
            break;
        case T1:
            expandFunction = Partition::expandPartitionT1;
            break;
        case B1:
            expandFunction = Partition::expandPartitionB1;
            break;
        case T2:
            expandFunction = Partition::expandPartitionT2;
            break;
        case B2:
            expandFunction = Partition::expandPartitionB2;
            break;
        default:
            throw new IllegalArgumentException("Unknown expand strategy: " + v);
        }
        return this;
    }

    /**
     * Sets the single-pivot linear select strategy.
     *
     * <p>Note: This value should be set after either {@link #setSPStrategy(SPStrategy)}
     * or {@link #setExpandStrategy(ExpandStrategy)}; the linear select strategy
     * will partition remaining range after computing a pivot from a sample by
     * single-pivot partitioning or by expanding the partition.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setLinearStrategy(LinearStrategy v) {
        switch (v) {
        case BFPRT:
            linearSpFunction = this::linearBFPRTBaseline;
            break;
        case RS:
            linearSpFunction = this::linearRepeatedStepBaseline;
            break;
        case BFPRT_IM:
            linearSpFunction = this::linearBFPRTImproved;
            break;
        case RS_IM:
            linearSpFunction = this::linearRepeatedStepImproved;
            break;
        case RSA:
            linearSpFunction = this::linearRepeatedStepAdaptive;
            break;
        default:
            throw new IllegalArgumentException("Unknown linear strategy: " + v);
        }
        return this;
    }

    /**
     * Sets the edge-select strategy.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setEdgeSelectStrategy(EdgeSelectStrategy v) {
        switch (v) {
        case ESH:
            edgeSelection = Partition::heapSelectRange;
            break;
        case ESH2:
            edgeSelection = Partition::heapSelectRange2;
            break;
        case ESS:
            edgeSelection = Partition::sortSelectRange;
            break;
        default:
            throw new IllegalArgumentException("Unknown edge select: " + v);
        }
        return this;
    }

    /**
     * Sets the stopper strategy (when quickselect progress is poor).
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setStopperStrategy(StopperStrategy v) {
        switch (v) {
        case SSH:
            stopperSelection = Partition::heapSelectRange;
            break;
        case SSH2:
            stopperSelection = Partition::heapSelectRange2;
            break;
        case SLS:
            // Linear select does not match the interface as it:
            // - requires the single-pivot partition function
            // - uses a bounds array to allow minimising the partition region size after pivot selection
            stopperSelection = (a, l, r, ka, kb) -> linearSelect(getSPFunction(),
                a, l, r, ka, kb, new int[2]);
            break;
        case SQA:
            // Linear select does not match the interface as it:
            // - uses a bounds array to allow minimising the partition region size after pivot selection
            // - uses control flags to set sampling mode on/off
            stopperSelection = (a, l, r, ka, kb) -> quickSelectAdaptive(a, l, r, ka, kb, new int[1],
                (controlFlags & FLAG_QA_NO_SAMPLING) != 0 ? -1 : 0);
            break;
        default:
            throw new IllegalArgumentException("Unknown stopper: " + v);
        }
        return this;
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
     * Sets the control flags for Floyd-Rivest sub-sampling.
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setControlFlags(int v) {
        this.controlFlags = v;
        // Set the QA mask
        qaFlagMask = 0;
        if ((v & FLAG_QA_NO_SAMPLING | FLAG_QA_PROPAGATE) != 0) {
            qaFlagMask = Integer.MIN_VALUE;
        }
        return this;
    }

    /**
     * Sets the recursion consumer. This is called with the value of the recursion
     * counter immediately before the introselect routine returns.
     *
     * @param v Value.
     */
    void setRecursionConsumer(IntConsumer v) {
        this.recursionConsumer = Objects.requireNonNull(v);
    }

    /**
     * Sets the size for sortselect for the linearselect algorithm.
     * Must be above 0 for the algorithm to return (else an infinite loop occurs).
     *
     * @param v Value.
     * @return {@code this} for chaining
     */
    Partition setLinearSortSelectSize(int v) {
        if (v < 1) {
            throw new IllegalArgumentException("Bad linear sortselect size: " + v);
        }
        this.linearSortSelectSize = v;
        return this;
    }

    /**
     * Gets the single-pivot partition function.
     *
     * @return the single-pivot partition function
     */
    SPEPartition getSPFunction() {
        return spFunction;
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * Does not respect the ordering of signed zeros.
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
            // Use the left child if right doesn't exist, or it is greater
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
     * Does not respect the ordering of signed zeros.
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
                // No left child
                break;
            }
            // Use the left child if right doesn't exist, or it is less
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * <p>Note: Requires that the range contains no NaN values. Does not respect the
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
     * Does not respect the ordering of signed zeros.
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
     * Does not respect the ordering of signed zeros.
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
     * Sort the data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method. Signed zeros
     * are corrected when encountered during processing.
     *
     * @param data Values.
     */
    void sortSBM(double[] data) {
        // Handle NaN
        final int right = sortNaN(data);
        sort((SPEPartitionFunction) this::partitionSBMWithZeros, data, right);
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
     * Sort the data using an introsort.
     *
     * <p>Uses the configured single-pivot quicksort method; falling back
     * to heapsort when quicksort recursion is slow.
     *
     * @param data Values.
     */
    void sortISP(double[] data) {
        // NaN processing is done in the introsort method
        introsort(getSPFunction(), data);
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
                pivotingStrategy.pivotIndex(a, l, r, l),
                upper);
            final int p1 = upper[0];

            // Recurse right side
            introsort(part, a, p1 + 1, r, --maxDepth);
            // Continue on the left side
            r = p0 - 1;
        }
    }

    /**
     * Sort the data using an introsort.
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
     * the fall-back on poor convergence of the quickselect is controlled by
     * current configuration.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices (assumed to be strictly positive).
     */
    void introselect(double[] a, int[] k, int count) {
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
            introselect(getSPFunction(), a, end - 1, k, n);
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
     * <p>Uses an introselect variant. The single-pivot quickselect is provided as an argument;
     * the fall-back on poor convergence of the quickselect is controlled by
     * current configuration.
     *
     * <p>The partition method is not required to handle signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices (assumed to be strictly positive).
     */
    private void introselect(SPEPartition part, double[] a, int[] k, int count) {
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
     * the fall-back on poor convergence of the quickselect is controlled by
     * current configuration.
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
            // Dedicated methods for a single key. These use different strategies
            // to trigger the stopper on quickselect recursion
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                introselect(part, a, 0, right, k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS_2) {
                // This uses the configured recursion constant c.
                // The length must halve every c iterations.
                introselect2(part, a, 0, right, k[0]);
            } else if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS_LEN) {
                introselect(part, a, 0, right, k[0]);
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two separate keys using the same key
                introselect(part, a, 0, right, k[0], k[0], maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.KEY_RANGE) {
                // Dedicated method for a range of keys using the same key
                introselect2(part, a, 0, right, k[0], k[0]);
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
            // Dedicated method for a single key, returns information about k+1
            if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS) {
                final int p = introselect(part, a, 0, right, k[0], maxDepth);
                // p <= k to signal k+1 is unsorted, or p+1 is a pivot.
                // if k is sorted, and p+1 is sorted, k+1 is sorted if k+1 == p.
                if (p > k[1]) {
                    selectMinIgnoreZeros(a, k[1], p);
                }
            } else if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS_2) {
                final int p = introselect2(part, a, 0, right, k[0]);
                if (p > k[1]) {
                    selectMinIgnoreZeros(a, k[1], p);
                }
            } else if (pairedKeyStrategy == PairedKeyStrategy.PAIRED_KEYS_LEN) {
                final int p = introselect(part, a, 0, right, k[0]);
                if (p > k[1]) {
                    selectMinIgnoreZeros(a, k[1], p);
                }
            } else if (pairedKeyStrategy == PairedKeyStrategy.TWO_KEYS) {
                // Dedicated method for two separate keys
                // Note: This can handle keys that are not adjacent
                // e.g. keys near opposite ends without a partition step.
                final int ka = Math.min(k[0], k[1]);
                final int kb = Math.max(k[0], k[1]);
                introselect(part, a, 0, right, ka, kb, maxDepth);
            } else if (pairedKeyStrategy == PairedKeyStrategy.KEY_RANGE) {
                // Dedicated method for a range of keys using the same key
                final int ka = Math.min(k[0], k[1]);
                final int kb = Math.max(k[0], k[1]);
                introselect2(part, a, 0, right, ka, kb);
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
            // It is possible to use edgeselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d2----
            final int d1 = k - l;
            final int d2 = r - k;
            if (Math.min(d1, d2) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            if (maxDepth == 0) {
                // Too much recursion
                // Note: For testing the Floyd-Rivest algorithm we trigger the recursion
                // consumer as a signal that FR failed due to a non-representative sample.
                recursionConsumer.accept(maxDepth);
                stopperSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            // Pick a pivot and partition
            int pivot;
            // length - 1
            int n = r - l;
            if (n > subSamplingSize) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                ++n;
                final int ith = k - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (k - ith * s / n + sd));
                final int rr = Math.min(r, (int) (k + (n - ith) * s / n + sd));
                // Optional random sampling
                if ((controlFlags & FLAG_RANDOM_SAMPLING) != 0) {
                    final IntUnaryOperator rng = createRNG(n, k);
                    // Shuffle [ll, k) from [l, k)
                    if (ll > l) {
                        for (int i = k; i > ll;) {
                            // l + rand [0, i - l + 1) : i is currently i+1
                            final int j = l + rng.applyAsInt(i - l);
                            final double t = a[--i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (k, r]
                    if (rr < r) {
                        for (int i = k; i < rr;) {
                            // r - rand [0, r - i + 1) : i is currently i-1
                            final int j = r - rng.applyAsInt(r - i);
                            final double t = a[++i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                }
                introselect(part, a, ll, rr, k, lnNtoMaxDepthSinglePivot(z));
                pivot = k;
            } else {
                // default pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, k);
            }

            final int p0 = part.partition(a, l, r, pivot, upper);
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
     * <p>Recursion is monitored by checking the partition is reduced by 2<sup>-x</sup> after
     * {@code c} iterations where {@code x} is the
     * {@link #setRecursionConstant(int) recursion constant} and {@code c} is the
     * {@link #setRecursionMultiple(double) recursion multiple} (variables reused for convenience).
     * Confidence bounds for dividing a length by 2<sup>-x</sup> are provided in Valois (2000)
     * as {@code c = floor((6/5)x) + b}:
     * <pre>
     * b  confidence (%)
     * 2  76.56
     * 3  92.92
     * 4  97.83
     * 5  99.33
     * 6  99.79
     * </pre>
     * <p>Ideally {@code c >= 3} using {@code x = 1}. E.g. We can use 3 iterations to be 76%
     * confident the sequence will divide in half; or 7 iterations to be 99% confident the
     * sequence will divide into a quarter. A larger factor {@code b} reduces the sensitivity
     * of introspection.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Index.
     * @return the index {@code p}
     */
    private int introselect2(SPEPartition part, double[] a, int left, int right, int k) {
        int l = left;
        int r = right;
        final int[] upper = {0};
        int counter = (int) recursionMultiple;
        int threshold = (right - left) >>> recursionConstant;
        int depth = singlePivotMaxDepth(right - left);
        while (true) {
            // It is possible to use edgeselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d2----
            final int d1 = k - l;
            final int d2 = r - k;
            if (Math.min(d1, d2) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            // length - 1
            int n = r - l;
            depth--;
            if (--counter < 0) {
                if (n > threshold) {
                    // Did not reduce the length after set number of iterations.
                    // Here riselect (Valois (2000)) would use random points to choose the pivot
                    // to inject entropy and restart. This continues until the sum of the partition
                    // lengths is too high (twice the original length). Here we just switch.

                    // Note: For testing we trigger the recursion consumer
                    recursionConsumer.accept(depth);
                    stopperSelection.partition(a, l, r, k, k);
                    // Last known unsorted value >= k
                    return r;
                }
                // Once the confidence has been achieved we use (6/5)x with x=1.
                // So check every 5/6 iterations that the length is halving.
                if (counter == -5) {
                    counter = 1;
                }
                threshold >>>= 1;
            }

            // Pick a pivot and partition
            int pivot;
            if (n > subSamplingSize) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                ++n;
                final int ith = k - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (k - ith * s / n + sd));
                final int rr = Math.min(r, (int) (k + (n - ith) * s / n + sd));
                // Optional random sampling
                if ((controlFlags & FLAG_RANDOM_SAMPLING) != 0) {
                    final IntUnaryOperator rng = createRNG(n, k);
                    // Shuffle [ll, k) from [l, k)
                    if (ll > l) {
                        for (int i = k; i > ll;) {
                            // l + rand [0, i - l + 1) : i is currently i+1
                            final int j = l + rng.applyAsInt(i - l);
                            final double t = a[--i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (k, r]
                    if (rr < r) {
                        for (int i = k; i < rr;) {
                            // r - rand [0, r - i + 1) : i is currently i-1
                            final int j = r - rng.applyAsInt(r - i);
                            final double t = a[++i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                }
                // Sample recursion restarts from [ll, rr]
                introselect2(part, a, ll, rr, k);
                pivot = k;
            } else {
                // default pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, k);
            }

            final int p0 = part.partition(a, l, r, pivot, upper);
            final int p1 = upper[0];

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
     * <p>Recursion is monitored by checking the sum of partition lengths is less than
     * {@code m * (r - l)} where {@code m} is the
     * {@link #setRecursionMultiple(double) recursion multiple}.
     * Ideally {@code c} should be a value above 1.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Index.
     * @return the index {@code p}
     */
    private int introselect(SPEPartition part, double[] a, int left, int right, int k) {
        int l = left;
        int r = right;
        final int[] upper = {0};
        // Set the limit on the sum of the length. Since the length is subtracted at the start
        // of the loop use (1 + recursionMultiple).
        long limit = (long) ((1 + recursionMultiple) * (right - left));
        int depth = singlePivotMaxDepth(right - left);
        while (true) {
            // It is possible to use edgeselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d2----
            final int d1 = k - l;
            final int d2 = r - k;
            if (Math.min(d1, d2) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            // length - 1
            int n = r - l;
            limit -= n;
            depth--;

            if (limit < 0) {
                // Excess total partition length
                // Note: For testing we trigger the recursion consumer
                recursionConsumer.accept(depth);
                stopperSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            // Pick a pivot and partition
            int pivot;
            if (n > subSamplingSize) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                ++n;
                final int ith = k - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (k - ith * s / n + sd));
                final int rr = Math.min(r, (int) (k + (n - ith) * s / n + sd));
                // Optional random sampling
                if ((controlFlags & FLAG_RANDOM_SAMPLING) != 0) {
                    final IntUnaryOperator rng = createRNG(n, k);
                    // Shuffle [ll, k) from [l, k)
                    if (ll > l) {
                        for (int i = k; i > ll;) {
                            // l + rand [0, i - l + 1) : i is currently i+1
                            final int j = l + rng.applyAsInt(i - l);
                            final double t = a[--i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (k, r]
                    if (rr < r) {
                        for (int i = k; i < rr;) {
                            // r - rand [0, r - i + 1) : i is currently i-1
                            final int j = r - rng.applyAsInt(r - i);
                            final double t = a[++i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                }
                // Sample recursion restarts from [ll, rr]
                introselect(part, a, ll, rr, k);
                pivot = k;
            } else {
                // default pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, k);
            }

            final int p0 = part.partition(a, l, r, pivot, upper);
            final int p1 = upper[0];

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

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                return;
            }

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
                Math.min(d1 + d3, Math.min(d2, d4)) < edgeSelectConstant) {
                // Too much recursion, or ka1 and kb1 are both close to the ends
                // Note: Does not use the edgeSelection function as the indices are not a range
                heapSelectPair(a, l, r, ka1, kb1);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r, ka),
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
     * Partition the array such that index {@code k} corresponds to its
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [ka, kb]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
     * }</pre>
     *
     * <p>This function accepts indices {@code [ka, kb]} that define the
     * range of indices to partition. It is expected that the range is small.
     *
     * <p>Uses an introselect variant. The quickselect is provided as an argument; the
     * fall-back on poor convergence of the quickselect is a heapselect.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>Recursion is monitored by checking the partition is reduced by 2<sup>-x</sup> after
     * {@code c} iterations where {@code x} is the
     * {@link #setRecursionConstant(int) recursion constant} and {@code c} is the
     * {@link #setRecursionMultiple(double) recursion multiple} (variables reused for convenience).
     * Confidence bounds for dividing a length by 2<sup>-x</sup> are provided in Valois (2000)
     * as {@code c = floor((6/5)x) + b}:
     * <pre>
     * b  confidence (%)
     * 2  76.56
     * 3  92.92
     * 4  97.83
     * 5  99.33
     * 6  99.79
     * </pre>
     * <p>Ideally {@code c >= 3} using {@code x = 1}. E.g. We can use 3 iterations to be 76%
     * confident the sequence will divide in half; or 7 iterations to be 99% confident the
     * sequence will divide into a quarter. A larger factor {@code b} reduces the sensitivity
     * of introspection.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka First key of interest.
     * @param kb Last key of interest.
     */
    private void introselect2(SPEPartition part, double[] a, int left, int right, int ka, int kb) {
        int l = left;
        int r = right;
        final int[] upper = {0};
        int counter = (int) recursionMultiple;
        int threshold = (right - left) >>> recursionConstant;
        while (true) {
            // It is possible to use edgeselect when k is close to the end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            if (Math.min(kb - l, r - ka) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka, kb);
                return;
            }

            // length - 1
            int n = r - l;
            if (--counter < 0) {
                if (n > threshold) {
                    // Did not reduce the length after set number of iterations.
                    // Here riselect (Valois (2000)) would use random points to choose the pivot
                    // to inject entropy and restart. This continues until the sum of the partition
                    // lengths is too high (twice the original length). Here we just switch.

                    // Note: For testing we trigger the recursion consumer with the remaining length
                    recursionConsumer.accept(r - l);
                    stopperSelection.partition(a, l, r, ka, kb);
                    return;
                }
                // Once the confidence has been achieved we use (6/5)x with x=1.
                // So check every 5/6 iterations that the length is halving.
                if (counter == -5) {
                    counter = 1;
                }
                threshold >>>= 1;
            }

            // Pick a pivot and partition
            int pivot;
            if (n > subSamplingSize) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                ++n;
                final int ith = ka - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (ka - ith * s / n + sd));
                final int rr = Math.min(r, (int) (ka + (n - ith) * s / n + sd));
                // Optional random sampling
                if ((controlFlags & FLAG_RANDOM_SAMPLING) != 0) {
                    final IntUnaryOperator rng = createRNG(n, ka);
                    // Shuffle [ll, k) from [l, k)
                    if (ll > l) {
                        for (int i = ka; i > ll;) {
                            // l + rand [0, i - l + 1) : i is currently i+1
                            final int j = l + rng.applyAsInt(i - l);
                            final double t = a[--i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (k, r]
                    if (rr < r) {
                        for (int i = ka; i < rr;) {
                            // r - rand [0, r - i + 1) : i is currently i-1
                            final int j = r - rng.applyAsInt(r - i);
                            final double t = a[++i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                }
                // Sample recursion restarts from [ll, rr]
                introselect2(part, a, ll, rr, ka, ka);
                pivot = ka;
            } else {
                // default pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, ka);
            }

            final int p0 = part.partition(a, l, r, pivot, upper);
            final int p1 = upper[0];

            // Note: Here we expect [ka, kb] to be small and splitting is unlikely.
            //                   p0 p1
            // |l|--|ka|kkkk|kb|--|P|-------------------|r|
            // |l|----------------|P|--|ka|kkk|kb|------|r|
            // |l|-----------|ka|k|P|k|kb|--------------|r|
            if (kb < p0) {
                // The element is in the left partition
                r = p0 - 1;
            } else if (ka > p1) {
                // The element is in the right partition
                l = p1 + 1;
            } else {
                // Pivot splits [ka, kb]. Expect ends to be close to the pivot and finish.
                if (ka < p0) {
                    sortSelectRight(a, l, p0 - 1, ka);
                }
                if (kb > p1) {
                    sortSelectLeft(a, p1 + 1, r, kb);
                }
                return;
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
            int ka = k[ia1];
            final int kb = k[ib1];

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                return;
            }

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (Math.min(kb - l, r - ka) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka, kb);
                return;
            }

            if (maxDepth == 0) {
                // Too much recursion
                heapSelectRange(a, l, r, ka, kb);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r, ka),
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
            int n = r - l;

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // It is possible to use heapselect when kaa and kb1 are close to the same end
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (Math.min(kb1 - l, r - ka1) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (maxDepth == 0) {
                // Too much recursion
                heapSelectRange(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick a pivot and partition
            int pivot;
            if (n > subSamplingSize) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                // Note: This targets ka1 and ignores kb1 for pivot selection.
                ++n;
                final int ith = ka1 - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (ka1 - ith * s / n + sd));
                final int rr = Math.min(r, (int) (ka1 + (n - ith) * s / n + sd));
                // Optional random sampling
                if ((controlFlags & FLAG_RANDOM_SAMPLING) != 0) {
                    final IntUnaryOperator rng = createRNG(n, ka1);
                    // Shuffle [ll, k) from [l, k)
                    if (ll > l) {
                        for (int i = ka1; i > ll;) {
                            // l + rand [0, i - l + 1) : i is currently i+1
                            final int j = l + rng.applyAsInt(i - l);
                            final double t = a[--i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (k, r]
                    if (rr < r) {
                        for (int i = ka1; i < rr;) {
                            // r - rand [0, r - i + 1) : i is currently i-1
                            final int j = r - rng.applyAsInt(r - i);
                            final double t = a[++i];
                            a[i] = a[j];
                            a[j] = t;
                        }
                    }
                }
                introselect(part, a, ll, rr, k, ka1, ka1, lnNtoMaxDepthSinglePivot(z));
                pivot = ka1;
            } else {
                // default pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, ka1);
            }

            final int p0 = part.partition(a, l, r, pivot, upper);
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

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (Math.min(kb - l, r - ka) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (maxDepth == 0) {
                // Too much recursion
                heapSelectRange(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // Pick a pivot and partition
            final int p0 = part.partition(a, l, r,
                pivotingStrategy.pivotIndex(a, l, r, ka),
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

            // If interval is close to one end then edgeselect.
            // Only elect left if there are no further indices in the range.
            // |l|-----|lo|--------|hi|------|right|
            //  ---------d1----------
            //          --------------d2-----------
            if (Math.min(hi - l, right - lo) < edgeSelectConstant) {
                if (hi - l > right - lo) {
                    // Right end. Do not check above hi, just select to the end
                    edgeSelection.partition(a, l, right, lo, right);
                    recursionConsumer.accept(maxDepth);
                    return;
                } else if (k.nextAfter(right)) {
                    // Left end
                    // Only if no further indices in the range.
                    // If false this branch will continue to be triggered until
                    // a partition is made to separate the next indices.
                    edgeSelection.partition(a, l, right, lo, hi);
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
                pivotingStrategy.pivotIndex(a, l, right, ka),
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
            // It is possible to use edgeselect when k is close to the end
            // |l|-----|k|---------|k|--------|r|
            //  ---d1----
            //                      -----d3----
            final int d1 = k - l;
            final int d3 = r - k;
            if (Math.min(d1, d3) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
            }

            if (maxDepth == 0) {
                // Too much recursion
                stopperSelection.partition(a, l, r, k, k);
                // Last known unsorted value >= k
                return r;
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

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                return;
            }

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
                Math.min(s1 + s3, Math.min(s2, s4)) < edgeSelectConstant) {
                // Too much recursion, or ka1 and kb1 are both close to the ends
                // Note: Does not use the edgeSelection function as the indices are not a range
                heapSelectPair(a, l, r, ka1, kb1);
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

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // It is possible to use heapselect when ka1 and kb1 are close to the same end
            // |l|-----|ka1|--------|kb1|------|r|
            //  ---------s2-----------
            //          ----------s4-----------
            if (Math.min(kb1 - l, r - ka1) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka1, kb1);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (maxDepth == 0) {
                // Too much recursion
                heapSelectRange(a, l, r, ka1, kb1);
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

            if (n < minQuickSelectSize) {
                // Sort selection on small data
                sortSelectRange(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            // It is possible to use heapselect when ka and kb are close to the same end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2-----------
            //          ----------s4-----------
            if (Math.min(kb - l, r - ka) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, ka, kb);
                recursionConsumer.accept(maxDepth);
                return;
            }

            if (maxDepth == 0) {
                // Too much recursion
                heapSelectRange(a, l, r, ka, kb);
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
            if (Math.min(hi - l, right - lo) < edgeSelectConstant) {
                if (hi - l > right - lo) {
                    // Right end. Do not check above hi, just select to the end
                    edgeSelection.partition(a, l, right, lo, right);
                    recursionConsumer.accept(maxDepth);
                    return;
                } else if (k.nextAfter(right)) {
                    // Left end
                    // Only if no further indices in the range.
                    // If false this branch will continue to be triggered until
                    // a partition is made to separate the next indices.
                    edgeSelection.partition(a, l, right, lo, hi);
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
                    // Must not use sortSelectRange in [lo, hi] as the iterator
                    // has not been advanced to check after hi
                    sortSelectRight(a, l, right, lo);
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
     * <p>Uses an introselect variant. Uses the configured single-pivot quicksort method;
     * the fall-back on poor convergence of the quickselect is controlled by
     * current configuration.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISP(double[] data, int[] k, int n) {
        introselect(getSPFunction(), data, k, n);
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
     * <p>Uses an introselect variant. Uses the configured single-pivot quicksort method;
     * the fall-back on poor convergence of the quickselect is controlled by
     * current configuration.
     *
     * @param data Values.
     * @param length Length of data.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionISP(double[] data, int length, int[] k, int n) {
        introselect(getSPFunction(), data, length - 1, k, n);
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
     * partition method by Vladimir Yaroslavskiy; the fall-back on poor convergence of
     * the quickselect is controlled by current configuration.
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
     * partition method by Vladimir Yaroslavskiy; the fall-back on poor convergence of
     * the quickselect is controlled by current configuration.
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses the <a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">
     * Floyd-Rivest Algorithm (Wikipedia)</a>
     *
     * <p>WARNING: Currently this only supports a single {@code k}. For parity with other
     * select methods this accepts an array {@code k} and pre/post processes the data for
     * NaN and signed zeros.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    void partitionFR(double[] a, int[] k, int count) {
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
            // Only handles a single k
            if (n != 0) {
                selectFR(a, 0, end - 1, k[0], controlFlags);
            }
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
    }

    /**
     * Select the k-th element of the array.
     *
     * <p>Uses the <a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">
     * Floyd-Rivest Algorithm (Wikipedia)</a>.
     *
     * <p>This code has been adapted from:
     * <pre>
     * Floyd and Rivest (1975)
     * Algorithm 489: The Algorithm SELECT—for Finding the ith Smallest of n elements.
     * Comm. ACM. 18 (3): 173.
     * </pre>
     *
     * @param a Values.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Key of interest.
     * @param flags Control behaviour.
     */
    private void selectFR(double[] a, int left, int right, int k, int flags) {
        int l = left;
        int r = right;
        while (true) {
            // The following edgeselect modifications are additions to the
            // FR algorithm. These have been added for testing and only affect the finishing
            // selection of small lengths.

            // It is possible to use edgeselect when k is close to the end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (Math.min(k - l, r - k) < edgeSelectConstant) {
                edgeSelection.partition(a, l, r, k, k);
                return;
            }

            // use SELECT recursively on a sample of size S to get an estimate for the
            // (k-l+1)-th smallest element into a[k], biased slightly so that the (k-l+1)-th
            // element is expected to lie in the smaller set after partitioning.
            int pivot = k;
            int p = l;
            int q = r;
            // length - 1
            int n = r - l;
            if (n > 600) {
                ++n;
                final int ith = k - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(ith - (n >> 1));
                final int ll = Math.max(l, (int) (k - ith * s / n + sd));
                final int rr = Math.min(r, (int) (k + (n - ith) * s / n + sd));
                // Optional: sample [l, r] into [ll, rr]
                if ((flags & FLAG_SUBSET_SAMPLING) != 0) {
                    // Create a random sample at the left end.
                    // This creates an unbiased random sample.
                    // This method is not as fast as sampling into [ll, rr] (see below).
                    final IntUnaryOperator rng = createRNG(n, k);
                    final int rs = l + rr - ll;
                    for (int i = l - 1; i < rs;) {
                        // r - rand [0, r - i + 1) : i is currently i-1
                        final int j = r - rng.applyAsInt(r - i);
                        final double t = a[++i];
                        a[i] = a[j];
                        a[j] = t;
                    }
                    selectFR(a, l, rs, k - ll + l, flags);
                    // Current:
                    // |l      |k-ll+l|     rs|                               r|
                    // |  < v  |   v  |  > v  |              ???               |
                    // Move partitioned data
                    // |l       |p                     |k|            q|      r|
                    // |  < v   |         ???          |v|      ???    |  > v  |
                    p = k - ll + l;
                    q = r - rs + p;
                    vectorSwap(a, p + 1, rs, r);
                    vectorSwap(a, p, p, k);
                } else {
                    // Note: Random sampling is a redundant overhead on fully random data
                    // and will part destroy sorted data. On data that is: partially partitioned;
                    // has many repeat elements; or is structured with repeat patterns, the
                    // shuffle removes side-effects of patterns and stabilises performance.
                    if ((flags & FLAG_RANDOM_SAMPLING) != 0) {
                        // This is not a random sample from [l, r] when k is not exactly
                        // in the middle. By sampling either side of k the sample
                        // will maintain the value of k if the data is already partitioned
                        // around k. However sorted data will be part scrambled by the shuffle.
                        // A second FR sample on the next smaller partition will have shuffled
                        // data at one end. The majority of the sorted data is unchanged.
                        // This sampling has the best performance overall across datasets.
                        final IntUnaryOperator rng = createRNG(n, k);
                        // Shuffle [ll, k) from [l, k)
                        if (ll > l) {
                            for (int i = k; i > ll;) {
                                // l + rand [0, i - l + 1) : i is currently i+1
                                final int j = l + rng.applyAsInt(i - l);
                                final double t = a[--i];
                                a[i] = a[j];
                                a[j] = t;
                            }
                        }
                        // Shuffle (k, rr] from (k, r]
                        if (rr < r) {
                            for (int i = k; i < rr;) {
                                // r - rand [0, r - i + 1) : i is currently i-1
                                final int j = r - rng.applyAsInt(r - i);
                                final double t = a[++i];
                                a[i] = a[j];
                                a[j] = t;
                            }
                        }
                    }
                    selectFR(a, ll, rr, k, flags);
                    // Current:
                    // |l                    |ll      |k|     rr|            r|
                    // |        ???          |  < v   |v|  > v  |      ???    |
                    // Optional: move partitioned data
                    // Unlikely to make a difference as the partitioning will skip
                    // over <v and >v.
                    // |l       |p                    |k|            q|      r|
                    // |  < v   |        ???          |v|      ???    |  > v  |
                    if ((flags & FLAG_MOVE_SAMPLE) != 0) {
                        vectorSwap(a, l, ll - 1, k - 1);
                        vectorSwap(a, k + 1, rr, r);
                        p += k - ll;
                        q -= rr - k;
                    }
                }
            } else {
                // Optional: use pivot strategy
                pivot = pivotingStrategy.pivotIndex(a, l, r, k);
            }

            // Partition a[p : q] about t.
            // Sub-script range checking has been eliminated by appropriate placement of t
            // at the p or q end.
            final double t = a[pivot];
            // swap(left, pivot)
            a[pivot] = a[p];
            if (a[q] > t) {
                // swap(right, left)
                a[p] = a[q];
                a[q] = t;
                // Here after the first swap: a[p] = t; a[q] > t
            } else {
                a[p] = t;
                // Here after the first swap: a[p] <= t; a[q] = t
            }
            int i = p;
            int j = q;
            while (i < j) {
                // swap(i, j)
                final double temp = a[i];
                a[i] = a[j];
                a[j] = temp;
                do {
                    ++i;
                } while (a[i] < t);
                do {
                    --j;
                } while (a[j] > t);
            }
            if (a[p] == t) {
                // data[j] <= t : swap(left, j)
                a[p] = a[j];
                a[j] = t;
            } else {
                // data[j+1] > t : swap(j+1, right)
                a[q] = a[++j];
                a[j] = t;
            }
            // Continue on the correct side
            if (k < j) {
                r = j - 1;
            } else if (k > j) {
                l = j + 1;
            } else {
                return;
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
     * <p>The method assumes all {@code k} are valid indices into the data.
     * It handles NaN and signed zeros in the data.
     *
     * <p>Uses the <a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">
     * Floyd-Rivest Algorithm (Wikipedia)</a>, modified by Kiwiel.
     *
     * <p>WARNING: Currently this only supports a single {@code k}. For parity with other
     * select methods this accepts an array {@code k} and pre/post processes the data for
     * NaN and signed zeros.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    void partitionKFR(double[] a, int[] k, int count) {
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
            // Only handles a single k
            if (n != 0) {
                final int[] bounds = new int[5];
                selectKFR(a, 0, end - 1, k[0], bounds, null);
            }
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
    }

    /**
     * Select the k-th element of the array.
     *
     * <p>Uses the <a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">
     * Floyd-Rivest Algorithm (Wikipedia)</a>, modified by Kiwiel.
     *
     * <p>References:
     * <ul>
     * <li>Floyd and Rivest (1975)
     * Algorithm 489: The Algorithm SELECT—for Finding the ith Smallest of n elements.
     * Comm. ACM. 18 (3): 173.
     * <li>Kiwiel (2005)
     * On Floyd and Rivest's SELECT algorithm.
     * Theoretical Computer Science 347, 214-238.
     * </ul>
     *
     * @param x Values.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param k Key of interest.
     * @param bounds Inclusive bounds {@code [k-, k+]} containing {@code k}.
     * @param rng Random generator for samples in {@code [0, n)}.
     */
    private void selectKFR(double[] x, int left, int right, int k, int[] bounds,
        IntUnaryOperator rng) {
        int l = left;
        int r = right;
        while (true) {
            // The following edgeselect modifications are additions to the
            // KFR algorithm. These have been added for testing and only affect the finishing
            // selection of small lengths.

            // It is possible to use edgeselect when k is close to the end
            // |l|-----|ka|--------|kb|------|r|
            //  ---------s2----------
            //          ----------s4-----------
            if (Math.min(k - l, r - k) < edgeSelectConstant) {
                edgeSelection.partition(x, l, r, k, k);
                bounds[0] = bounds[1] = k;
                return;
            }

            // length - 1
            int n = r - l;
            if (n < 600) {
                // Switch to quickselect
                final int p0 = partitionKBM(x, l, r,
                    pivotingStrategy.pivotIndex(x, l, r, k), bounds);
                final int p1 = bounds[0];
                if (k < p0) {
                    // The element is in the left partition
                    r = p0 - 1;
                } else if (k > p1) {
                    // The element is in the right partition
                    l = p1 + 1;
                } else {
                    // The range contains the element we wanted.
                    bounds[0] = p0;
                    bounds[1] = p1;
                    return;
                }
                continue;
            }

            // Floyd-Rivest sub-sampling
            ++n;
            // Step 1: Choose sample size s <= n-1 and gap g > 0
            final double z = Math.log(n);
            // sample size = alpha * n^(2/3) * ln(n)^1/3  (4.1)
            // sample size = alpha * n^(2/3)              (4.17; original Floyd-Rivest size)
            final double s = 0.5 * Math.exp(0.6666666666666666 * z) * Math.cbrt(z);
            //final double s = 0.5 * Math.exp(0.6666666666666666 * z);
            // gap = sqrt(beta * s * ln(n))
            final double g = Math.sqrt(0.25 * s * z);
            final int rs = (int) (l + s - 1);
            // Step 2: Sample selection
            // Convenient to place the random sample in [l, rs]
            if (rng == null) {
                rng = createRNG(n, k);
            }
            for (int i = l - 1; i < rs;) {
                // r - rand [0, r - i + 1) : i is currently i-1
                final int j = r - rng.applyAsInt(r - i);
                final double t = x[++i];
                x[i] = x[j];
                x[j] = t;
            }

            // Step 3: pivot selection
            final double isn = (k - l + 1) * s / n;
            final int ku = (int) Math.max(Math.floor(l - 1 + isn - g), l);
            final int kv = (int) Math.min(Math.ceil(l - 1 + isn + g), rs);
            // Find u and v by recursion
            selectKFR(x, l, rs, ku, bounds, rng);
            final int kum = bounds[0];
            int kup = bounds[1];
            int kvm;
            int kvp;
            if (kup >= kv) {
                kvm = kv;
                kvp = kup;
                kup = kv - 1;
                // u == v will use single-pivot ternary partitioning
            } else {
                selectKFR(x, kup + 1, rs, kv, bounds, rng);
                kvm = bounds[0];
                kvp = bounds[1];
            }

            // Step 4: Partitioning
            final double u = x[kup];
            final double v = x[kvm];
            // |l      |ku- ku+|                   |kv- kv+|     rs|            r|     (6.4)
            // | x < u | x = u |     u < x < v     | x = v | x > v |      ???    |
            final int ll = kum;
            int pp = kup;
            final int rr = r - rs + kvp;
            int qq = rr - kvp + kvm;
            vectorSwap(x, kvp + 1, rs, r);
            vectorSwap(x, kvm, kvp, rr);
            // |l      |ll   pp|                   |kv-          |qq   rr|      r|     (6.5)
            // | x < u | x = u |     u < x < v     |      ???    | x = v | x > v |

            int a;
            int b;
            int c;
            int d;

            if (u == v) {
                // Can be optimised by omitting step A1 (moving of sentinels). Here the
                // size of ??? is large and initialisation is insignificant.
                a = partitionKBM(x, ll, rr, pp, bounds);
                d = bounds[0];
                // Make ternary and quintary partitioning compatible
                b = d + 1;
                c = a - 1;
//            } else {
//                // Yaroslavskiy dual-pivot partitioning
//                a = partitionDP(x, l, r, pp, qq, bounds);
//                d = bounds[2];
//                if (bounds[1] - bounds[0] > 1) {
//                    b = bounds[0] + 1;
//                    c = bounds[1] - 1;
//                } else {
//                    // No central region
//                    b = d + 1;
//                    c = a - 1;
//                }
//            }
            } else if (k < (r + l) >>> 1) {
                // Left k: u < x[k] < v --> expects x > v.
                // Quintary partitioning using the six-part array:
                // |ll   pp|              p|          |i        j|       |q    rr|     (6.6)
                // | x = u |    u < x < v  |   x < u  |   ???    | x > v | x = v |
                //
                // |ll   pp|              p|              j|i            |q    rr|     (6.7)
                // | x = u |    u < x < v  |   x < u       |       x > v | x = v |
                //
                // Swap the second and third part:
                // |ll   pp|               |b             c|i            |q    rr|     (6.8)
                // | x = u |   x < u       |    u < x < v  |       x > v | x = v |
                //
                // Swap the extreme parts with their neighbours:
                // |ll             |a      |b             c|      d|           rr|     (6.9)
                // |   x < u       | x = u |    u < x < v  | x = v |       x > v |
                quintaryPartitionL(x, pp, kvm - 1, qq, u, v, bounds);
                pp = bounds[0];
                final int p = bounds[1];
                final int q = bounds[2];
                final int i = bounds[3];
                final int j = bounds[4];
//                int p = kvm - 1;
//                int q = qq;
//                int i = p;
//                int j = q;
//                for (;;) {
//                    while (x[++i] < v) {
//                        if (x[i] < u) {
//                            continue;
//                        }
//                        // u <= xi < v
//                        final double xi = x[i];
//                        x[i] = x[++p];
//                        if (xi > u) {
//                            x[p] = xi;
//                        } else {
//                            x[p] = x[++pp];
//                            x[pp] = xi;
//                        }
//                    }
//                    while (x[--j] >= v) {
//                        if (x[j] == v) {
//                            final double xj = x[j];
//                            x[j] = x[--q];
//                            x[q] = xj;
//                        }
//                    }
//                    // Here x[j] < v <= x[i]
//                    if (i >= j) {
//                        break;
//                    }
//                    //swap(x, i, j)
//                    final double xi = x[j];
//                    final double xj = x[i];
//                    x[i] = xi;
//                    x[j] = xj;
//                    if (xi > u) {
//                        x[i] = x[++p];
//                        x[p] = xi;
//                    } else if (xi == u) {
//                        x[i] = x[++p];
//                        x[p] = x[++pp];
//                        x[pp] = xi;
//                    }
//                    if (xj == v) {
//                        x[j] = x[--q];
//                        x[q] = xj;
//                    }
//                }
                a = ll + i - p - 1;
                b = a + pp + 1 - ll;
                d = rr - q + 1 + j;
                c = d - rr + q - 1;
                vectorSwap(x, pp + 1, p, j);
                //vectorSwap(x, ll, pp, b - 1);
                //vectorSwap(x, i, q - 1, rr);
                vectorSwapL(x, ll, pp, b - 1, u);
                vectorSwapR(x, i, q - 1, rr, v);
            } else {
                // Right k: u < x[k] < v --> expects x < u.
                // Symmetric quintary partitioning replacing 6.6-6.8 with:
                // |ll    p|          |i        j|       |q              |qq   rr|     (6.10)
                // | x = u |   x < u  |   ???    | x > v |    u < x < v  | x = v |
                //
                // |ll    p|                j|i      |q                  |qq   rr|     (6.11)
                // | x = u |   x < u         | x > v |        u < x < v  | x = v |
                //
                // |ll    p|                j|b                 c|       |qq   rr|     (6.12)
                // | x = u |   x < u         |        u < x < v  | x > v | x = v |
                //
                // |ll               |a      |b                 c|      d|     rr|     (6.9)
                // |   x < u         | x = u |        u < x < v  | x = v | x > v |
                vectorSwap(x, pp + 1, kvm - 1, qq - 1);
                quintaryPartitionR(x, pp, qq - kvm + kup + 1, qq, u, v, bounds);
                final int p = bounds[0];
                final int q = bounds[1];
                qq = bounds[2];
                final int i = bounds[3];
                final int j = bounds[4];
//                int p = pp;
//                int q = qq - kvm + kup + 1;
//                int i = p;
//                int j = q;
//                vectorSwap(x, pp + 1, kvm - 1, qq - 1);
//                for (;;) {
//                    while (x[++i] <= u) {
//                        if (x[i] == u) {
//                            final double xi = x[i];
//                            x[i] = x[++p];
//                            x[p] = xi;
//                        }
//                    }
//                    while (x[--j] > u) {
//                        if (x[j] > v) {
//                            continue;
//                        }
//                        // u < xj <= v
//                        final double xj = x[j];
//                        x[j] = x[--q];
//                        if (xj < v) {
//                            x[q] = xj;
//                        } else {
//                            x[q] = x[--qq];
//                            x[qq] = xj;
//                        }
//                    }
//                    // Here x[j] < v <= x[i]
//                    if (i >= j) {
//                        break;
//                    }
//                    //swap(x, i, j)
//                    final double xi = x[j];
//                    final double xj = x[i];
//                    x[i] = xi;
//                    x[j] = xj;
//                    if (xi == u) {
//                        x[i] = x[++p];
//                        x[p] = xi;
//                    }
//                    if (xj < v) {
//                        x[j] = x[--q];
//                        x[q] = xj;
//                    } else if (xj == v) {
//                        x[j] = x[--q];
//                        x[q] = x[--qq];
//                        x[qq] = xj;
//                    }
//                }
                a = ll + i - p - 1;
                b = a + p + 1 - ll;
                d = rr - q + 1 + j;
                c = d - rr + qq - 1;
                vectorSwap(x, i, q - 1, qq - 1);
                //vectorSwap(x, ll, p, j);
                //vectorSwap(x, c + 1, qq - 1, rr);
                vectorSwapL(x, ll, p, j, u);
                vectorSwapR(x, c + 1, qq - 1, rr, v);
            }

            // Step 5/6/7: Stopping test, reduction and recursion
            // |l              |a      |b             c|      d|            r|
            // |   x < u       | x = u |    u < x < v  | x = v |       x > v |
            if (a <= k) {
                l = b;
            }
            if (c < k) {
                l = d + 1;
            }
            if (k <= d) {
                r = c;
            }
            if (k < b) {
                r = a - 1;
            }
            if (l >= r) {
                if (l == r) {
                    // [b, c]
                    bounds[0] = bounds[1] = k;
                } else {
                    // l > r
                    bounds[0] = r + 1;
                    bounds[1] = l - 1;
                }
                return;
            }
        }
    }

    /**
     * Vector swap x[a:b] <-> x[b+1:c] means the first m = min(b+1-a, c-b)
     * elements of the array x[a:c] are exchanged with its last m elements.
     *
     * @param x Array.
     * @param a Index.
     * @param b Index.
     * @param c Index.
     */
    private static void vectorSwap(double[] x, int a, int b, int c) {
        for (int i = a - 1, j = c + 1, m = Math.min(b + 1 - a, c - b); --m >= 0;) {
            final double v = x[++i];
            x[i] = x[--j];
            x[j] = v;
        }
    }

    /**
     * Vector swap x[a:b] <-> x[b+1:c] means the first m = min(b+1-a, c-b)
     * elements of the array x[a:c] are exchanged with its last m elements.
     *
     * <p>This is a specialisation of {@link #vectorSwap(double[], int, int, int)}
     * where the current left-most value is a constant {@code v}.
     *
     * @param x Array.
     * @param a Index.
     * @param b Index.
     * @param c Index.
     * @param v Constant value in [a, b]
     */
    private static void vectorSwapL(double[] x, int a, int b, int c, double v) {
        for (int i = a - 1, j = c + 1, m = Math.min(b + 1 - a, c - b); --m >= 0;) {
            x[++i] = x[--j];
            x[j] = v;
        }
    }

    /**
     * Vector swap x[a:b] <-> x[b+1:c] means the first m = min(b+1-a, c-b)
     * elements of the array x[a:c] are exchanged with its last m elements.
     *
     * <p>This is a specialisation of {@link #vectorSwap(double[], int, int, int)}
     * where the current right-most value is a constant {@code v}.
     *
     * @param x Array.
     * @param a Index.
     * @param b Index.
     * @param c Index.
     * @param v Constant value in (b, c]
     */
    private static void vectorSwapR(double[] x, int a, int b, int c, double v) {
        for (int i = a - 1, j = c + 1, m = Math.min(b + 1 - a, c - b); --m >= 0;) {
            x[--j] = x[++i];
            x[i] = v;
        }
    }

    /**
     * Quintary partition of the six-part array:
     * <pre>
     * |ll   pp|              p|          |i        j|       |q    rr|     (6.6)
     * | x = u |    u < x < v  |   x < u  |   ???    | x > v | x = v |
     *
     * |ll   pp|              p|              j|i            |q    rr|     (6.7)
     * | x = u |    u < x < v  |   x < u       |       x > v | x = v |
     * </pre>
     *
     * @param x Data.
     * @param pp Index.
     * @param p Index.
     * @param q Index.
     * @param u Pivot value.
     * @param v Pivot value.
     * @param bounds Output [pp, p, q, i, j]
     */
    private static void quintaryPartitionL(double[] x, int pp, int p, int q, double u, double v,
        int[] bounds) {
        int i = p;
        int j = q;
        for (;;) {
            while (x[++i] < v) {
                if (x[i] < u) {
                    continue;
                }
                // u <= xi < v
                final double xi = x[i];
                x[i] = x[++p];
                if (xi > u) {
                    x[p] = xi;
                } else {
                    x[p] = x[++pp];
                    x[pp] = xi;
                }
            }
            while (x[--j] >= v) {
                if (x[j] == v) {
                    final double xj = x[j];
                    x[j] = x[--q];
                    x[q] = xj;
                }
            }
            // Here x[j] < v <= x[i]
            if (i >= j) {
                break;
            }
            //swap(x, i, j)
            final double xi = x[j];
            final double xj = x[i];
            x[i] = xi;
            x[j] = xj;
            if (xi > u) {
                x[i] = x[++p];
                x[p] = xi;
            } else if (xi == u) {
                x[i] = x[++p];
                x[p] = x[++pp];
                x[pp] = xi;
            }
            if (xj == v) {
                x[j] = x[--q];
                x[q] = xj;
            }
        }
        bounds[0] = pp;
        bounds[1] = p;
        bounds[2] = q;
        bounds[3] = i;
        bounds[4] = j;
    }
    /**
     * Quintary partition of the six-part array:
     * <pre>
     * |ll    p|          |i        j|       |q              |qq   rr|     (6.10)
     * | x = u |   x < u  |   ???    | x > v |    u < x < v  | x = v |
     *
     * |ll    p|                j|i      |q                  |qq   rr|     (6.11)
     * | x = u |   x < u         | x > v |        u < x < v  | x = v |
     * </pre>
     *
     * @param x Data.
     * @param p Index.
     * @param q Index.
     * @param qq Index.
     * @param u Pivot value.
     * @param v Pivot value.
     * @param bounds Output [p, q, qq, i, j]
     */
    private static void quintaryPartitionR(double[] x, int p, int q, int qq, double u, double v,
        int[] bounds) {
        int i = p;
        int j = q;
        for (;;) {
            while (x[++i] <= u) {
                if (x[i] == u) {
                    final double xi = x[i];
                    x[i] = x[++p];
                    x[p] = xi;
                }
            }
            while (x[--j] > u) {
                if (x[j] > v) {
                    continue;
                }
                // u < xj <= v
                final double xj = x[j];
                x[j] = x[--q];
                if (xj < v) {
                    x[q] = xj;
                } else {
                    x[q] = x[--qq];
                    x[qq] = xj;
                }
            }
            // Here x[j] < v <= x[i]
            if (i >= j) {
                break;
            }
            //swap(x, i, j)
            final double xi = x[j];
            final double xj = x[i];
            x[i] = xi;
            x[j] = xj;
            if (xi == u) {
                x[i] = x[++p];
                x[p] = xi;
            }
            if (xj < v) {
                x[j] = x[--q];
                x[q] = xj;
            } else if (xj == v) {
                x[j] = x[--q];
                x[q] = x[--qq];
                x[qq] = xj;
            }
        }
        bounds[0] = p;
        bounds[1] = q;
        bounds[2] = qq;
        bounds[3] = i;
        bounds[4] = j;
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
     * <p>Uses the configured single-pivot quicksort method;
     * and median of medians algorithm for pivot selection with medians-of-5.
     *
     * <p>Note:
     * <p>This method is not configurable with the exception of the single-pivot quickselect method
     * and the size to stop quickselect recursion and finish using sort select. It has been superceded by
     * {@link #partitionLinear(double[], int[], int)} which has configurable deterministic
     * pivot selection including those using partition expansion in-place of full partitioning.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionLSP(double[] data, int[] k, int n) {
        linearSelect(getSPFunction(), data, k, n);
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
     * <p>Uses the median of medians algorithm for pivot selection.
     *
     * <p>WARNING: Currently this only supports a single or range of {@code k}.
     * For parity with other select methods this accepts an array {@code k} and pre/post
     * processes the data for NaN and signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    private void linearSelect(SPEPartition part, double[] a, int[] k, int count) {
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
            if (n != 0) {
                final int ka = Math.min(k[0], k[n - 1]);
                final int kb = Math.max(k[0], k[n - 1]);
                linearSelect(part, a, 0, end - 1, ka, kb, new int[2]);
            }
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [ka, kb]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
     * }</pre>
     *
     * <p>This function accepts indices {@code [ka, kb]} that define the
     * range of indices to partition. It is expected that the range is small.
     *
     * <p>Uses quickselect with median-of-medians pivot selection to provide Order(n)
     * performance.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>Returns the bounds containing {@code [ka, kb]}. These may be lower/higher
     * than the keys if equal values are present in the data. This is to be used by
     * {@link #pivotMedianOfMedians(SPEPartition, double[], int, int, int[])} to identify
     * the equal value range of the pivot.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka First key of interest.
     * @param kb Last key of interest.
     * @param bounds Bounds of the range containing {@code [ka, kb]} (inclusive).
     * @see <a href="https://en.wikipedia.org/wiki/Median_of_medians">Median of medians (Wikipedia)</a>
     */
    private void linearSelect(SPEPartition part, double[] a, int left, int right, int ka, int kb,
            int[] bounds) {
        int l = left;
        int r = right;
        while (true) {
            // Select when ka and kb are close to the same end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            // Optimal value for this is much higher than standard quickselect due
            // to the high cost of median-of-medians pivot computation and reuse via
            // mutual recursion so we have a different value.
            if (Math.min(kb - l, r - ka) < linearSortSelectSize) {
                sortSelectRange(a, l, r, ka, kb);
                // We could scan left/right to extend the bounds here after the sort.
                // Since the move_sample strategy is not generally useful we do not bother.
                bounds[0] = ka;
                bounds[1] = kb;
                return;
            }
            int p0 = pivotMedianOfMedians(part, a, l, r, bounds);
            if ((controlFlags & FLAG_MOVE_SAMPLE) != 0) {
                // Note: medians with 5 elements creates a sample size of 20%.
                // Avoid partitioning the sample known to be above the pivot.
                // The pivot identified the lower pivot (lp) and upper pivot (p).
                // This strategy is not faster unless there are a large number of duplicates
                // (e.g. less than 10 unique values).
                // On random data with no duplicates this is slower.
                // Note: The methods based on quickselect adaptive create the sample in
                // a region corresponding to expected k and expand the partition (much faster).
                //
                // |l  |lp p0| rr|                              r|
                // | < |  == | > |        ???                    |
                //
                // Move region above P to r
                //
                // |l  |pp p0|                                  r|
                // | < |  == |           ???                 | > |
                final int lp = bounds[0];
                final int rr = bounds[1];
                vectorSwap(a, p0 + 1, rr, r);
                // 20% less to partition
                final int p = part.partition(a, p0, r - rr + p0, p0, bounds);
                // |l    |pp  |p0         |p  u|                r|
                // |  <  | == |    <      | == |        >        |
                //
                // Move additional equal pivot region to the centre:
                // |l                |p0      u|                r|
                // |        <        |   ==    |        >        |
                vectorSwapL(a, lp, p0 - 1, p - 1, a[p]);
                p0 = p - p0 + lp;
            } else {
                p0 = part.partition(a, l, r, p0, bounds);
            }
            final int p1 = bounds[0];

            // Note: Here we expect [ka, kb] to be small and splitting is unlikely.
            //                   p0 p1
            // |l|--|ka|kkkk|kb|--|P|-------------------|r|
            // |l|----------------|P|--|ka|kkk|kb|------|r|
            // |l|-----------|ka|k|P|k|kb|--------------|r|
            if (kb < p0) {
                // Entirely on left side
                r = p0 - 1;
            } else if (ka > p1) {
                // Entirely on right side
                l = p1 + 1;
            } else {
                // Pivot splits [ka, kb]. Expect ends to be close to the pivot and finish.
                // Here we set the bounds for use after median-of-medians pivot selection.
                // In the event there are many equal values this allows collecting those
                // known to be equal together when moving around the medians sample.
                bounds[0] = p0;
                bounds[1] = p1;
                if (ka < p0) {
                    sortSelectRight(a, l, p0 - 1, ka);
                    bounds[0] = ka;
                }
                if (kb > p1) {
                    sortSelectLeft(a, p1 + 1, r, kb);
                    bounds[1] = kb;
                }
                return;
            }
        }
    }

    /**
     * Compute the median of medians pivot. Divides the length {@code n} into groups
     * of at most 5 elements, computes the median of each group, and the median of the
     * {@code n/5} medians. Assumes {@code l <= r}.
     *
     * <p>The median of medians in computed in-place at the left end. The range containing
     * the medians is {@code [l, rr]} with the right bound {@code rr} returned.
     * In the event the pivot is a region of equal values, the range of the pivot values
     * is {@code [lp, p]}, with the {@code p} returned and {@code lp} set in the output bounds.
     *
     * @param part Partition function.
     * @param a Values.
     * @param l Lower bound of data (inclusive, assumed to be strictly positive).
     * @param r Upper bound of data (inclusive, assumed to be strictly positive).
     * @param bounds Bounds {@code [lp, rr]}.
     * @return the pivot index {@code p}
     */
    private int pivotMedianOfMedians(SPEPartition part, double[] a, int l, int r, int[] bounds) {
        // Process blocks of 5.
        // Moves the median of each block to the left of the array.
        int rr = l - 1;
        for (int e = l + 5;; e += 5) {
            if (e > r) {
                // Final block of size 1-5
                Sorting.sort(a, e - 5, r);
                final int m = (e - 5 + r) >>> 1;
                final double v = a[m];
                a[m] = a[++rr];
                a[rr] = v;
                break;
            }

            // Various methods for time-critical step.
            // Each must be compiled and run on the same benchmark data.
            // Decision tree is fastest.
            //final int m = Sorting.median5(a, e - 5);
            //final int m = Sorting.median5(a, e - 5, e - 4, e - 3, e - 2, e - 1);
            // Bigger decision tree (same as median5)
            //final int m = Sorting.median5b(a, e - 5);
            // Sorting network of 4 + insertion (3-4% slower)
            //final int m = Sorting.median5c(a, e - 5);
            // In-place median: Sorting of 5, or median of 5
            final int m = e - 3;
            //Sorting.sort(a, e - 5, e - 1); // insertion sort
            //Sorting.sort5(a, e - 5, e - 4, e - 3, e - 2, e - 1);
            Sorting.median5d(a, e - 5, e - 4, e - 3, e - 2, e - 1);

            final double v = a[m];
            a[m] = a[++rr];
            a[rr] = v;
        }

        int m = (l + rr + 1) >>> 1;
        // mutual recursion
        linearSelect(part, a, l, rr, m, m, bounds);
        // bounds contains the range of the pivot.
        // return the upper pivot and record the end of the range.
        m = bounds[1];
        bounds[1] = rr;
        return m;
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
     * <p>Uses the median of medians algorithm to provide Order(n) performance.
     * This method has configurable deterministic pivot selection including those using
     * partition expansion in-place of full partitioning. The methods are based on the
     * QuickselectAdaptive method of Alexandrescu.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionLinear(double[] data, int[] k, int n) {
        quickSelect(linearSpFunction, data, k, n);
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
     * <p>This method assumes that the partition function can compute a pivot.
     * It is used for variants of the median of medians algorithm which use mutual
     * recursion for pivot selection.
     *
     * <p>WARNING: Currently this only supports a single or range of {@code k}.
     * For parity with other select methods this accepts an array {@code k} and pre/post
     * processes the data for NaN and signed zeros.
     *
     * @param part Partition function.
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    private void quickSelect(SPEPartition part, double[] a, int[] k, int count) {
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
            if (n != 0) {
                final int ka = Math.min(k[0], k[n - 1]);
                final int kb = Math.max(k[0], k[n - 1]);
                quickSelect(part, a, 0, end - 1, ka, kb, new int[2]);
            }
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [ka, kb]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
     * }</pre>
     *
     * <p>This function accepts indices {@code [ka, kb]} that define the
     * range of indices to partition. It is expected that the range is small.
     *
     * <p>This method assumes that the partition function can compute a pivot.
     * It is used for variants of the median of medians algorithm which use mutual
     * recursion for pivot selection. This method is based on the improvements
     * for median-of-medians algorithm in Alexandrescu (2016).
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>Returns the bounds containing {@code [ka, kb]}. These may be lower/higher
     * than the keys if equal values are present in the data.
     *
     * @param part Partition function.
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka First key of interest.
     * @param kb Last key of interest.
     * @param bounds Bounds of the range containing {@code [ka, kb]} (inclusive).
     */
    private void quickSelect(SPEPartition part, double[] a, int left, int right, int ka, int kb,
            int[] bounds) {
        int l = left;
        int r = right;
        while (true) {
            // Select when ka and kb are close to the same end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            // Optimal value for this is much higher than standard quickselect due
            // to the high cost of median-of-medians pivot computation and reuse via
            // mutual recursion so we have a different value.
            // Note: Use of this will not break the Order(n) performance for worst
            // case data, i.e. data where all values require full insertion.
            // This will be Order(n * k) == Order(n); k becomes a multiplier as long as
            // k << n; otherwise worst case is Order(n^2 / 2) when k=n/2.
            if (Math.min(kb - l, r - ka) < linearSortSelectSize) {
                sortSelectRange(a, l, r, ka, kb);
                // We could scan left/right to extend the bounds here after the sort.
                // TODO - update sortSelectRange to sortSelectRange2 and return the
                // known equal value below/above the target left/right k.
                bounds[0] = ka;
                bounds[1] = kb;
                return;
            }
            // Only target ka; kb is assumed to be close
            final int p0 = part.partition(a, l, r, ka, bounds);
            final int p1 = bounds[0];

            // Note: Here we expect [ka, kb] to be small and splitting is unlikely.
            //                   p0 p1
            // |l|--|ka|kkkk|kb|--|P|-------------------|r|
            // |l|----------------|P|--|ka|kkk|kb|------|r|
            // |l|-----------|ka|k|P|k|kb|--------------|r|
            if (kb < p0) {
                // Entirely on left side
                r = p0 - 1;
            } else if (ka > p1) {
                // Entirely on right side
                l = p1 + 1;
            } else {
                // Pivot splits [ka, kb]. Expect ends to be close to the pivot and finish.
                // Here we set the bounds for use after median-of-medians pivot selection.
                // In the event there are many equal values this allows collecting those
                // known to be equal together when moving around the medians sample.
                bounds[0] = p0;
                bounds[1] = p1;
                if (ka < p0) {
                    sortSelectRight(a, l, p0 - 1, ka);
                    bounds[0] = ka;
                }
                if (kb > p1) {
                    sortSelectLeft(a, p1 + 1, r, kb);
                    bounds[1] = kb;
                }
                return;
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
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>Uses the QuickselectAdaptive method of Alexandrescu. This is based on the
     * median of medians algorithm. The median sample is strategy is chosen based on
     * the target index.
     *
     * @param data Values.
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     */
    void partitionQA(double[] data, int[] k, int n) {
        quickSelectAdaptive(data, k, n);
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
     * <p>WARNING: Currently this only supports a single or range of {@code k}.
     * For parity with other select methods this accepts an array {@code k} and pre/post
     * processes the data for NaN and signed zeros.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @param count Count of indices.
     */
    private void quickSelectAdaptive(double[] a, int[] k, int count) {
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
            if (n != 0) {
                final int ka = Math.min(k[0], k[n - 1]);
                final int kb = Math.max(k[0], k[n - 1]);
                quickSelectAdaptive(a, 0, end - 1, ka, kb, new int[1],
                    (controlFlags & FLAG_QA_NO_SAMPLING) != 0 ? -1 : 0);
            }
        }
        // Restore signed zeros
        t.postProcess(a, k, n);
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [ka, kb]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
     * }</pre>
     *
     * <p>This function accepts indices {@code [ka, kb]} that define the
     * range of indices to partition. It is expected that the range is small.
     *
     * <p>Uses the QuickselectAdaptive method of Alexandrescu. This is based on the
     * median of medians algorithm. The median sample is strategy is chosen based on
     * the target index.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * <p>The control {@code flags} sign bit is set when the full repeated step algorithm
     * should be used. Otherwise the sampling mode is enabled which skips the first median
     * step in all repeated step algorithms. This reduces the deterministic margins
     * around the pivot but increases speed.
     *
     * <p>Returns the bounds containing {@code [ka, kb]}. These may be lower/higher
     * than the keys if equal values are present in the data.
     *
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka First key of interest.
     * @param kb Last key of interest.
     * @param bounds Upper bound of the range containing {@code [ka, kb]} (inclusive).
     * @param flags Control flags.
     * @return Lower bound of the range containing {@code [ka, kb]} (inclusive).
     */
    private int quickSelectAdaptive(double[] a, int left, int right, int ka, int kb,
            int[] bounds, int flags) {
        int l = left;
        int r = right;
        int cf = flags;
        while (true) {
            // Select when ka and kb are close to the same end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            // Optimal value for this is much higher than standard quickselect due
            // to the high cost of median-of-medians pivot computation and reuse via
            // mutual recursion so we have a different value.
            // Note: Use of this will not break the Order(n) performance for worst
            // case data, i.e. data where all values require full insertion.
            // This will be Order(n * k) == Order(n); k becomes a multiplier as long as
            // k << n; otherwise worst case is Order(n^2 / 2) when k=n/2.
            if (Math.min(kb - l, r - ka) < linearSortSelectSize) {
                sortSelectRange(a, l, r, ka, kb);
                bounds[0] = kb;
//                // Extending the equal value range here is slower
//                if (kb - l < r - ka) {
//                    // Check left
//                    int lo = ka;
//                    if (lo > l && a[ka] == a[--lo]) {
//                        while (--lo >= r) {
//                            if (a[ka] != a[lo]) {
//                                break;
//                            }
//                        }
//                        return lo + 1;
//                    }
//                } else {
//                    // Check right
//                    int hi = kb;
//                    if (hi < r && a[kb] == a[++hi]) {
//                        while (++hi <= r) {
//                            if (a[kb] != a[hi]) {
//                                break;
//                            }
//                        }
//                        bounds[0] = hi - 1;
//                    }
//                }
                return ka;
            }

            // Only target ka; kb is assumed to be close
            int p0;
            int n = r - l + 1;
            final double f = (double) (ka - l) / n;
            // Note: Margins for fraction left/right of pivot L : R.
            // Subtract the larger margin to create the estimated size
            // after partitioning. If the new size subtracted from
            // the estimated size is negative (partition did not meet
            // the margin guarantees) then sampling is disabled by setting the
            // control flags sign bit.
            if (f <= STEP_LEFT) {
                if (f <= STEP_FAR_LEFT) {
                    // 1/12 : 3/8
                    n -= (n >> 2) + (n >> 3);
                    p0 = repeatedStepLeft(a, l, r, ka, bounds, cf, true);
                } else {
                    // 1/6 : 1/4
                    n -= n >> 2;
                    p0 = repeatedStepLeft(a, l, r, ka, bounds, cf, false);
                }
            } else if (f >= STEP_RIGHT) {
                if (f >= STEP_FAR_RIGHT) {
                    // 3/8 : 1/12
                    n -= (n >> 2) + (n >> 3);
                    p0 = repeatedStepRight(a, l, r, ka, bounds, cf, true);
                } else {
                    // 1/4 : 1/6
                    n -= n >> 2;
                    p0 = repeatedStepRight(a, l, r, ka, bounds, cf, false);
                }
            } else {
                // 2/9 : 2/9 (use 1/4 - 1/32 ~ 0.219)
                n -= (n >> 2) - (n >> 5);
                p0 = repeatedStep(a, l, r, ka, bounds, cf);
            }

            // Note: Here we expect [ka, kb] to be small and splitting is unlikely.
            //                   p0 p1
            // |l|--|ka|kkkk|kb|--|P|-------------------|r|
            // |l|----------------|P|--|ka|kkk|kb|------|r|
            // |l|-----------|ka|k|P|k|kb|--------------|r|
            final int p1 = bounds[0];
            if (kb < p0) {
                // Entirely on left side
                r = p0 - 1;
            } else if (ka > p1) {
                // Entirely on right side
                l = p1 + 1;
            } else {
                // Pivot splits [ka, kb]. Expect ends to be close to the pivot and finish.
                // Here we set the bounds for use after median-of-medians pivot selection.
                // In the event there are many equal values this allows collecting those
                // known to be equal together when moving around the medians sample.
                if (kb > p1) {
                    sortSelectLeft(a, p1 + 1, r, kb);
                    bounds[0] = kb;
                }
                if (ka < p0) {
                    sortSelectRight(a, l, p0 - 1, ka);
                    p0 = ka;
                }
                return p0;
            }
            // Update sampling mode: set sign bit if did not reach expected size n
            cf |= n - r + l;
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
            select(a, 0, length - 1, k[0], k[0]);
            return;
        }
        if (n == 2 && Math.abs(k[0] - k[1]) < SORTSELECT_SIZE) {
            final int k1 = Math.min(k[0], k[1]);
            final int kn = Math.max(k[0], k[1]);
            select(a, 0, length - 1, k1, kn);
            return;
        }

        // TODO:
        // If the range [k1, kn] < length / 2 then bracket the ends using single-pivot
        // quickselect.

        final UpdatingInterval keys = IndexIntervals.createUpdatingInterval(k, n);
        final int k1 = keys.left();
        final int kn = keys.right();
        // If the keys are not separated then they are effectively a single key.
        // Use the sort select size. Any split of keys separated by this distance
        // will be finished on the next iteration.
        if (kn - k1 < SORTSELECT_SIZE) {
            select(a, 0, length - 1, k1, kn);
            return;
        }

        // Dual-pivot mode with small range sort length configured using index density
        //select(a, 0, length - 1, keys, dualPivotMaxDepth(length),
        //    dualPivotSortSelectSize(k1, kn, n));
        select(a, 0, length - 1, keys,
            dualPivotFlags(0, length - 1, k1, kn, n));
    }

    /**
     * Partition the array such that indices {@code k} correspond to their
     * correctly sorted value in the equivalent fully sorted array.
     *
     * <p>For all indices {@code [ka, kb]} and any index {@code i}:
     *
     * <pre>{@code
     * data[i < ka] <= data[ka] <= data[kb] <= data[kb < i]
     * }</pre>
     *
     * <p>This function accepts indices {@code [ka, kb]} that define the
     * range of indices to partition. It is expected that the range is small.
     *
     * <p>Uses an introselect variant. The quickselect is a Bentley-McIlroy partition
     * method by Kiwiel; switching to heapselect if convergence of quickselect is poor.
     * Pivot selection uses Floyd-Rivest subset sampling.
     *
     * <p>Data are assumed to contain no {@code NaN} values; mixed signed zeros may be
     * destroyed (the mixture updated during partitioning). The caller is responsible for
     * counting a mixture of signed zeros and restoring them if required.
     *
     * @param a Values.
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param ka First key of interest.
     * @param kb Last key of interest.
     */
    // package-private for benchmarking
    static void select(double[] a, int left, int right, int ka, int kb) {
        int l = left;
        int r = right;
        final int[] upper = {0};

        // TODO: Get the best introspection approach.
        // Try using the k-steps to half the sequence length.
        // k should be 4/5 according to Valois.
        // Ignore maxDepth
        // Limit recursion using the sum of partition lengths.
        // The sum must not exceed 2 * length so count down to zero using half of (r - l).
        // Since (r - l)/2 is subtracted at the start of the loop add it here.
        //int limit = r - l + ((r - l) >> 1);

        int maxDepth = singlePivotMaxDepth(r - l);
        while (true) {
            // select when ka and kb are close to the same end
            // |l|-----|ka|kkkkkkkk|kb|------|r|
            if (Math.min(kb - l, r - ka) < SORTSELECT_SIZE) {
                sortSelectRange(a, l, r, ka, kb);
                return;
            }
            if (--maxDepth < 0) {
            //limit -= (r - l) >> 1;
            //if (limit < 0) {
                // quickselect convergence is poor, switch to stopper function
                DEFAULT.stopperSelection.partition(a, l, r, ka, kb);
                return;
            }

            // Pick a pivot and partition
            int pivot;
            if (r - l > SELECT_SUB_SAMPLING_SIZE) {
                // Floyd-Rivest: use SELECT recursively on a sample of size S to get an estimate
                // for the (k-l+1)-th smallest element into a[k], biased slightly so that the
                // (k-l+1)-th element is expected to lie in the smaller set after partitioning.
                // Note: This targets ka and ignores kb for pivot selection.
                final int n = r - l + 1;
                final int i = ka - l + 1;
                final double z = Math.log(n);
                final double s = 0.5 * Math.exp(0.6666666666666666 * z);
                final double sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Integer.signum(i - (n >> 1));
                final int ll = Math.max(l, (int) (ka - i * s / n + sd));
                final int rr = Math.min(r, (int) (ka + (n - i) * s / n + sd));
                if (n > RANDOM_SUB_SAMPLING_SIZE) {
                    // Create a representative sample from [l, r] into [ll, rr]
                    final IntUnaryOperator rng = createFastRNG(n, ka);
                    // Shuffle [ll, ka) from [l, ka)
                    if (l < ll) {
                        for (int ii = ka; ii > ll;) {
                            // l + rand [0, ii - l + 1) : ii is currently ii+1
                            final int j = l + rng.applyAsInt(ii - l);
                            final double t = a[--ii];
                            a[ii] = a[j];
                            a[j] = t;
                        }
                    }
                    // Shuffle (k, rr] from (ka, r]
                    if (rr < r) {
                        for (int ii = ka; ii < rr;) {
                            // r - rand [0, r - ii + 1) : ii is currently ii-1
                            final int j = r - rng.applyAsInt(r - ii);
                            final double t = a[++ii];
                            a[ii] = a[j];
                            a[j] = t;
                        }
                    }
                }
                // Convert ln(n) to 2 * log2(n) for recursion depth
                select(a, ll, rr, ka, ka);
                pivot = ka;
            } else {
                // default pivot strategy
                pivot = pivotIndex(a, l, r);
            }

            // Single-pivot partitioning handling equal values.
            // TODO: Update pivot selection to ensure l<=p, r>=p.
            // Can be done using post processing of the FR sample.
            // ninther pivot already ensures this. Then use a modified
            // partition method that requires x[l] <= v <= x[r].
            final int p0 = partitionKBM(a, l, r, pivot, upper);
            final int p1 = upper[0];

            // Note: Here we expect [ka, kb] to be small and splitting is unlikely.
            //                   p0 p1
            // |l|--|ka|kkkk|kb|--|P|-------------------|r|
            // |l|----------------|P|--|ka|kkk|kb|------|r|
            // |l|-----------|ka|k|P|k|kb|--------------|r|
            if (kb < p0) {
                // Entirely on left side
                r = p0 - 1;
            } else if (ka > p1) {
                // Entirely on right side
                l = p1 + 1;
            } else {
                // Pivot splits [ka, kb]. Expect ends to be close to the pivot and finish.
                if (ka < p0) {
                    sortSelectRight(a, l, p0 - 1, ka);
                }
                if (kb > p1) {
                    sortSelectLeft(a, p1 + 1, r, kb);
                }
                return;
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
            if (kb - ka < SORTSELECT_SIZE) {
                // Switch to single-pivot mode with Floyd-Rivest sub-sampling
                select(a, l, r, ka, kb);
                return;
            }
            // Select when ka and kb are close to the same end,
            // or the entire range is small
            // |l|-----|ka|--------|kb|------|r|
            final int n = r - l;
            if (Math.min(kb - l, r - ka) < SORTSELECT_SIZE ||
                n < (flags & SORTSELECT_MASK)) {
                sortSelectRange(a, l, r, ka, kb);
                return;
            }
            if (flags < 0) {
                // Excess recursion, switch to heap select
                heapSelectRange2(a, l, r, ka, kb);
                return;
            }

            // Dual-pivot partitioning
//            final int n = r - l;
//            final int step = 1 + (n >>> 3) + (n >>> 6);
//            final int i3 = l + (n >>> 1);
//            final int i2 = i3 - step;
//            final int i1 = i2 - step;
//            final int i4 = i3 + step;
//            final int i5 = i4 + step;
//            Sorting.sort5(a, i1, i2, i3, i4, i5);
//            final int p0 = partitionDP(a, l, r, i2, i4, upper);
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
//                select(a, p3 + 1, r, k.splitRight(p2, p3), flags);
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
        final int pivot = pivotingStrategy.pivotIndex(data, left, right, left);
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
        //
        // Adapted from Floyd and Rivest (1975)
        // Algorithm 489: The Algorithm SELECT—for Finding the ith Smallest of n elements.
        // Comm. ACM. 18 (3): 173.
        //
        // Sub-script range checking has been eliminated by appropriate placement
        // of values at the ends to act as sentinels.
        //
        // left           i            j               right
        // |<=P|     <P   |     ???    |   >P          |>=P|
        //
        // At the end P is swapped back to the centre.
        //
        // |         <P          |P|             >P        |
        final double v = data[pivot];
        // swap(left, pivot)
        data[pivot] = data[l];
        if (data[r] > v) {
            // swap(right, left)
            data[l] = data[r];
            data[r] = v;
            // Here after the first swap: a[l] = v; a[r] > v
        } else {
            data[l] = v;
            // Here after the first swap: a[l] <= v; a[r] = v
        }
        int i = l;
        int j = r;
        while (i < j) {
            // swap(i, j)
            final double temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            do {
                ++i;
            } while (data[i] < v);
            do {
                --j;
            } while (data[j] > v);
        }
        // Move pivot back to the correct location from either l or r
        if (data[l] == v) {
            // data[j] <= v : swap(left, j)
            data[l] = data[j];
            data[j] = v;
        } else {
            // data[j+1] > v : swap(j+1, right)
            data[r] = data[++j];
            data[j] = v;
        }
        upper[0] = j;
        return j;
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
        // - Added a fast-forward over initial range containing the pivot.
        // - Changed the final move to perform the minimum moves.

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
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method by Kiwiel.
     *
     * @param x Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param pivot Pivot index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    static int partitionKBM(double[] x, int l, int r, int pivot, int[] upper) {
        // Single-pivot Bentley-McIlroy quicksort handling equal keys.
        //
        // Partition data using pivot v into less-than, greater-than or equal.
        // The basic idea is to work with the 5 inner parts of the array [ll, rr]
        // by positioning sentinels at l and r:
        //
        // |l |ll   p|          |i          j|         |q   rr| r|           (6.1)
        // |<v|  ==v |     <v   |     ???    |   >v    | ==v  |>v|
        //
        // until the middle part is empty or just contains an element equal to the pivot:
        //
        // |ll   p|              j|   |i          |q   rr|                   (6.2)
        // |  ==v |     <v        |==v|     >v    | ==v  |
        //
        // i.e. j = i-1 or i-2, then swap the ends into the middle:
        //
        // |ll              |a         d|              rr|                   (6.3)
        // |        <v      |     ==v   |      >v        |
        //
        // Adapted from Kiwiel (2005) "On Floyd and Rivest's SELECT algorithm"
        // Theoretical Computer Science 347, 214-238.
        // This is the safeguarded ternary partition Scheme E with modification to
        // prevent vacuous swaps of equal keys (section 5.6) in Kiwiel (2003)
        // Partitioning schemes for quicksort and quickselect,
        // Technical report, Systems Research Institute, Warsaw.
        // http://arxiv.org/abs/cs.DS/0312054
        //
        // Note: The difference between this and Sedgewick's BM is the use of sentinals
        // at either end to remove index checks at both ends and changing the behaviour
        // when i and j meet on a pivot value.
        //
        // The listing in Kiwiel (2005) has been updated:
        // - p and q mark the *inclusive* end of ==v regions.
        // - Added a fast-forward over initial range containing the pivot.
        // - Vector swap is optimised given one side of the exchange is v.

        final double v = x[pivot];
        x[pivot] = x[l];
        x[l] = v;

        int ll = l;
        int rr = r;

        // Ensure x[l] <= v <= x[r]
        if (v < x[r]) {
            --rr;
        } else if (v > x[r]) {
            x[l] = x[r];
            x[r] = v;
            ++ll;
        }

        // Position p and q for pre-in/decrement to write into edge pivot regions
        // Fast-forward over equal regions to reduce swaps
        int p = l;
        while (x[p + 1] == v) {
            if (++p == rr) {
                // Edge-case: constant value in [ll, rr]
                // Return the full range [l, r] as a single edge element
                // will also be partitioned.
                upper[0] = r;
                return l;
            }
        }
        // Cannot overrun as the prior scan using p stopped before the end
        int q = r;
        while (x[q - 1] == v) {
            --q;
        }

        // Check set-up: [ll, p] and [q, rr] are pivot
        for (int i = ll; i <= p; i++) {
            assert x[i] == v;
        }
        for (int i = q; i <= rr; i++) {
            assert x[i] == v;
        }

        // Position for pre-in/decrement
        int i = p;
        int j = q;

        for (;;) {
            do {
                ++i;
            } while (x[i] < v);
            do {
                --j;
            } while (x[j] > v);
            // Here x[j] <= v <= x[i]
            if (i >= j) {
                if (i == j) {
                    // x[i]=x[j]=v; update to leave the pivot in between (j, i)
                    ++i;
                    --j;
                }
                break;
            }
            //swap(x, i, j)
            final double vi = x[j];
            final double vj = x[i];
            x[i] = vi;
            x[j] = vj;
            // Move the equal values to the ends
            if (vi == v) {
                x[i] = x[++p];
                x[p] = v;
            }
            if (vj == v) {
                x[j] = x[--q];
                x[q] = v;
            }
        }

        // Set [a, d] (p and q are offset by 1 from Kiwiel)
        final int a = ll + j - p;
        upper[0] = rr - q + i;

        // Vector swap x[a:b] <-> x[b+1:c] means the first m = min(b+1-a, c-b)
        // elements of the array x[a:c] are exchanged with its last m elements.
        // x[ll:p] <-> x[p+1:j]
//        int m = Math.min(p + 1 - ll, j - p);
//        for (int k = ll; --m >= 0; --j, ++k) {
//            x[k] = x[j];
//            x[j] = v;
//        }
        vectorSwapL(x, ll, p, j, v);
        // x[i:q-1] <-> x[q:rr]
//        m = Math.min(q - i, rr - q + 1);
//        for (int k = rr; --m >= 0; ++i, --k) {
//            x[k] = x[i];
//            x[i] = v;
//        }
        vectorSwapR(x, i, q - 1, rr, v);
        return a;
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
     * Expand a partition around a single pivot. Partitioning exchanges array
     * elements such that all elements smaller than pivot are before it and all
     * elements larger than pivot are after it. The central region is already
     * partitioned.
     *
     * <pre>{@code
     * |l             |s   |p0 p1|   e|                r|
     * |    ???       | <P | ==P | >P |        ???      |
     * }</pre>
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param start Start of the partition range (inclusive).
     * @param end End of the partitioned range (inclusive).
     * @param pivot0 Lower pivot location (inclusive).
     * @param pivot1 Upper pivot location (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range [k1].
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    private static int expandPartitionT1(double[] a, int left, int right, int start, int end,
        int pivot0, int pivot1, int[] upper) {
        // 3-way partition of the data using a pivot value into
        // less-than, equal or greater-than.
        // Based on Sedgewick's Bentley-McIroy partitioning: always swap i<->j then
        // check for equal to the pivot and move again.
        //
        // Move sentinels from start and end to left and right. Scan towards the
        // sentinels until >=,<=. Swap then move == to the pivot region.
        //           <-i                           j->
        // |l |        |            |p0  p1|       |             | r|
        // |>=|   ???  |     <      |  ==  |   >   |     ???     |<=|
        //
        // When either i or j reach the edge perform finishing loop.
        // Finish loop for a[j] <= v replaces j with p1+1, moves value to p0
        // for < or p1+1 for == and updates the pivot range:
        //                                             j->
        // |l                       |p0  p1|           |         | r|
        // |         <              |  ==  |       >   |   ???   |<=|

        // Positioned for pre-in/decrement to write to pivot region
        int p0 = pivot0;
        int p1 = pivot1;
        final double v = a[p0];
        if (a[left] < v) {
            // a[left] is not a sentinel
            final double w = a[left];
            if (a[right] > v) {
                // Most likely case: ends can be sentinels
                a[left] = a[right];
                a[right] = w;
            } else {
                // a[right] is a sentinel; use pivot for left
                a[left] = v;
                a[p0] = w;
                p0++;
            }
        } else if (a[right] > v) {
            // a[right] is not a sentinel; use pivot
            a[p1] = a[right];
            p1--;
            a[right] = v;
        }

        int i = start;
        int j = end;
        while (true) {
            do {
                --i;
            } while (a[i] < v);
            do {
                ++j;
            } while (a[j] > v);
            final double vj = a[i];
            final double vi = a[j];
            a[i] = vi;
            a[j] = vj;
            // Move the equal values to pivot region
            if (vi == v) {
                a[i] = a[--p0];
                a[p0] = v;
            }
            if (vj == v) {
                a[j] = a[++p1];
                a[p1] = v;
            }
            // Termination check and finishing loops.
            // Note: this works even if pivot region is zero length (p1 == p0-1)
            // due to pivot use as a sentinel on one side because we pre-inc/decrement
            // one side and post-inc/decrement the other side.
            if (i == left) {
                while (j < right) {
                    do {
                        ++j;
                    } while (a[j] > v);
                    final double w = a[j];
                    // Move upper bound of pivot region
                    a[j] = a[++p1];
                    a[p1] = v;
                    if (w < v) {
                        // Move lower bound of pivot region
                        a[p0] = w;
                        p0++;
                    }
                }
                break;
            }
            if (j == right) {
                while (i > left) {
                    do {
                        --i;
                    } while (a[i] < v);
                    final double w = a[i];
                    // Move lower bound of pivot region
                    a[i] = a[--p0];
                    a[p0] = v;
                    if (w > v) {
                        // Move upper bound of pivot region
                        a[p1] = w;
                        p1--;
                    }
                }
                break;
            }
        }

        upper[0] = p1;
        return p0;
    }

    /**
     * Expand a partition around a single pivot. Partitioning exchanges array
     * elements such that all elements smaller than pivot are before it and all
     * elements larger than pivot are after it. The central region is already
     * partitioned.
     *
     * <pre>{@code
     * |l             |s   |p0 p1|   e|                r|
     * |    ???       | <P | ==P | >P |        ???      |
     * }</pre>
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>This is similar to {@link #expandPartitionT1(double[], int, int, int, int, int, int, int[])}
     * with a change to binary partitioning.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param start Start of the partition range (inclusive).
     * @param end End of the partitioned range (inclusive).
     * @param pivot0 Lower pivot location (inclusive).
     * @param pivot1 Upper pivot location (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range [k1].
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    private static int expandPartitionB1(double[] a, int left, int right, int start, int end,
        int pivot0, int pivot1, int[] upper) {
        // 2-way partition of the data using a pivot value into
        // less-than, or greater-than.
        //
        // Move sentinels from start and end to left and right. Scan towards the
        // sentinels until >=,<= then swap.
        //           <-i                           j->
        // |l |        |              | p|         |             | r|
        // |>=|   ???  |     <        |==|     >   |     ???     |<=|
        //
        // When either i or j reach the edge perform finishing loop.
        // Finish loop for a[j] <= v replaces j with p1+1, moves value to p
        // and moves the pivot up:
        //                                            j->
        // |l                         | p|            |         | r|
        // |         <                |==|        >   |   ???   |<=|

        // Pivot may be moved to use as a sentinel
        int p = pivot0;
        final double v = a[p];
        if (a[left] < v) {
            // a[left] is not a sentinel
            final double w = a[left];
            if (a[right] > v) {
                // Most likely case: ends can be sentinels
                a[left] = a[right];
                a[right] = w;
            } else {
                // a[right] is a sentinel; use pivot for left
                a[left] = v;
                a[p] = w;
                p++;
            }
        } else if (a[right] > v) {
            // a[right] is not a sentinel; use pivot
            a[p] = a[right];
            p--;
            a[right] = v;
        }

        int i = start;
        int j = end;
        while (true) {
            do {
                --i;
            } while (a[i] < v);
            do {
                ++j;
            } while (a[j] > v);
            final double vj = a[i];
            final double vi = a[j];
            a[i] = vi;
            a[j] = vj;
            // Termination check and finishing loops.
            // These reset the pivot if it was moved then slide it as required.
            if (i == left) {
                if (j == right) {
                    break;
                }
                // Reset the pivot and sentinel
                if (p < pivot0) {
                    // Pivot is in right; a[p] <= v
                    a[right] = a[p];
                    a[p] = v;
                } else if (p > pivot0) {
                    // Pivot was in left (now swapped to j); a[p] >= v
                    a[j] = a[p];
                    a[p] = v;
                }
                while (j < right) {
                    do {
                        ++j;
                    } while (a[j] > v);
                    // Move pivot
                    a[p] = a[j];
                    a[j] = a[++p];
                    a[p] = v;
                }
                break;
            }
            if (j == right) {
                if (i == left) {
                    break;
                }
                // Reset the pivot and sentinel
                if (p < pivot0) {
                    // Pivot was in right (now swapped to i); a[p] <= v
                    a[i] = a[p];
                    a[p] = v;
                } else if (p > pivot0) {
                    // Pivot is in left; a[p] >= v
                    a[left] = a[p];
                    a[p] = v;
                }
                while (i > left) {
                    do {
                        --i;
                    } while (a[i] < v);
                    // Move pivot
                    a[p] = a[i];
                    a[i] = a[--p];
                    a[p] = v;
                }
                break;
            }
        }

        upper[0] = p;
        return p;
    }

    /**
     * Expand a partition around a single pivot. Partitioning exchanges array
     * elements such that all elements smaller than pivot are before it and all
     * elements larger than pivot are after it. The central region is already
     * partitioned.
     *
     * <pre>{@code
     * |l             |s   |p0 p1|   e|                r|
     * |    ???       | <P | ==P | >P |        ???      |
     * }</pre>
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>This is similar to {@link #expandPartitionT1(double[], int, int, int, int, int, int, int[])}
     * with a change to how the end-point sentinels are created. It does not use the pivot
     * but uses values at start and end. This increases the length of the lower/upper ranges
     * by 1 for the main scan.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param start Start of the partition range (inclusive).
     * @param end End of the partitioned range (inclusive).
     * @param pivot0 Lower pivot location (inclusive).
     * @param pivot1 Upper pivot location (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range [k1].
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    private static int expandPartitionT2(double[] a, int left, int right, int start, int end,
        int pivot0, int pivot1, int[] upper) {
        // 3-way partition of the data using a pivot value into
        // less-than, equal or greater-than.
        // Based on Sedgewick's Bentley-McIroy partitioning: always swap i<->j then
        // check for equal to the pivot and move again.
        //
        // Move sentinels from start and end to left and right. Scan towards the
        // sentinels until >=,<=. Swap then move == to the pivot region.
        //           <-i                           j->
        // |l |        |            |p0  p1|       |             | r|
        // |>=|   ???  |     <      |  ==  |   >   |     ???     |<=|
        //
        // When either i or j reach the edge perform finishing loop.
        // Finish loop for a[j] <= v replaces j with p1+1, moves value to p0
        // for < or p1+1 for == and updates the pivot range:
        //                                             j->
        // |l                       |p0  p1|           |         | r|
        // |         <              |  ==  |       >   |   ???   |<=|

        final double v = a[pivot0];
        // Use start/end as sentinels.
        // This requires start != end
        assert start != end;
        double vi = a[start];
        double vj = a[end];
        a[start] = a[left];
        a[end] = a[right];
        a[left] = vj;
        a[right] = vi;

        int i = start + 1;
        int j = end - 1;

        // Positioned for pre-in/decrement to write to pivot region
        int p0 = pivot0 == start ? i : pivot0;
        int p1 = pivot1 == end ? j : pivot1;

        while (true) {
            do {
                --i;
            } while (a[i] < v);
            do {
                ++j;
            } while (a[j] > v);
            vj = a[i];
            vi = a[j];
            a[i] = vi;
            a[j] = vj;
            // Move the equal values to pivot region
            if (vi == v) {
                a[i] = a[--p0];
                a[p0] = v;
            }
            if (vj == v) {
                a[j] = a[++p1];
                a[p1] = v;
            }
            // Termination check and finishing loops.
            // Note: this works even if pivot region is zero length (p1 == p0-1
            // due to single length pivot region at either start/end) because we pre-inc/decrement
            // one side and post-inc/decrement the other side.
            if (i == left) {
                while (j < right) {
                    do {
                        ++j;
                    } while (a[j] > v);
                    final double w = a[j];
                    // Move upper bound of pivot region
                    a[j] = a[++p1];
                    a[p1] = v;
                    a[p0] = w;
                    // Move lower bound of pivot region
                    p0 += w != v ? 1 : 0;
                    //if (w != v) {
                    //    p0++;
                    //}
                }
                break;
            }
            if (j == right) {
                while (i > left) {
                    do {
                        --i;
                    } while (a[i] < v);
                    final double w = a[i];
                    // Move lower bound of pivot region
                    a[i] = a[--p0];
                    a[p0] = v;
                    a[p1] = w;
                    // Move upper bound of pivot region
                    p1 -= w != v ? 1 : 0;
                    //if (w != v) {
                    //    p1--;
                    //}
                }
                break;
            }
        }

        upper[0] = p1;
        return p0;
    }

    /**
     * Expand a partition around a single pivot. Partitioning exchanges array
     * elements such that all elements smaller than pivot are before it and all
     * elements larger than pivot are after it. The central region is already
     * partitioned.
     *
     * <pre>{@code
     * |l             |s   |p0 p1|   e|                r|
     * |    ???       | <P | ==P | >P |        ???      |
     * }</pre>
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>This is similar to {@link #expandPartitionT2(double[], int, int, int, int, int, int, int[])}
     * with a change to binary partitioning. It is simpler than
     * {@link #expandPartitionB1(double[], int, int, int, int, int, int, int[])} as the pivot is
     * not moved.
     *
     * @param a Data array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param start Start of the partition range (inclusive).
     * @param end End of the partitioned range (inclusive).
     * @param pivot0 Lower pivot location (inclusive).
     * @param pivot1 Upper pivot location (inclusive).
     * @param upper Upper bound (inclusive) of the pivot range [k1].
     * @return Lower bound (inclusive) of the pivot range [k0].
     */
    private static int expandPartitionB2(double[] a, int left, int right, int start, int end,
        int pivot0, int pivot1, int[] upper) {
        // 2-way partition of the data using a pivot value into
        // less-than, or greater-than.
        //
        // Move sentinels from start and end to left and right. Scan towards the
        // sentinels until >=,<= then swap.
        //           <-i                           j->
        // |l |        |              | p|         |             | r|
        // |>=|   ???  |     <        |==|     >   |     ???     |<=|
        //
        // When either i or j reach the edge perform finishing loop.
        // Finish loop for a[j] <= v replaces j with p1+1, moves value to p
        // and moves the pivot up:
        //                                            j->
        // |l                         | p|            |         | r|
        // |         <                |==|        >   |   ???   |<=|

        // Pivot
        int p = pivot0;
        final double v = a[p];
        // Use start/end as sentinels.
        // This requires start != end
        assert start != end;
        // Note: Must not move pivot as this invalidates the finishing loops.
        // See logic in method B1 to see added complexity of pivot location.
        // This method is not better than T2 for data with no repeat elements
        // and is slower for repeat elements when used with the improved
        // versions (e.g. linearBFPRTImproved).
        if (p == start || p == end) {
            return expandPartitionB1(a, left, right, start, end, pivot0, pivot1, upper);
        }
        double vi = a[start];
        double vj = a[end];
        a[start] = a[left];
        a[end] = a[right];
        a[left] = vj;
        a[right] = vi;

        int i = start + 1;
        int j = end - 1;
        while (true) {
            do {
                --i;
            } while (a[i] < v);
            do {
                ++j;
            } while (a[j] > v);
            vj = a[i];
            vi = a[j];
            a[i] = vi;
            a[j] = vj;
            // Termination check and finishing loops
            if (i == left) {
                while (j < right) {
                    do {
                        ++j;
                    } while (a[j] > v);
                    // Move pivot
                    a[p] = a[j];
                    a[j] = a[++p];
                    a[p] = v;
                }
                break;
            }
            if (j == right) {
                while (i > left) {
                    do {
                        --i;
                    } while (a[i] < v);
                    // Move pivot
                    a[p] = a[i];
                    a[i] = a[--p];
                    a[p] = v;
                }
                break;
            }
        }

        upper[0] = p;
        return p;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>The index {@code k} is the target element. This method ignores this value.
     * The value is included to match the method signature of the {@link SPEPartition} interface.
     * Assumes the range {@code r - l >= 4}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Blum, Floyd, Pratt, Rivest, and Tarjan (BFPRT) median-of-medians algorithm
     * with medians of 5 with the sample medians computed in the first quintile.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int linearBFPRTBaseline(double[] a, int l, int r, int k, int[] upper) {
        // Adapted from Alexandrescu (2016), algorithm 3.
        // Moves the responsibility for selection when r-l <= 4 to the caller.
        // Compute the median of each contiguous set of 5 to the first quintile.
        int rr = l - 1;
        for (int e = l + 4; e <= r; e += 5) {
            Sorting.median5d(a, e - 4, e - 3, e - 2, e - 1, e);
            // Median to first quintile
            final double v = a[e - 2];
            a[e - 2] = a[++rr];
            a[rr] = v;
        }
        final int m = (l + rr + 1) >>> 1;
        // mutual recursion
        quickSelect(this::linearBFPRTBaseline, a, l, rr, m, m, upper);
        // Note: repartions already partitioned data [l, rr]
        return spFunction.partition(a, l, r, m, upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>The index {@code k} is the target element. This method ignores this value.
     * The value is included to match the method signature of the {@link SPEPartition} interface.
     * Assumes the range {@code r - l >= 8}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with medians of 3 with the samples computed in the first tertile and 9th-tile.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int linearRepeatedStepBaseline(double[] a, int l, int r, int k, int[] upper) {
        // Adapted from Alexandrescu (2016), algorithm 5.
        // Moves the responsibility for selection when r-l <= 8 to the caller.
        // Compute the median of each contiguous set of 3 to the first tertile, and repeat.
        int j = l - 1;
        for (int e = l + 2; e <= r; e += 3) {
            Sorting.sort3(a, e - 2, e - 1, e);
            // Median to first tertile
            final double v = a[e - 1];
            a[e - 1] = a[++j];
            a[j] = v;
        }
        int rr = l - 1;
        for (int e = l + 2; e <= j; e += 3) {
            Sorting.sort3(a, e - 2, e - 1, e);
            // Median to first 9th-tile
            final double v = a[e - 1];
            a[e - 1] = a[++rr];
            a[rr] = v;
        }
        final int m = (l + rr + 1) >>> 1;
        // mutual recursion
        quickSelect(this::linearRepeatedStepBaseline, a, l, rr, m, m, upper);
        // Note: repartions already partitioned data [l, rr]
        return spFunction.partition(a, l, r, m, upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>The index {@code k} is the target element. This method ignores this value.
     * The value is included to match the method signature of the {@link SPEPartition} interface.
     * Assumes the range {@code r - l >= 4}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Blum, Floyd, Pratt, Rivest, and Tarjan (BFPRT) median-of-medians algorithm
     * with medians of 5 with the sample medians computed in the central quintile.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int linearBFPRTImproved(double[] a, int l, int r, int k, int[] upper) {
        // Adapted from Alexandrescu (2016), algorithm 6.
        // Moves the responsibility for selection when r-l <= 4 to the caller.
        // Compute the median of each non-contiguous set of 5 to the middle quintile.
        final int f = (r - l + 1) / 5;
        final int f3 = 3 * f;
        // middle quintile: [2f:3f)
        final int s = l + (f << 1);
        final int e = s + f - 1;
        for (int i = l, j = s; i < s; i += 2, j++) {
            Sorting.median5d(a, i, i + 1, j, f3 + i, f3 + i + 1);
        }
        final int m = (s + e + 1) >>> 1;
        // mutual recursion
        quickSelect(this::linearBFPRTImproved, a, s, e, m, m, upper);
        //return spFunction.partition(a, l, r, m, upper);
        // broken
        return expandFunction.partition(a, l, r, s, e, upper[0], upper[1], upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>The index {@code k} is the target element. This method ignores this value.
     * The value is included to match the method signature of the {@link SPEPartition} interface.
     * Assumes the range {@code r - l >= 8}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with medians of 3 with the samples computed in the middle tertile and 9th-tile.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int linearRepeatedStepImproved(double[] a, int l, int r, int k, int[] upper) {
        // Adapted from Alexandrescu (2016), algorithm 7.
        // Moves the responsibility for selection when r-l <= 8 to the caller.
        // Compute the median of each non-contiguous set of 3 to the middle tertile, and repeat.
        final int f = (r - l + 1) / 9;
        final int f3 = 3 * f;
        // i in middle tertile [3f:6f)
        for (int i = l + f3, e = l + (f3 << 1); i < e; i++) {
            Sorting.sort3(a, i - f3, i, i + f3);
        }
        // i in middle 9th-tile: [4f:5f)
        final int s = l + (f << 2);
        final int e = s + f - 1;
        for (int i = s; i <= e; i++) {
            Sorting.sort3(a, i - f, i, i + f);
        }
        // TODO - control flag for adaptive k
        final int m = (s + e + 1) >>> 1;
        // mutual recursion
        quickSelect(this::linearRepeatedStepImproved, a, s, e, m, m, upper);
        return expandFunction.partition(a, l, r, s, e, upper[0], upper[1], upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Assumes the range {@code r - l >= 8}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with medians of 3 with the samples computed in the middle tertile and 9th-tile.
     * The pivot chosen from the sample is adaptive using the input {@code k}.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int linearRepeatedStepAdaptive(double[] a, int l, int r, int k, int[] upper) {
        // Adapted from Alexandrescu (2016), algorithm 8.
        // Moves the responsibility for selection when r-l <= 8 to the caller.
        // Compute the median of each non-contiguous set of 3 to the middle tertile, and repeat.
        final int f = (r - l + 1) / 9;
        final int f3 = 3 * f;
        // i in middle tertile [3f:6f)
        for (int i = l + f3, e = l + (f3 << 1); i < e; i++) {
            Sorting.sort3(a, i - f3, i, i + f3);
        }
        // i in middle 9th-tile: [4f:5f)
        final int s = l + (f << 2);
        final int e = s + f - 1;
        for (int i = s; i <= e; i++) {
            Sorting.sort3(a, i - f, i, i + f);
        }
        // Adaption to target kf/|A|
        final int p = s + mapK(k, l, r, f);
        // mutual recursion
        quickSelect(this::linearRepeatedStepAdaptive, a, s, e, p, p, upper);
        return expandFunction.partition(a, l, r, s, e, upper[0], upper[1], upper);
    }

    /**
     * Map the index {@code [l <= k <= r]} to a new index in {@code [0, n)}.
     *
     * @param k Index.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param n New upper bound (exclusive).
     * @return the mapped index in [0, n)
     */
    private static int mapK(int k, int l, int r, int n) {
        // If k==r this returns n-1
        return mapDistance(k - l, l, r, n);
    }

    /**
     * Map the distance from the edge of {@code [l, r]} to a new distance in {@code [0, n)}.
     *
     * @param d Distance from the edge in {@code [0, r - l]}.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param n New upper bound (exclusive).
     * @return the mapped index in [0, n)
     */
    private static int mapDistance(int d, int l, int r, int n) {
        // If distance==r-l this returns n-1
        //return (int) ((double) (k - l) * n / (r - l + 1.0));
        return (int) Math.round(d * (n - 1.0) / (r - l));
        //return n >>> 1;
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Assumes the range {@code r - l >= 8}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with the median of 3 then median of 3; the final sample is placed in the
     * 5-th 9th-tile; the pivot chosen from the sample is adaptive using the input {@code k}.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @param flags Control flags.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int repeatedStep(double[] a, int l, int r, int k, int[] upper, int flags) {
        // Adapted from Alexandrescu (2016), algorithm 8.
        // Moves the responsibility for selection when r-l <= 8 to the caller.
        final int f = (r - l + 1) / 9;
        if (flags < 0) {
            // i in tertile [3f:6f)
            final int f3 = 3 * f;
            for (int i = l + f3, e = l + (f3 << 1); i < e; i++) {
                Sorting.sort3(a, i - f3, i, i + f3);
            }
        }
        // i in 9th-tile: [4f:5f)
        final int s = l + (f << 2);
        final int e = s + f - 1;
        for (int i = s; i <= e; i++) {
            Sorting.sort3(a, i - f, i, i + f);
        }
        // Adaption to target kf/|A|
        int p = s + mapK(k, l, r, f);
        p = quickSelectAdaptive(a, s, e, p, p, upper, flags & qaFlagMask);
        return expandFunction.partition(a, l, r, s, e, p, upper[0], upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Assumes the range {@code r - l >= 11}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with the lower median of 4 then either median of 3 with the final sample placed in the
     * 5-th 12th-tile, or min of 3 with the final sample in the 4th 12-th tile;
     * the pivot chosen from the sample is adaptive using the input {@code k}.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @param flags Control flags.
     * @param far Set to {@code true} to perform repeatedStepFarLeft.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int repeatedStepLeft(double[] a, int l, int r, int k, int[] upper, int flags,
        boolean far) {
        // Adapted from Alexandrescu (2016), algorithm 9 and 10.
        // Moves the responsibility for selection when r-l <= 11 to the caller.
        final int f = (r - l + 1) >> 2;
        if (flags < 0) {
            // i in 2nd quartile
            final int f2 = f + f;
            for (int i = l + f, e = l + f2; i < e; i++) {
                Sorting.lowerMedian4(a, i - f, i, i + f, i + f2);
            }
        }
        final int fp = f / 3;
        int s;
        int e;
        if (far) {
            final int fp2 = fp << 1;
            // i in 4th 12th-tile
            s = l + f;
            e = s + fp - 1;
            for (int i = s; i <= e; i++) {
                // min into i
                if (a[i] > a[i + fp]) {
                    final double u = a[i];
                    a[i] = a[i + fp];
                    a[i + fp] = u;
                }
                if (a[i] > a[i + fp2]) {
                    final double v = a[i];
                    a[i] = a[i + fp2];
                    a[i + fp2] = v;
                }
            }
        } else {
            // i in 5th 12-th tile
            // This is a modification from Alexandrescu rather than 4th 12-th tile.
            // Otherwise this is identical to repeatedStepFarLeft. This matches the text
            // stating |A|/6 are on the left (1/4 * 1/3 * 2).
            s = l + f + fp;
            e = s + fp - 1;
            for (int i = s; i <= e; i++) {
                Sorting.sort3(a, i - fp, i, i + fp);
            }
        }
        // Adaption to target kf'/|A|
        int p = s + mapDistance(k - l, l, r, fp);
        p = quickSelectAdaptive(a, s, e, p, p, upper, flags & qaFlagMask);
        return expandFunction.partition(a, l, r, s, e, p, upper[0], upper);
    }

    /**
     * Partition an array slice around a pivot. Partitioning exchanges array elements such
     * that all elements smaller than pivot are before it and all elements larger than
     * pivot are after it.
     *
     * <p>Assumes the range {@code r - l >= 11}; the caller is responsible for selection on a smaller
     * range.
     *
     * <p>Note: Requires that the range contains no NaN values.
     * This does not respect the ordering of signed zeros.
     *
     * <p>Uses the Chen and Dumitrescu repeated step median-of-medians-of-medians algorithm
     * with the upper median of 4 then either median of 3 with the final sample placed in the
     * 8-th 12th-tile, or max of 3 with the final sample in the 9th 12-th tile;
     * the pivot chosen from the sample is adaptive using the input {@code k}.
     *
     * @param a Data array.
     * @param l Lower bound (inclusive).
     * @param r Upper bound (inclusive).
     * @param k Target index.
     * @param upper Upper bound (inclusive) of the pivot range.
     * @param flags Control flags.
     * @param far Set to {@code true} to perform repeatedStepFarRight.
     * @return Lower bound (inclusive) of the pivot range.
     */
    private int repeatedStepRight(double[] a, int l, int r, int k, int[] upper, int flags,
        boolean far) {
        // Mirror image repeatedStepLeft using upper median into 3rd quartile
        final int f = (r - l + 1) >> 2;
        if (flags < 0) {
            // i in 3rd quartile
            final int f2 = f + f;
            for (int i = r - f, e = r - f2; i > e; i--) {
                Sorting.upperMedian4(a, i - f2, i - f, i, i + f);
            }
        }
        final int fp = f / 3;
        int s;
        int e;
        if (far) {
            // i in 9th 12th-tile
            e = r - f;
            s = e - fp + 1;
            final int fp2 = fp << 1;
            for (int i = s; i <= e; i++) {
                // max into i
                if (a[i] < a[i - fp]) {
                    final double u = a[i];
                    a[i] = a[i - fp];
                    a[i - fp] = u;
                }
                if (a[i] < a[i - fp2]) {
                    final double v = a[i];
                    a[i] = a[i - fp2];
                    a[i - fp2] = v;
                }
            }
        } else {
            // i in 8th 12-th tile
            e = r - f - fp;
            s = e - fp + 1;
            for (int i = s; i <= e; i++) {
                Sorting.sort3(a, i - fp, i, i + fp);
            }
        }
        // Adaption to target kf'/|A|
        int p = e - mapDistance(r - k, l, r, fp);
        p = quickSelectAdaptive(a, s, e, p, p, upper, flags & qaFlagMask);
        return expandFunction.partition(a, l, r, s, e, p, upper[0], upper);
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
     * Convert {@code ln(n)} to the single-pivot max depth.
     *
     * @param x ln(n)
     * @return the maximum recursion depth
     */
    private int lnNtoMaxDepthSinglePivot(double x) {
        final double maxDepth = x * LOG2_E;
        return (int) Math.floor(maxDepth * recursionMultiple) + recursionConstant;
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

    /**
     * Creates the source of random numbers in {@code [0, n)}.
     * This is configurable via the control flags.
     *
     * @param n Data length.
     * @param k Target index.
     * @return the RNG
     */
    private IntUnaryOperator createRNG(int n, int k) {
        // Configurable
        if ((controlFlags & FLAG_MSWS) != 0) {
            // Middle-Square Weyl Sequence is fastest int generator
            final UniformRandomProvider rng = RandomSource.MSWS.create(n * 31L + k);
            if ((controlFlags & FLAG_BIASED_RANDOM) != 0) {
                // result = i * [0, 2^32) / 2^32
                return i -> (int) ((i * Integer.toUnsignedLong(rng.nextInt())) >>> Integer.SIZE);
            }
            return rng::nextInt;
        }
        if ((controlFlags & FLAG_SPLITTABLE_RANDOM) != 0) {
            final SplittableRandom rng = new SplittableRandom(n * 31L + k);
            if ((controlFlags & FLAG_BIASED_RANDOM) != 0) {
                // result = i * [0, 2^32) / 2^32
                return i -> (int) ((i * Integer.toUnsignedLong(rng.nextInt())) >>> Integer.SIZE);
            }
            return rng::nextInt;
        }
        return createFastRNG(n, k);
    }

    /**
     * Creates the source of random numbers in {@code [0, n)}.
     *
     * @param n Data length.
     * @param k Target index.
     * @return the RNG
     */
    static IntUnaryOperator createFastRNG(int n, int k) {
        return new Gen(n * 31L + k);
    }

    /**
     * Random generator for numbers in {@code [0, n)}.
     * The random sample should be fast in preference to statistically robust.
     * Here we implement a biased sampler for the range [0, n)
     * as n * f with f a fraction with base 2^32.
     * Source of randomness is a 64-bit LCG using the constants from MMIX by Donald Knuth.
     * https://en.wikipedia.org/wiki/Linear_congruential_generator
     */
    private static final class Gen implements IntUnaryOperator {
        /** LCG state. */
        private long s;

        /**
         * @param seed Seed.
         */
        Gen(long seed) {
            // Update state
            this.s = seed * 6364136223846793005L + 1442695040888963407L;
        }

        @Override
        public int applyAsInt(int n) {
            final long x = s;
            // Update state
            s = s * 6364136223846793005L + 1442695040888963407L;
            // Use the upper 32-bits from the state as the random 32-bit sample
            // result = n * [0, 2^32) / 2^32
            return (int) ((n * (x >>> Integer.SIZE)) >>> Integer.SIZE);
        }
    }
}
