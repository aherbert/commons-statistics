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
 * Select indices in array data.
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
 * <p>This implementation can select on multiple indices and will handle duplicate and
 * unordered indices. The method detects ordered indices (with or without duplicates) and
 * uses this during processing. Passing ordered indices is recommended if the order is already
 * known; for example using uniform spacing through the array data, or to select the top and
 * bottom {@code n} values from the data.
 *
 * <p>A quickselect adaptive method is used for single indices. This uses analysis of the
 * partition sizes after each division to update the algorithm mode. If the partition
 * containing the target does not sufficiently reduce in size then the algorithm is
 * progressively changed to use partitions with guaranteed margins. This ensures a set fraction
 * of data is eliminated each step and worse-case linear run time performance. The initial
 * mode uses Floyd-Rivest sampling and progresses through to adaptive quickselect with
 * fixed margins. This method can handle a range of indices {@code [ka, kb]} with a small
 * separation by targeting the start of the range {@code ka} and then selecting the remaining
 * elements {@code (ka, kb]} that are at the edge of the partition bounded by {@code ka}.
 *
 * <p>Multiple keys are partitioned collectively using an introsort method which only recurses
 * into partitions containing indices. Excess recursion will trigger use of a heapselect
 * on the remaining range of indices ensuring non-quadratic worse case performance. Any
 * partition containing a single index, adjacent pair of indices, or range of indices with a
 * small separation will use the quickselect adaptive method for single keys. Note that the
 * maximum number of times that {@code n} indices can be split is {@code n - 1} before all
 * indices are handled as singles.
 *
 * <p>References
 *
 * <p>Quickselect is introduced in Hoare [1]. This selects an element {@code k} from {@code n}
 * using repeat division of the data around a partition element, recursing into the
 * partition that contains {@code k}.
 *
 * <p>Introsort/select is introduced in Musser [2]. This detects excess recursion in
 * quicksort/select and reverts to a heapsort or linear select to achieve an improved worst
 * case bound.
 *
 * <p>Use of dual-pivot quickselect is analysed in Wild et al [3] and shown to require
 * marginally more comparisons than single-pivot quickselect on a uniformly chosen order
 * statistic {@code k} and extremal order statistic (see table 1, page 19). This implementation
 * uses dual-pivot quickselect to partition a range with multiple indices into smaller ranges
 * with single indices.
 *
 * <p>Use of sampling to identify a pivot that places {@code k} in the smaller partition is
 * performed in the SELECT algorithm of Floyd and Rivest [4, 5].
 *
 * <p>A worst-case linear time algorithm PICK is described in Blum <i>et al</i> [6]. This uses
 * the median of medians as a partition element for selection which ensures a minimum fraction of
 * the elements are eliminated per iteration. This was extended to use an asymmetric pivot choice
 * with efficient placement of the medians sample location in the QuickselectAdpative algorithm of
 * Alexandrescu [7].
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
 * Theoretical Computer Science 347, 214-238.
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
 * <li><a href="https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm">Floyd-Rivest algorithm (Wikipedia)</a>
 * <li><a href="https://en.wikipedia.org/wiki/Median_of_medians">Median of medians (Wikipedia)</a>
 * </ol>
 *
 * @since 1.1
 */
final class Selection {
    // Implementation Notes
    //
    // Selection is performed using a quickselect variant to recursively divide the range
    // to select the target index, or indices. Partition size or recursion are monitored
    // will fall-backs on poor convergence of a linearselect (single index) or heapselect.
    //
    // Many implementations were tested, each with strengths and weaknesses on different
    // input data containing random elements, repeat elements, elements with repeat
    // patterns, and constant elements. The final implementation performs well across data
    // types for single and multiple indices with no obvious weakness.
    //
    // Single indices are selected using a quickselect adaptive method based on Alexandrescu.
    // The algorithm is a quickselect around a pivot identified using a
    // sample-of-sample-of-samples created from the entire range data. This pivot will
    // have known lower and upper margins and ensures elimination of a minimum fraction of
    // data at each step. To increase speed the pivot can be identified using less of the data
    // but without margin guarantees (sampling mode). The algorithm monitors partition sizes
    // against the known margins. If the reduction in the partition size is not large enough
    // then the algorithm can disable sampling mode and ensure linear performance by removing
    // a set fraction of the data each iteration.
    //
    // Modifications from Alexandrescu are:
    // 1. Initialise sampling mode using the Floyd-Rivest (FR) SELECT algorithm.
    // 2. Adaption is adjusted to force use of the lower margin in the far-step method when
    //    sampling is disabled.
    // 3. Change the far-step method to a min-of-4 then median-of-3 into the 2nd 12th-tile.
    //    The original method uses a lower-median-of-4, min-of-3 into the 4th 12th-tile.
    // 4. Position the sample around the target k when in sampling mode for the non-far-step
    //    methods.
    //
    // The far step method is used when target k is with 1/12 of the end of the data A.
    // The differences in the far-step method are:
    // - The upper margin when not sampling is 8/24 vs. 9/24; the lower margin remains at 1/12.
    // - The position of the sample is closer to the expected location of k < |A|/12.
    // - Sampling mode uses a median-of-3 with adaptive k, matching the other step methods.
    //   Note the original min-of-3 sample is more likely to create a pivot too small if used
    //   with adaption of k leaving k in the larger partition and a wasted iteration.
    //
    // The Floyd-Rivest (FR) SELECT algorithm is preferred for sampling over using quickselect
    // adaptive sampling. It uses a smaller sample and has improved heuristics to place the sample
    // pivot. However the FR sample is a small range of the data and pivot selection can be poor
    // if the sample is not representative. This can be mitigated by creating a random sample
    // of the entire range for the pivot selection. This implementation does not use random
    // sampling for the FR mode. Performance is identical on random data (randomisation is a
    // negligible overhead) and faster on sorted data. Any data not suitable for the FR algorithm
    // are immediately switched to the quickselect adaptive algorithm with sampling. Performance
    // across a range of data shows this strategy is approximately mid-way in performance between
    // FR with random sampling, and quickselect adaptive in sampling mode. The benefit is that
    // sorted or partially partitioned data are not intentionally unordered as the method will
    // only move elements known to be incorrectly placed in the array.
    //
    // Multiple indices are selected using a dual-pivot partition method by
    // Yaroslavskiy to divide the interval containing the indices. When indices are effectively
    // a single index the method can switch to the single index selection to use the FR algorithm.
    // Alternative schemes to partition multiple indices are to repeat call single index select
    // with cached pivots, or without cached pivots if processing indices in order as the previous
    // index brackets the range for the next search. Caching pivots is the most effective
    // alternative. It requires storing all pivots during select, and using the cache to look-up
    // the search bounds (sub-range) for each target index. This requires 2n searches for n indices.
    // All pivots must be stored to avoid destroying previously partitioned data on repeat entry
    // to the array. The current scheme inverts this by requiring at most n-1 divides of the
    // indices during recursion and has the advantage of tracking recursion depth during selection
    // for each sub-range. Division of indices is a small overhead for the common case where
    // the number of indices is far smaller than the size of the data.
    //
    // For some indices and data a full sort of the data will be faster; this is impossible to
    // predict on unknown data input and attempts to analyse the indices and data impact
    // performance for the majority of use cases where sorting is not a suitable choice.
    // Use of the sortselect finisher allows the current multiple indices method to degrade
    // to a (non-optimised) dual-pivot quicksort (see below).
    //
    // heapselect vs sortselect
    //
    // Quickselect can switch to an alternative when: the range is very small
    // (e.g. insertion sort); or the target index is close to the end (e.g. heapselect).
    // Small ranges and a target index close to the end are handled using a hybrid of insertion
    // sort and selection (sortselect). This is faster than heapselect for small distance from
    // the edge (m) for a single index and has the advantage of sorting all upstream values from
    // the target index (heapselect requires push-down of each successive value to sort). This
    // allows the dual-pivot quickselect on multiple indices that saturate the range to degrade
    // to a (non-optimised) dual-pivot quicksort. However sortselect is worst case Order(m * (r-l))
    // for range [l, r] so cannot be used when quickselect fails to converge as m may be very large.
    // Thus heapselect is used as the stopper algorithm when quickselect progress is slow on
    // multiple indices. If heapselect is used for small range handling the performance on
    // saturated indices is significantly slower. Hence the presence of two final selection
    // methods for different purposes.

    /** No instances. */
    private Selection() {}

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>All indices are assumed to be within {@code [0, a.length)}.
     *
     * <p>This method respects the ordering imposed by {@link Double#compare(double, double)}.
     * {@code -0.0} is treated as less than value {@code 0.0}; {@code Double.NaN} is
     * considered greater than any other value; and all {@code Double.NaN} values are
     * considered equal.
     *
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @throws IndexOutOfBoundsException if any index {@code k} is not within the
     * sub-range {@code [0, a.length)}
     */
    static void select(double[] a, int... k) {
        doubleSelect(0, a.length, a, k);
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>All indices are assumed to be within {@code [fromIndex, toIndex)}.
     *
     * <p>This method respects the ordering imposed by {@link Double#compare(double, double)}.
     * {@code -0.0} is treated as less than value {@code 0.0}; {@code Double.NaN} is
     * considered greater than any other value; and all {@code Double.NaN} values are
     * considered equal.
     *
     * @param fromIndex Index of the first element (inclusive)
     * @param toIndex Index of the last element (exclusive)
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @throws IndexOutOfBoundsException if the sub-range {@code [fromIndex, toIndex)} is out of
     * bounds of range {@code [0, a.length)}; or if any index {@code k} is not within the
     * sub-range {@code [fromIndex, toIndex)}
     */
    static void select(int fromIndex, int toIndex, double[] a, int... k) {
        checkFromToIndex(fromIndex, toIndex, a.length);
        doubleSelect(fromIndex, toIndex, a, k);
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>This method pre/post-processes the data and indices to respect the ordering
     * imposed by {@link Double#compare(double, double)}.
     *
     * @param fromIndex Index of the first element (inclusive)
     * @param toIndex Index of the last element (exclusive)
     * @param a Values.
     * @param k Indices (may be destructively modified).
     * @throws IndexOutOfBoundsException if any index {@code k} is not within the
     * sub-range {@code [fromIndex, toIndex)}
     */
    private static void doubleSelect(int fromIndex, int toIndex, double[] a, int[] k) {
        if (k.length == 0 || toIndex - fromIndex <= 1) {
            // If data length == 0 then any index is invalid and will raise an exception, otherwise return
            checkIndices(fromIndex, toIndex - 1, k);
            return;
        }

        // Sort NaN / count signed zeros
        int cn = 0;
        int end = toIndex;
        for (int i = end; i > fromIndex;) {
            final double v = a[--i];
            // Count negative zeros using a sign bit check
            if (Double.doubleToRawLongBits(v) == Long.MIN_VALUE) {
                cn++;
                // Change to positive zero.
                // Data must be repaired after selection.
                a[i] = 0.0;
            } else if (v != v) {
                // Move NaN to end
                a[i] = a[--end];
                a[end] = v;
            }
        }

        // Partition
        int n = 0;
        if (end - fromIndex > 1) {
            n = k.length;
            // Filter indices invalidated by NaN check
            if (end < toIndex) {
                for (int i = n; --i >= 0;) {
                    final int v = k[i];
                    if (v >= end) {
                        // Check ignored index was valid, and move to end
                        checkIndex(fromIndex, toIndex - 1, v);
                        k[i] = k[--n];
                        k[n] = v;
                    }
                }
            }
            // Return n, the count of used indices in k.
            // Use this to post-process zeros.
            n = select(a, fromIndex, end - 1, k, n);
        }

        // Restore signed zeros
        if (cn != 0) {
            // Use partition indices below zero to fast-forward to zero as much as possible
            int j = -1;
            if (n < 0) {
                // Binary search on -n sorted indices: r = (-n) - 1
                int lo = 0;
                int hi = ~n;
                while (lo <= hi) {
                    final int mid = (lo + hi) >>> 1;
                    if (a[k[mid]] < 0) {
                        j = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
            } else {
                // Unsorted, process all indices
                for (int i = n; --i >= 0;) {
                    if (a[k[i]] < 0) {
                        j = k[i];
                    }
                }
            }
            for (;;) {
                if (a[++j] == 0) {
                    a[j] = -0.0;
                    if (--cn == 0) {
                        break;
                    }
                }
            }
        }
    }

    /**
     * Partition the array such that indices {@code k} correspond to their correctly
     * sorted value in the equivalent fully sorted array.
     *
     * <p>The method assumes all {@code k} are valid indices into the data in {@code [0, length)}.
     * It assumes no NaNs or signed zeros in the data. Data must be pre- and post-processed.
     *
     * <p>The count of the number of used indices is returned. If the keys are sorted in-place,
     * the count is returned as a negative.
     *
     * @param a Values.
     * @param left Lower bound of data (inclusive).
     * @param right Upper bound of data (inclusive).
     * @param k Indices (may be destructively modified).
     * @param n Count of indices.
     * @return the count of used indices
     * @throws IndexOutOfBoundsException if any index {@code k} is not within the
     * sub-range {@code [left, right]}
     */
    private static int select(double[] a, int left, int right, int[] k, int n) {
        if (n < 1) {
            return 0;
        }
        if (n == 1) {
            checkIndex(left, right, k[0]);
            Partition.select(a, left, right, k[0], k[0]);
            return -1;
        }

        // Interval creation validates the indices are in [left, right]
        final UpdatingInterval keys = IndexIntervals.createUpdatingInterval(left, right, k, n);

        // Save number of used indices
        final int count = countIndices(keys, n);

        // Note: If the keys are not separated then they are effectively a single key.
        // Any split of keys separated by the sort select size
        // will be finished on the next iteration.
        final int k1 = keys.left();
        final int kn = keys.right();
        if (kn - k1 < Partition.SORTSELECT_SIZE) {
            Partition.select(a, left, right, k1, kn);
        } else {
            // Dual-pivot mode with small range sort length configured using index density
            Partition.select(a, left, right, keys,
                Partition.dualPivotFlags(left, right, k1, kn, n));
        }
        return count;
    }

    /**
     * Count the number of indices. Returns a negative value if the indices are sorted.
     *
     * @param keys Keys.
     * @param n Count of indices.
     * @return the count of (sorted) indices
     */
    private static int countIndices(UpdatingInterval keys, int n) {
        if (keys instanceof KeyUpdatingInterval) {
            return -((KeyUpdatingInterval) keys).size();
        }
        return n;
    }

    /**
     * Checks if the sub-range from fromIndex (inclusive) to toIndex (exclusive) is
     * within the bounds of range from 0 (inclusive) to length (exclusive).
     *
     * <p>This function provides the functionality of
     * {@code java.utils.Objects.checkFromToIndex} introduced in JDK 9. The <a
     * href="https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/util/Objects.html#checkFromToIndex(int,int,int)">Objects</a>
     * javadoc has been reproduced for reference.
     *
     * <p>The sub-range is defined to be out of bounds if any of the following
     * inequalities is true:
     * <ul>
     * <li>{@code fromIndex < 0}
     * <li>{@code fromIndex > toIndex}
     * <li>{@code toIndex > length}
     * <li>{@code length < 0}, which is implied from the former inequalities
     * </ul>
     *
     * @param fromIndex Lower-bound (inclusive) of the sub-range.
     * @param toIndex Upper-bound (exclusive) of the sub-range.
     * @param length Upper-bound (exclusive) of the range
     * @return fromIndex if the sub-range is within the bounds of the range
     * @throws IndexOutOfBoundsException if the sub-range is out of bounds
     */
    private static int checkFromToIndex(int fromIndex, int toIndex, int length) {
        // Checks as documented above
        if (fromIndex < 0 || fromIndex > toIndex || toIndex > length) {
            throw new IndexOutOfBoundsException(
                String.format("Range [%d, %d) out of bounds for length %d", fromIndex, toIndex, length));
        }
        return fromIndex;
    }

    /**
     * Checks if the {@code index} is within the range {@code [left, right]}.
     *
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param index Index.
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    static void checkIndex(int left, int right, int index) {
        if (index < left || index > right) {
            throw new IndexOutOfBoundsException(
                String.format("Index %d out of bounds for range [%d, %d]", index, left, right));
        }
    }

    /**
     * Checks if the {@code index} is within the range {@code [left, right]}.
     *
     * @param left Lower bound of data (inclusive, assumed to be strictly positive).
     * @param right Upper bound of data (inclusive, assumed to be strictly positive).
     * @param k Indices.
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    private static void checkIndices(int left, int right, int[] k) {
        for (final int i : k) {
            checkIndex(left, right, i);
        }
    }
}
