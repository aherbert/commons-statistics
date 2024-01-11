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
 * Support for testing ranges.
 *
 * <p>The predicates returned by this class will return true if a
 * range {@code [left, right]} contain an index of interest.
 *
 * <p>These predicates can be used to control the sort branch decisions
 * of a partition algorithm.
 *
 * <p>Specialisations are provided for low number of indices or ranges of indices.
 * A high number of indices requires use of a data structure to store indices for efficient
 * range look up.
 *
 * @since 1.1
 */
final class RangePredicates {
    /** Singleton to test that the range is always of interest.
     * Can be used as a marker than partitioning is too complex. */
    private static final IntIntBiPredicate ANY_RANGE = (left, right) -> true;

    /** No instances. */
    private RangePredicates() {}

    /**
     * Returns a range predicate that is always true for any range.
     * Note: This includes invalid ranges {@code left > right}.
     *
     * @return the predicate
     */
    static IntIntBiPredicate anyRange() {
        return ANY_RANGE;
    }

    /**
     * Creates a range predicate for index {@code k1}.
     *
     * <p>{@code true} if:
     * <ul>
     * <li>{@code left <= k1 <= right}
     * </ul>
     *
     * @param k Index.
     * @return the range predicate
     */
    static IntIntBiPredicate ofIndex(int k) {
        return (left, right) -> left <= k && k <= right;
    }

    /**
     * Creates a range predicate for indices {@code k1, k2}.
     * This method handles duplicate indices; indices can be in any order.
     *
     * <p>{@code true} if:
     * <ul>
     * <li>{@code left <= k1 <= right}; or
     * <li>{@code left <= k2 <= right}
     * </ul>
     *
     * @param k1 Index.
     * @param k2 Index.
     * @return the range predicate
     */
    static IntIntBiPredicate ofIndex(int k1, int k2) {
        // Eliminate duplicates
        if (k1 == k2) {
            return ofIndex(k1);
        }
        // Sort
        final int i1 = k1 < k2 ? k1 : k2;
        final int i2 = k1 < k2 ? k2 : k1;
        if (i1 + 1 == i2) {
            return ofRange(i1, i2);
        }
        // Sorted order allows a short-circuiting predicate
        // Find first index above left and test it is below right
        return (left, right) -> {
            if (left <= i1) {
                return i1 <= right;
            }
            return left <= i2 && i2 <= right;
        };
    }

    /**
     * Creates a range predicate for indices {@code k1, k2, k3}.
     * This method handles duplicate indices; indices can be in any order.
     *
     * <p>{@code true} if:
     * <ul>
     * <li>{@code left <= k1 <= right}; or
     * <li>{@code left <= k2 <= right}; or
     * <li>{@code left <= k3 <= right}
     * </ul>
     *
     * @param k1 Index.
     * @param k2 Index.
     * @param k3 Index.
     * @return the range predicate
     */
    static IntIntBiPredicate ofIndex(int k1, int k2, int k3) {
        // Eliminate duplicates
        if (k1 == k2) {
            return ofIndex(k1, k3);
        }
        if (k2 == k3) {
            return ofIndex(k1, k2);
        }
        // Sort
        int a = k1 < k2 ? k1 : k2;
        int b = k1 < k2 ? k2 : k1;
        int c = k3;
        if (k3 < b) {
            c = b;
            b = k3;
            if (k3 < a) {
                b = a;
                a = k3;
            }
        }
        if (a + 2 == c) {
            return ofRange(a, c);
        }
        // Sorted order allows a short-circuiting predicate
        // Find first index above left and test it is below right
        final int i1 = a;
        final int i2 = b;
        final int i3 = c;
        return (left, right) -> {
            if (left <= i1) {
                return i1 <= right;
            }
            if (left <= i2) {
                return i2 <= right;
            }
            return left <= i3 && i3 <= right;
        };
    }

    /**
     * Creates a range predicate for a the range {@code [k1, k2]}.
     * Requires the indices to be strictly ordered.
     *
     * <p>{@code true} if {@code [left, right]} overlaps {@code [k1, k2]}.
     *
     * @param k1 Index.
     * @param k2 Index.
     * @return the range predicate
     * @throws IllegalArgumentException if {@code k2 <= k1}
     */
    static IntIntBiPredicate ofRange(int k1, int k2) {
        if (k2 <= k1) {
            // Signal error. The alternative is to return an always false predicate.
            throw new IllegalArgumentException("Invalid range");
        }
        // Overlap:
        // L-----------R
        // ......... K1---------K2
        //
        // Gap:
        // L-----------R
        // .............. K1---------K2
        return (left, right) -> {
            if (k1 <= left) {
                return k2 >= left;
            }
            return k1 <= right;
        };
    }

    /**
     * Creates a range predicate for indices {@code k}.
     * This method handles duplicate indices; indices can be in any order.
     *
     * <p>{@code true} if:
     * <ul>
     * <li>{@code left <= k <= right}; for any {@code k}
     * </ul>
     *
     * <p>If a minimum separation is provided as {@code minSeparation > 0} then
     * any point {@code i} in the interval {@code k[n] < i < k[n+1]} is included
     * in the test if {@code k} is ordered and {@code k[n+1] - k[n] <= minSeparation}.
     * For example:
     * <pre>
     *   k   [0, 3, 5]
     *   separation   test indices
     *   0            [0, 3, 5]
     *   1            [0, 3, 4, 5]
     *   5            [0, 1, 2, 3, 4, 5]
     * </pre>
     *
     * <p>A saturated range will return {@link #anyRange()}.
     *
     * @param minSeparation Minimum separation.
     * @param k Indices.
     * @return the range predicate
     */
    static IntIntBiPredicate ofIndex(int minSeparation, int[] k) {
        // TODO
        // Purify the indices and store in an OffsetIndexSet
        // Find the min
        // Find the max
        return ANY_RANGE;
    }

    // TODO
    // More specialisations for indices up to 5 ???
    // Allow or of predicates to collect them together.
    // Multiple indices using an offset index set using min as the offset.
}
