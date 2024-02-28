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
 * A searchable interval that contains indices used for partitioning an array into multiple regions.
 *
 * <p>The interval provides the following functionality:
 *
 * <ul>
 * <li>Return the supported bounds of the search {@code [left <= right]}.
 * <li>Return the previous index contained in the interval from a search point {@code k}.
 * <li>Return the next index contained in the interval from a search point {@code k}.
 * </ul>
 *
 * <p>Note that the interval provides the supported bounds. If a search begins outside
 * the supported bounds the result is undefined.
 *
 * <p>Implementations may assume indices are positive.
 *
 * @since 1.1
 */
interface IndexInterval {

    /**
     * The start (inclusive) of the range of indices supported.
     *
     * @return start of the supported range
     */
    int left();

    /**
     * The end (inclusive) of the range of indices supported.
     *
     * @return end of the supported range
     */
    int right();

    /**
     * Returns the nearest index that occurs on or before the specified starting
     * index.
     *
     * <p>If {@code k < left} or {@code k > right} the result is undefined.
     *
     * @param k Index to start checking from (inclusive).
     * @return the previous index
     */
    int previousIndex(int k);

    /**
     * Returns the nearest index that occurs on or after the specified starting
     * index.
     *
     * <p>If {@code k < left} or {@code k > right} the result is undefined.
     *
     * @param k Index to start checking from (inclusive).
     * @return the next index
     */
    int nextIndex(int k);

    /**
     * Returns the nearest free-index that occurs on or after the specified starting
     * index. This is an index that is not within the interval {@code [left, right]}.
     *
     * <p>If {@code k < left} or {@code k > right} the result is undefined.
     * A free index outside the supported interval have special meaning:
     *
     * <ul>
     * <li>If the next free index is {@code > right} this indicates that the
     * closed interval {@code [left, right]} is saturated.
     * <li>If the next free index is {@code < left} this indicates that the
     * identification of a free index is not supported. Implementations may return -1
     * for this case.
     * </ul>
     *
     * <p>The default implementation advances using {@link #nextIndex(int)} until
     * there is a gap in the indices (i.e. a free index):
     *
     * <pre>{@code
     * int n = k - 1;
     * while (++n < right()) {
     *     if (nextIndex(n) > n) {
     *         // n is a free-index
     *         return n;
     *     }
     * }
     * return right() + 1;
     * }</pre>
     *
     * <p>Implementations may support this behaviour with a more efficient implementation;
     * or return -1.
     *
     * @param k Index to start checking from (inclusive).
     * @return the next free index
     */
    // TODO - is this useful ? An unsorted point has to be maintained at each split 
    default int nextFreeIndex(int k) {
        // Implement using nextIndex. Assumes left <= k <= right.
        final int r = right();
        for (int n = k - 1; ++n < r;) {
            if (nextIndex(n) > n) {
                // next index is after n: n is a free index
                return n;
            }
        }
        return r + 1;
    }

    /**
     * Test if the interval {@code [ka, kb]} is saturated (there are no free indices
     * within the range).
     *
     * @param ka First index.
     * @param kb Last index.
     * @return true if saturated
     */
    // TODO - is this useful ? It could be supported using a compressed index set where
    // compression is set relative to the data length.
    default boolean saturated(int ka, int kb) {
        return false;
    }

    /**
     * Returns the nearest index that occurs after the specified split
     * index, and the nearest index that occurs before the specified split index.
     *
     * <p>Note: Requires {@code left <= k < right}.
     * Used to split an interval with the upper side known to be within {@code [left, right]};
     * it is allowed to return lower below the {@code left} bound if {@code k == left}.
     *
     * <pre>{@code
     * l-----------k-----------r
     *              |--> upper
     *   lower <--|
     *
     * upper > k
     * lower < k
     * }</pre>
     *
     *
     * <p>The default implementation uses:
     *
     * <pre>{@code
     * lower = k == left() ? k - 1 : previousIndex(k - 1);
     * upper = nextIndex(k + 1);
     * }</pre>
     *
     * <p>Implementations may override this method if both indices can be obtained together.
     *
     * <p>If {@code k < left} or {@code k >= right} the result is undefined.
     *
     * @param k Split index in {@code [left, right)}
     * @param lower Lower index.
     * @return the upper index
     */
    default int splitUpper(int k, int[] lower) {
        lower[0] = k == left() ? k - 1 : previousIndex(k - 1);
        return nextIndex(k + 1);
    }

    /**
     * Returns the nearest index that occurs before the specified split
     * index, and the nearest index that occurs after the specified split index.
     *
     * <p>Note: Requires {@code left < k <= right}.
     * Used to split an interval with the lower side known to be within {@code [left, right]};
     * it is allowed to return upper above the {@code right} bound if {@code k == right}.
     *
     * <pre>{@code
     * l-----------k-----------r
     *   lower <--|
     *              |--> upper
     *
     * lower < k
     * upper > k
     * }</pre>
     *
     * <p>The default implementation uses:
     *
     * <pre>{@code
     * upper = k == right() ? k + 1 : nextIndex(k + 1);
     * lower = previousIndex(k - 1);
     * }</pre>
     *
     * <p>Implementations may override this method if both indices can be obtained together.
     *
     * <p>If {@code k <= left} or {@code k > right} the result is undefined.
     *
     * @param k Split index {@code (left, right]}
     * @param upper Upper index.
     * @return the lower index
     */
    default int splitLower(int k, int[] upper) {
        upper[0] = k == right() ? k + 1 : nextIndex(k + 1);
        return previousIndex(k - 1);
    }
}
