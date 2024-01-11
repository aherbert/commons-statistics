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
 * Support for analysis of keys {@code k} to be partitioned
 * in a range of an array.
 *
 * @since 1.1
 */
final class KeyAnalysis {
    /** No instances. */
    private KeyAnalysis() {}

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
     * <p>A saturated range will return {@link RangePredicates#anyRange()}.
     *
     * @param minSeparation Minimum separation.
     * @param k Indices.
     * @return the range predicate
     */
    static IntIntBiPredicate evaluate(int minSeparation, int[] k) {
        // TODO
        // Purify the indices and store in an OffsetIndexSet
        // Find the min
        // Find the max
        return null;
    }

    // TODO
    // Sort indices up to length 10. Collect ranges and individual points
    // based on minSeparation and return using RangePredicates.
    // Otherwise do a key analysis
    // using a PivotSet (Multiple indices using an offset index set using
    // min as the offset).
    //
    // KeyAnalysis should return the predicate for search recursion and
    // a consumer to store the sorted regions.
}
