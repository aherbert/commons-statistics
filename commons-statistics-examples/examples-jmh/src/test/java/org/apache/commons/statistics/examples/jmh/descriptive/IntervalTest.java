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
import java.util.function.BiFunction;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Interval} implementations.
 */
class IntervalTest {
    @ParameterizedTest
    @CsvSource({
        "1, 1",
        "1, 2",
        "1, 3",
        "10, 42",
    })
    void testRangeInterval(int lo, int hi) {
        final Interval interval = IndexIntervals.interval(lo, hi);
        Assertions.assertEquals(lo, interval.left());
        Assertions.assertEquals(hi, interval.right());
        if (interval.left() < interval.right()) {
            Assertions.assertEquals(lo + 1, interval.updateLeft(lo));
        }
        if (interval.left() < interval.right()) {
            Assertions.assertEquals(hi - 1, interval.updateRight(hi));
        }
        if (interval.left() + 2 < interval.right()) {
            final int left = interval.left();
            final int right = interval.right();
            final int m1 = (interval.left() + interval.right()) >>> 1;
            final int m2 = m1 + 1;
            final Interval leftInterval = interval.split(m1, m2);
            Assertions.assertEquals(left, leftInterval.left());
            Assertions.assertEquals(m1 - 1, leftInterval.right());
            Assertions.assertEquals(m2 + 1, interval.left());
            Assertions.assertEquals(right, interval.right());
        }
    }

    @Test
    void testKeyIntervalInvalidIndicesThrows() {
        assertInvalidIndicesThrows(KeyInterval::of);
        // Invalid indices: not in [0, Integer.MAX_VALUE)
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> KeyInterval.of(new int[] {-1, 2, 3}, 3));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> KeyInterval.of(new int[] {1, 2, Integer.MAX_VALUE}, 3));
    }

    private static void assertInvalidIndicesThrows(BiFunction<int[], Integer, Interval> constructor) {
        // Size zero
        Assertions.assertThrows(IllegalArgumentException.class, () -> constructor.apply(new int[0], 0));
        Assertions.assertThrows(IllegalArgumentException.class, () -> constructor.apply(new int[10], 0));
        // Not sorted
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> constructor.apply(new int[] {3, 2, 1}, 3));
        // Not unique
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> constructor.apply(new int[] {1, 2, 2, 3}, 4));
    }

    @ParameterizedTest
    @MethodSource(value = {"testIndices"})
    void testUpdateKeyInterval(int[] indices) {
        assertUpdate(KeyInterval::of, indices);
    }

    @ParameterizedTest
    @MethodSource(value = {"testIndices"})
    void testUpdateIndexSetInterval(int[] indices) {
        // Skip this due to excess memory consumption
        Assumptions.assumeTrue(indices[indices.length - 1] < Integer.MAX_VALUE - 1);
        assertUpdate((k, n) -> IndexSet.of(k, n).interval(), indices);
    }

//    @ParameterizedTest
//    @MethodSource(value = {"testIndices"})
//    void testPreviousNextIndexInterval(int[] indices) {
//        assertPreviousNextIndex(IndexIntervals.create(indices, indices.length), indices);
//    }

    @ParameterizedTest
    @MethodSource(value = {"testIndices"})
    void testSplitKeyInterval(int[] indices) {
        assertSplit(KeyInterval::of, indices);
    }

    @ParameterizedTest
    @MethodSource(value = {"testIndices"})
    void testSplitIndexSetInterval(int[] indices) {
        // Skip this due to excess memory consumption
        Assumptions.assumeTrue(indices[indices.length - 1] < Integer.MAX_VALUE - 1);
        assertSplit((k, n) -> IndexSet.of(k, n).interval(), indices);
    }

    /**
     * Assert the {@link Interval#updateLeft(int)} and {@link Interval#updateRight(int)} methods.
     * These are tested by successive calls to reduce the interval by 1 index until it
     * has only 1 index remaining.
     *
     * @param constructor Interval constructor.
     * @param indices Indices.
     */
    private static void assertUpdate(BiFunction<int[], Integer, Interval> constructor,
            int[] indices) {
        Interval interval = constructor.apply(indices, indices.length);
        final int nm1 = indices.length - 1;
        Assertions.assertEquals(indices[0], interval.left());
        Assertions.assertEquals(indices[nm1], interval.right());

        // Use updateLeft to reduce the interval to length 1
        for (int i = 1; i < indices.length; i++) {
            // rounded down median between indices
            final int k = (indices[i - 1] + indices[i]) >>> 1;
            interval.updateLeft(k);
            Assertions.assertEquals(indices[i], interval.left());
        }
        Assertions.assertEquals(interval.left(), interval.right());

        // Use updateRight to reduce the interval to length 1
        interval = constructor.apply(indices, indices.length);
        for (int i = indices.length; --i > 0;) {
            // rounded up median between indices
            final int k = 1 + ((indices[i - 1] + indices[i]) >>> 1);
            interval.updateRight(k);
            Assertions.assertEquals(indices[i - 1], interval.right());
        }
        Assertions.assertEquals(interval.left(), interval.right());
    }

    /**
     * Assert the {@link Interval#split(int, int)} method.
     * These are tested by successive calls to split the interval around the mid-point.
     *
     * @param constructor Interval constructor.
     * @param indices Indices.
     */
    private static void assertSplit(BiFunction<int[], Integer, Interval> constructor, int[] indices) {
        assertSplitMedian(constructor.apply(indices, indices.length),
            indices, 0, indices.length - 1);
        assertSplitMiddleIndices(constructor.apply(indices, indices.length),
            indices, 0, indices.length - 1);
    }

    /**
     * Assert a split using the median value between the split median.
     *
     * @param interval Interval.
     * @param indices Indices.
     * @param i Low index into the indices (inclusive).
     * @param j High index into the indices (inclusive).
     */
    private static void assertSplitMedian(Interval interval, int[] indices, int i, int j) {
        if (indices[i] + 1 >= indices[j]) {
            // Cannot split - no value between the low and high points
            return;
        }
        // Find the expected split about the median
        final int m = (indices[i] + indices[j]) >>> 1;
        // Binary search finds the value or the insertion index of the value
        int hi = Arrays.binarySearch(indices, i, j + 1, m + 1);
        if (hi < 0) {
            // Use the insertion index
            hi = ~hi;
        }
        // Scan for the lower index
        int lo = hi;
        while (indices[--lo] >= m);

        final int left = interval.left();
        final int right = interval.right();
        final Interval leftInterval = interval.split(m, m);
        Assertions.assertEquals(left, leftInterval.left());
        Assertions.assertEquals(indices[lo], leftInterval.right());
        Assertions.assertEquals(indices[hi], interval.left());
        Assertions.assertEquals(right, interval.right());

        // Recurse
        assertSplitMedian(leftInterval, indices, i, lo);
        assertSplitMedian(interval, indices, hi, j);
    }

    /**
     * Assert a split using the two middle indices.
     *
     * @param interval Interval.
     * @param indices Indices.
     * @param i Low index into the indices (inclusive).
     * @param j High index into the indices (inclusive).
     */
    private static void assertSplitMiddleIndices(Interval interval, int[] indices, int i, int j) {
        if (i + 3 >= j) {
            // Cannot split - not two indices between low and high index
            return;
        }
        // Middle two indices
        final int m1 = (i + j) >>> 1;
        final int m2 = m1 + 1;

        final int left = interval.left();
        final int right = interval.right();
        final Interval leftInterval = interval.split(indices[m1], indices[m2]);
        Assertions.assertEquals(left, leftInterval.left());
        Assertions.assertEquals(indices[m1 - 1], leftInterval.right());
        Assertions.assertEquals(indices[m2 + 1], interval.left());
        Assertions.assertEquals(right, interval.right());

        // Recurse
        assertSplitMiddleIndices(leftInterval, indices, i, m1 - 1);
        assertSplitMiddleIndices(interval, indices, m2 + 1, j);
    }

    static Stream<int[]> testIndices() {
        return IndexIntervalTest.testPreviousNextIndex();
    }

//    @Test
//    void testIndexIntervalCreate() {
//        // The above tests verify the IndexInterval implementations all work.
//        // Hit all paths in the key analysis performed to create an interval.
//
//        // Small number of keys; no analysis
//        Assertions.assertEquals(ScanningKeyIndexInterval.class,
//            IndexIntervals.create(new int[] {1}, 1).getClass());
//
//        // >10 keys for key analysis
//
//        // Small number of keys saturating the range
//        Assertions.assertEquals(IndexSet.class,
//            IndexIntervals.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 11).getClass());
//        // Keys over a huge range
//        Assertions.assertEquals(ScanningKeyIndexInterval.class,
//            IndexIntervals.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Integer.MAX_VALUE - 1}, 11).getClass());
//
//        // Small number of keys over a moderate range
//        int[] k = IntStream.range(0, 30).map(i -> i * 64) .toArray();
//        Assertions.assertEquals(IndexSet.class,
//            IndexIntervals.create(k.clone(), k.length).getClass());
//        // Same keys over a huge range
//        k[k.length - 1] = Integer.MAX_VALUE - 1;
//        Assertions.assertEquals(ScanningKeyIndexInterval.class,
//            IndexIntervals.create(k, k.length).getClass());
//
//        // Moderate number of keys over a moderate range
//        k = IntStream.range(0, 3000).map(i -> i * 64) .toArray();
//        Assertions.assertEquals(IndexSet.class,
//            IndexIntervals.create(k.clone(), k.length).getClass());
//        // Same keys over a huge range - switch to binary search on the keys
//        k[k.length - 1] = Integer.MAX_VALUE - 1;
//        Assertions.assertEquals(BinarySearchKeyIndexInterval.class,
//            IndexIntervals.create(k, k.length).getClass());
//    }
}
