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
import java.util.Comparator;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Test for {@link RangePredicates}.
 */
class RangePredicatesTest {
    @Test
    void testAlways() {
        final IntIntBiPredicate r = RangePredicates.anyRange();
        Assertions.assertSame(r, RangePredicates.anyRange());
        int[] shift = {-1, 0, 1};
        for (int i : shift) {
            for (int j : shift) {
                Assertions.assertTrue(r.test(i, j));
            }
        }
    }

    @Test
    void testOfRangeThrows() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> RangePredicates.ofRange(1, 0));
    }

    @ParameterizedTest
    // Values must be valid indices into an array
    @ValueSource(ints = {0, 1, Integer.MAX_VALUE - 1})
    void testOfIndexK(int k) {
        final IntIntBiPredicate r = RangePredicates.ofIndex(k);
        assertRanges(r, k, k);
    }

    @ParameterizedTest
    @MethodSource
    void testOfIndexK1K2(int k1, int k2) {
        final IntIntBiPredicate r = RangePredicates.ofIndex(k1, k2);
        assertRanges(r, k1, k1, k2, k2);
    }

    @ParameterizedTest
    @MethodSource
    void testOfIndexK1K2K3(int k1, int k2, int k3) {
        final IntIntBiPredicate r = RangePredicates.ofIndex(k1, k2, k3);
        assertRanges(r, k1, k1, k2, k2, k3, k3);
    }

    @ParameterizedTest
    @MethodSource
    void testOfRangeK1K2(int k1, int k2) {
        final IntIntBiPredicate r = RangePredicates.ofRange(k1, k2);
        assertRanges(r, k1, k2);
    }

    /**
     * Assert the predicate is true for ranges that include the specified indices. Indices
     * are paired to make ranges.
     *
     * @param r Predicate.
     * @param k Indices.
     */
    static void assertRanges(IntIntBiPredicate r, int... k) {
        Assertions.assertEquals(0, k.length & 0x1, "Must be even");
        int n = k.length;
        int k1 = Arrays.stream(k).min().getAsInt();
        int kn = Arrays.stream(k).max().getAsInt();

        // Cover the entire range
        assertRange(false, r, k1 - 1, k1 - 1);
        assertRange(false, r, kn + 1, kn + 1);
        assertRange(true, r, k1 - 1, kn + 1);
        assertRange(true, r, k1 - 1, k1);
        assertRange(true, r, kn, kn + 1);

        // Sort ranges
        int[][] ranges = new int[n >> 1][];
        for (int i = 0; i < n; i += 2) {
            ranges[i >> 1] = new int[] {k[i], k[i + 1]};
        }
        Arrays.sort(ranges, Comparator.comparingInt(x -> x[0]));

        // Test windows that overlap multiple ranges
        for (int i = 0; i < ranges.length; i++) {
            int ka = ranges[i][0];
            int kb = ranges[i][1];
            Assertions.assertTrue(ka <= kb, "Invalid range");
            // Test range and window around range
            assertRange(true, r, ka, kb);
            assertRange(true, r, ka - 1, kb + 1);
            // Invalid range
            // assertRange(false, r, ka, ka - 1);
            // Test window inside the range
            for (int kc = ka + 1; kc < kb - 1; kc++) {
                assertRange(true, r, kc, kc);
            }
            if (i == ranges.length - 1) {
                break;
            }
            int kc = ranges[i + 1][0];
            int kd = ranges[i + 1][1];
            // Test window around adjacent ranges
            assertRange(true, r, ka, kd);
            assertRange(true, r, ka - 1, kd + 1);
            // Test window between adjacent ranges
            for (int ke = kb + 1; ke < kc - 1; ke++) {
                assertRange(false, r, ke, ke);
            }
        }
    }

    static void assertRange(boolean expected, IntIntBiPredicate r, int left, int right) {
        Assertions.assertEquals(expected, r.test(left, right), () -> left + " " + right);
    }

    static Stream<Arguments> testOfIndexK1K2() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(1, 10));
        builder.add(Arguments.of(1, 2));
        builder.add(Arguments.of(10, 2));
        builder.add(Arguments.of(3, 3));
        return builder.build();
    }

    static Stream<Arguments> testOfIndexK1K2K3() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(1, 10, 15));
        builder.add(Arguments.of(1, 2, 3));
        builder.add(Arguments.of(11, 2, 7));
        builder.add(Arguments.of(11, 7, 7));
        builder.add(Arguments.of(7, 7, 7));
        return builder.build();
    }

    static Stream<Arguments> testOfRangeK1K2() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(1, 10));
        builder.add(Arguments.of(1, 2));
        return builder.build();
    }

    @Test
    void test() {
        assertRange(true, RangePredicates.ofRange(1, 10), 2, 2);
    }
}
