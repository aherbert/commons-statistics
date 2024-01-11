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
import java.util.BitSet;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link IndexSet}.
 */
class IndexSetTest {

    @Test
    void testInvalidRangeThrows() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> IndexSet.ofRange(-1, 3));
        Assertions.assertThrows(IllegalArgumentException.class, () -> IndexSet.ofRange(0, -1));
        Assertions.assertThrows(IllegalArgumentException.class, () -> IndexSet.ofRange(456, 123));
        Assertions.assertThrows(IllegalArgumentException.class, () -> IndexSet.of(new int[0]));
    }

    @Test
    void testInvalidPivotCacheRangeThrows() {
        final int left = 10;
        final int right = 156;
        final IndexSet set = IndexSet.ofRange(left, right);
        Assertions.assertThrows(IllegalArgumentException.class, () -> set.asScanningPivotCache(left - 1, right));
        Assertions.assertThrows(IllegalArgumentException.class, () -> set.asScanningPivotCache(right, left));
        Assertions.assertDoesNotThrow(() -> set.asScanningPivotCache(left, right));
        Assertions.assertDoesNotThrow(() -> set.asScanningPivotCache(left + 1, right - 1));
        // We must know the capacity (the highest bit that can be stored).
        final int noOfLongs = 1 + (right - left) / Long.SIZE;
        final int highBit = left + noOfLongs * Long.SIZE - 1;
        // Show this is correct
        set.add(highBit);
        final int capacity = set.nextClearBit(highBit);
        Assertions.assertEquals(highBit + 1, capacity);
        Assertions.assertDoesNotThrow(() -> set.asScanningPivotCache(left, highBit));
        Assertions.assertThrows(IllegalArgumentException.class, () -> set.asScanningPivotCache(left, capacity));
    }

    @ParameterizedTest
    @MethodSource
    void testGetSet(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        final BitSet ref = new BitSet(n);
        for (final int i : indices) {
            Assertions.assertEquals(ref.get(i), set.get(i), () -> String.valueOf(i));
            set.add(i);
            ref.set(i);
            Assertions.assertTrue(set.get(i));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testSetRange1(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        final BitSet ref = new BitSet(n);
        for (final int i : indices) {
            Assertions.assertEquals(ref.get(i), set.get(i));
            // inclusive end
            set.add(i, i);
            ref.set(i);
            Assertions.assertTrue(set.get(i));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testSetRange(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        Arrays.sort(indices);
        for (int i = 1; i < indices.length; i++) {
            final int from = indices[i - 1];
            final int to = indices[i];
            // inclusive end so skip duplicates
            if (from == to) {
                continue;
            }
            for (int j = from; j < to; j++) {
                Assertions.assertFalse(set.get(j));
            }
            set.add(from, to - 1);
            for (int j = from; j < to; j++) {
                Assertions.assertTrue(set.get(j));
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testPreviousNextSetBit(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        final BitSet ref = new BitSet(n);
        Arrays.sort(indices);
        Assertions.assertEquals(-1, set.previousSetBit(0));
        Assertions.assertEquals(-1, set.nextSetBit(0));
        final int highBit = indices[indices.length - 1];
        Assertions.assertEquals(-1, set.previousSetBit(highBit));
        Assertions.assertEquals(-1, set.nextSetBit(highBit));
        for (int i = 1; i < indices.length; i++) {
            final int from = indices[i - 1];
            final int to = indices[i];
            final int middle = (from + to) >>> 1;
            Assertions.assertEquals(ref.previousSetBit(middle), set.previousSetBit(middle));
            Assertions.assertEquals(ref.nextSetBit(middle), set.nextSetBit(middle));
            set.add(from);
            set.add(to);
            ref.set(from);
            ref.set(to);
            Assertions.assertEquals(ref.previousSetBit(middle), set.previousSetBit(middle));
            Assertions.assertEquals(ref.nextSetBit(middle), set.nextSetBit(middle));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testPreviousNextClearBit(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        final BitSet ref = new BitSet(n);
        Arrays.sort(indices);
        // Note: Different from using a BitSet. The IndexSet does not support
        // clear bits outside of the range used to construct it.
        final int highBit = indices[indices.length - 1];

        // When empty this should return the start index. Only call with indices <= right.
        Assertions.assertEquals(0, set.nextClearBit(0));
        Assertions.assertEquals(indices[0] - 1, set.nextClearBit(indices[0] - 1));
        Assertions.assertEquals(indices[0], set.nextClearBit(indices[0]));
        Assertions.assertEquals(highBit, set.nextClearBit(highBit));
        Assertions.assertEquals(0, set.previousClearBit(0));
        Assertions.assertEquals(indices[0] - 1, set.previousClearBit(indices[0] - 1));
        Assertions.assertEquals(indices[0], set.previousClearBit(indices[0]));
        Assertions.assertEquals(highBit, set.previousClearBit(highBit));

        for (int i = 1; i < indices.length; i++) {
            final int from = indices[i - 1];
            final int to = indices[i];
            final int middle = (from + to) >>> 1;
            Assertions.assertEquals(ref.nextClearBit(middle), set.nextClearBit(middle));
            Assertions.assertEquals(ref.previousClearBit(middle), set.previousClearBit(middle));
            // inclusive end
            set.add(from, to);
            ref.set(from, to + 1);
            Assertions.assertEquals(ref.nextClearBit(middle), set.nextClearBit(middle));
            Assertions.assertEquals(ref.previousClearBit(middle), set.previousClearBit(middle));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testForEach(int[] indices, int n) {
        final IndexSet set = createIndexSet(indices);
        final BitSet ref = new BitSet(n);
        Arrays.stream(indices).forEach(i -> {
            set.add(i);
            ref.set(i);
        });
        final int[] e = ref.stream().toArray();
        // Check the output is the same
        final int[] a = new int[e.length];
        final int[] c = {0};
        set.forEach(i -> a[c[0]++] = i);
        Assertions.assertArrayEquals(e, a);
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testOfIndices(int[] indices) {
        assertOfIndices(indices, -1);
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testOfIndicesTruncated(int[] indices) {
        Assumptions.assumeTrue(indices.length > 1);
        final int n = ThreadLocalRandom.current().nextInt(1, indices.length);
        assertOfIndices(indices, n);
    }

    private static void assertOfIndices(int[] indices, int n) {
        final IndexSet set = n < 0 ? IndexSet.of(indices) : IndexSet.of(indices, n);
        final BitSet ref = new BitSet();
        final int upper = n < 0 ? indices.length : n;
        for (int i = 0; i < upper; i++) {
            ref.set(indices[i]);
            Assertions.assertTrue(set.get(indices[i]));
        }
        final int[] e = ref.stream().toArray();
        final int[] a = new int[e.length];
        final int[] c = {0};
        set.forEach(i -> a[c[0]++] = i);
        Assertions.assertArrayEquals(e, a);
    }

    static Stream<Arguments> testGetSet() {
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        final Stream.Builder<Arguments> builder = Stream.builder();
        for (final int size : new int[] {5, 500}) {
            final int[] a = rng.ints(10, 0, size).toArray();
            builder.accept(Arguments.of(a.clone(), size));
            // Force use of index 0
            a[0] = 0;
            builder.accept(Arguments.of(a, size));
        }
        // Large offset with an index at the end of the range
        final int n = 513;
        // Use 1, 2, or 3 longs for storage
        builder.accept(Arguments.of(new int[] {n - 13, n - 1}, n));
        builder.accept(Arguments.of(new int[] {n - 78, n - 1}, n));
        builder.accept(Arguments.of(new int[] {n - 137, n - 1}, n));
        // Uses a capacity of 512. BitSet will increase this to 512 + 64 to store index
        // 513.
        builder.accept(Arguments.of(new int[] {1, n - 1}, n));
        return builder.build();
    }

    private IndexSet createIndexSet(int[] indices) {
        final int min = Arrays.stream(indices).min().getAsInt();
        final int max = Arrays.stream(indices).max().getAsInt();
        return IndexSet.ofRange(min, max);
    }
}
