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
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link CompressedIndexSet}.
 */
class CompressedIndexSetTest {
    /** Compression levels to test. */
    private static final int[] COMPRESSION = {1, 2, 3};

    @Test
    void testInvalidRangeThrows() {
        // Valid compression
        final int c = 1;
        Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.ofRange(c, -1, 3));
        Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.ofRange(c, 0, -1));
        Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.ofRange(c, 456, 123));
        Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.of(c, new int[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.of(c, new int[] {-1}));
    }

    @Test
    void testInvalidCompressionThrows() {
        for (final int c : new int[] {0, -1, 32}) {
            Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.ofRange(c, 1, 3));
            Assertions.assertThrows(IllegalArgumentException.class, () -> CompressedIndexSet.of(c, new int[] {1}));
        }
    }

    @ParameterizedTest
    @MethodSource
    void testGetSet(int[] indices, int n) {
        for (final int c : COMPRESSION) {
            final CompressedIndexSet set = createCompressedIndexSet(c, indices);
            final BitSet ref = new BitSet(n);
            final int left = set.left();
            final int range = 1 << c;
            for (final int i : indices) {
                // The contains value is a probability due to the compression.
                // It will be true if any of the indices in the compressed range are set.
                final int mapped = getCompressedIndexLow(c, left, i);
                boolean contains = ref.get(mapped);
                for (int j = 1; !contains && j < range; j++) {
                    contains = ref.get(mapped + j);
                }
                Assertions.assertEquals(contains, set.get(i), () -> String.valueOf(i));
                set.set(i);
                ref.set(i);
                Assertions.assertTrue(set.get(i));
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testPreviousIndex(int[] indices, int n) {
        for (final int c : COMPRESSION) {
            final CompressedIndexSet set = createCompressedIndexSet(c, indices);
            final BitSet ref = new BitSet(n);
            final int left = set.left();
            final int right = set.right();
            final int range = 1 << c;
            Arrays.sort(indices);
            final int highBit = indices[indices.length - 1];
            Assertions.assertEquals(set.left() - 1, set.previousIndex(0));
            Assertions.assertEquals(set.left() - 1, set.previousIndex(highBit));
            Assertions.assertEquals(set.left() - 1, set.previousIndex(highBit * 2));
            for (final int i : indices) {
                final int lo = getCompressedIndexLow(c, left, i);
                final int hi = getCompressedIndexHigh(c, left, right, i);
                boolean contains = ref.get(lo);
                for (int j = lo + 1; !contains && j <= hi; j++) {
                    contains = ref.get(j);
                }
                if (contains) {
                    for (int j = lo; j <= hi; j++) {
                        Assertions.assertEquals(j, set.previousIndex(j), () -> "contains: " + i);
                    }
                } else {
                    int prev = ref.previousSetBit(lo);
                    if (prev < 0) {
                        prev = left - 1;
                    } else {
                        prev = getCompressedIndexHigh(c, left, right, prev);
                    }
                    Assertions.assertEquals(prev, set.previousIndex(i), () -> "previous upper: " + i);
                }
                set.set(i);
                ref.set(i);
                // Re-check within
                for (int j = lo; j <= hi; j++) {
                    Assertions.assertEquals(j, set.previousIndex(j), () -> "within: " + i);
                }
                // Check after
                Assertions.assertEquals(hi, set.previousIndex(hi + 1), () -> "after: " + i);
                Assertions.assertEquals(hi, set.previousIndex(hi + 42), () -> "after: " + i);
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testNextIndex(int[] indices, int n) {
        for (final int c : COMPRESSION) {
            final CompressedIndexSet set = createCompressedIndexSet(c, indices);
            final BitSet ref = new BitSet(n);
            final int left = set.left();
            final int right = set.right();
            final int range = 1 << c;
            Arrays.sort(indices);
            final int highBit = indices[indices.length - 1];
            Assertions.assertEquals(set.right() + 1, set.nextIndex(0));
            Assertions.assertEquals(set.right() + 1, set.nextIndex(highBit));
            Assertions.assertEquals(set.right() + 1, set.nextIndex(highBit * 2));
            // Process in descending order
            for (int i = -1, j = indices.length; ++i < --j;) {
                final int k = indices[i];
                indices[i] = indices[j];
                indices[j] = k;
            }
            for (final int i : indices) {
                final int lo = getCompressedIndexLow(c, left, i);
                final int hi = getCompressedIndexHigh(c, left, right, i);
                boolean contains = ref.get(lo);
                for (int j = lo + 1; !contains && j <= hi; j++) {
                    contains = ref.get(j);
                }
                if (contains) {
                    for (int j = lo; j <= hi; j++) {
                        Assertions.assertEquals(j, set.nextIndex(j), () -> "contains: " + i);
                    }
                } else {
                    int next = ref.nextSetBit(lo);
                    if (next < 0) {
                        next = right + 1;
                    } else {
                        next = getCompressedIndexLow(c, left, next);
                    }
                    Assertions.assertEquals(next, set.nextIndex(i), () -> "next upper: " + i);
                }
                set.set(i);
                ref.set(i);
                // Re-check within
                for (int j = lo; j <= hi; j++) {
                    Assertions.assertEquals(j, set.nextIndex(j), () -> "within: " + i);
                }
                // Check before
                Assertions.assertEquals(lo, set.nextIndex(lo - 1), () -> "before: " + i);
                Assertions.assertEquals(lo, set.nextIndex(lo - 42), () -> "before: " + i);
            }
        }
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

    /**
     * Creates the compressed index set using the min/max of the indices.
     *
     * @param compression Compression level.
     * @param indices Indices.
     * @return the set
     */
    private static CompressedIndexSet createCompressedIndexSet(int compression, int[] indices) {
        final int min = Arrays.stream(indices).min().getAsInt();
        final int max = Arrays.stream(indices).max().getAsInt();
        final CompressedIndexSet set = CompressedIndexSet.ofRange(compression, min, max);
        Assertions.assertEquals(min, set.left());
        Assertions.assertEquals(max, set.right());
        return set;
    }

    /**
     * Gets the lower bound of the index range covered by the compressed index.
     * A compressed index covers a range of real indices. For example the
     * indices i, j, k, and l are all represented by the same compressed index
     * with a compression level of 2.
     * <pre>
     * -------ijkl------
     * </pre>
     *
     * @param c Compression.
     * @param left Lower bound of the set of compressed indices.
     * @param i Index.
     * @return the lower bound of the index range
     */
    private static int getCompressedIndexLow(int c, int left, int i) {
        return (((i - left) >>> c) << c) + left;
    }

    /**
     * Gets the lower bound of the index range covered by the compressed index.
     * A compressed index covers a range of real indices. For example the
     * indices i, j, k, and l are all represented by the same compressed index
     * with a compression level of 2.
     * <pre>
     *          right
     *          |
     * -------ijkl------
     * </pre>
     *
     * <p>This method returns l, unless clipped by the upper bound of the supported indices (right).
     *
     * @param c Compression.
     * @param left Lower bound of the set of compressed indices.
     * @param right Upper bound of the set of compressed indices.
     * @param i Index.
     * @return the upper bound of the index range
     */
    private static int getCompressedIndexHigh(int c, int left, int right, int i) {
        return Math.min(right, (((i - left) >>> c) << c) + left + (1 << c) - 1);
    }
}
