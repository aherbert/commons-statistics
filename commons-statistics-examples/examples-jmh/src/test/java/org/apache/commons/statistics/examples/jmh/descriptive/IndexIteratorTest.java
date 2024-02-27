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
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Test for {@link IndexIterator} implementations.
 */
class IndexIteratorTest {
    @ParameterizedTest
    @ValueSource(ints = {0, 1, 42, Integer.MAX_VALUE - 1})
    void testSingleIndex(int k) {
        final IndexIterator iterator = IndexIterators.ofIndex(k);
        Assertions.assertEquals(k,  iterator.left());
        Assertions.assertEquals(k,  iterator.right());
        Assertions.assertEquals(k,  iterator.end());
        Assertions.assertFalse(iterator.next());
        Assertions.assertFalse(iterator.positionAfter(k + 1));
        Assertions.assertFalse(iterator.positionAfter(k));
        Assertions.assertTrue(iterator.positionAfter(k - 1));
        Assertions.assertEquals(k,  iterator.left());
        Assertions.assertEquals(k,  iterator.right());
        Assertions.assertEquals(k,  iterator.end());
    }

    @ParameterizedTest
    @CsvSource({
        "0, 0",
        "10, 0",
        "0, 10",
        "5615236, 1263818376",
    })
    void testSingleInterval(int l, int r) {
        if (r < l) {
            final int t = l;
            l = r;
            r = t;
        }
        final IndexIterator iterator = IndexIterators.ofInterval(l, r);
        Assertions.assertEquals(l,  iterator.left());
        Assertions.assertEquals(r,  iterator.right());
        Assertions.assertEquals(r,  iterator.end());
        Assertions.assertFalse(iterator.next());
        Assertions.assertFalse(iterator.positionAfter(r + 1));
        Assertions.assertFalse(iterator.positionAfter(r));
        Assertions.assertTrue(iterator.positionAfter(r - 1));
        Assertions.assertEquals(r > l, iterator.positionAfter(l));
        Assertions.assertEquals(r > l, iterator.positionAfter((l + r) >>> 1));
        Assertions.assertEquals(l,  iterator.left());
        Assertions.assertEquals(r,  iterator.right());
        Assertions.assertEquals(r,  iterator.end());
    }

    @Test
    void testKeyIndexIteratorInvalidIndicesThrows() {
        assertInvalidIndicesThrows(KeyIndexIterator::of);
        // Invalid indices: not in [0, Integer.MAX_VALUE)
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> KeyIndexIterator.of(new int[] {-1, 2, 3}, 3));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> KeyIndexIterator.of(new int[] {1, 2, Integer.MAX_VALUE}, 3));
    }

    private static void assertInvalidIndicesThrows(BiFunction<int[], Integer, IndexIterator> constructor) {
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
    @MethodSource(value = {"testNextIndex"})
    void testNextKeyIndexIterator(int[] indices) {
        assertNext(KeyIndexIterator::of, indices, false);
    }

    private static void assertNext(BiFunction<int[], Integer, IndexIterator> constructor,
        int[] indices, boolean sparse) {
        IndexIterator iterator = constructor.apply(indices, indices.length);
        final int nm1 = indices.length - 1;
        Assertions.assertEquals(indices[0], iterator.left());
        Assertions.assertEquals(indices[nm1], iterator.end());
        // Check invariants
        Assertions.assertTrue(iterator.left() <= iterator.right());
        Assertions.assertTrue(iterator.right() <= iterator.end());
        int j = 0;
        if (!sparse) {
            int i = j;
            j = Arrays.binarySearch(indices, iterator.right());
            Assertions.assertTrue(j >= i, "Index not in original indices");
        }
        // Iterate
        while (iterator.right() < iterator.end()) {
            final int previous = iterator.right();
            Assertions.assertTrue(iterator.next());
            Assertions.assertTrue(previous < iterator.left(), "Did not advance");
            // Check invariants
            Assertions.assertTrue(iterator.left() <= iterator.right());
            Assertions.assertTrue(iterator.right() <= iterator.end());
            if (!sparse) {
                int i = Arrays.binarySearch(indices, iterator.left());
                Assertions.assertEquals(j + 1, i, "next skipped original indices");
                j = Arrays.binarySearch(indices, iterator.right());
                Assertions.assertTrue(j >= i, "Index not in original indices");
            }
        }
        Assertions.assertEquals(indices[nm1], iterator.right());
        Assertions.assertFalse(iterator.next());
        Assertions.assertEquals(indices[nm1], iterator.right());
        Assertions.assertFalse(iterator.next());

        // Test position after
        for (final int jump : new int[] {1, 2, 3}) {
            iterator = constructor.apply(indices, indices.length);
            IndexIterator iterator2 = constructor.apply(indices, indices.length);

            for (int i = jump; i < indices.length; i += jump) {
                final int k = indices[i];
                if (k == iterator.end()) {
                    Assertions.assertFalse(iterator.positionAfter(k));
                    Assertions.assertEquals(k, iterator.right());
                } else {
                    Assertions.assertTrue(iterator.positionAfter(k));
                    Assertions.assertTrue(k < iterator.right());
                }
                // Iterate using next. Ensures the sequence output is the same.
                boolean result = true;
                while (result && iterator2.right() <= k) {
                    result = iterator2.next();
                }
                Assertions.assertEquals(iterator2.left(), iterator.left(), () -> "left after " + k);
                Assertions.assertEquals(iterator2.right(), iterator.right(), () -> "right after " + k);
            }
            Assertions.assertFalse(iterator.positionAfter(indices[nm1]));
            Assertions.assertEquals(indices[nm1], iterator.right());
            Assertions.assertFalse(iterator.positionAfter(indices[nm1]));
            Assertions.assertEquals(indices[nm1], iterator.right());
        }
    }

    static Stream<int[]> testNextIndex() {
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        final Stream.Builder<int[]> builder = Stream.builder();
        builder.accept(new int[] {4});
        builder.accept(new int[] {4, 78});
        builder.accept(new int[] {4, 78, 999});
        builder.accept(new int[] {4, 78, 79, 999});
        builder.accept(new int[] {4, 5, 6, 7, 8});
        for (final int size : new int[] {10, 50, 500}) {
            for (final int n : new int[] {2, 5, 10}) {
                final int[] a = rng.ints(n, 0, size).distinct().sorted().toArray();
                builder.accept(a.clone());
                // Force use of index 0 and max index
                a[0] = 0;
                a[a.length - 1] = Integer.MAX_VALUE - 1;
                builder.accept(a);
            }
        }
        return builder.build();
    }
}
