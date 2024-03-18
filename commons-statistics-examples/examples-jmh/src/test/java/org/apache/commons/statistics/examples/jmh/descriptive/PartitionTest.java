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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Formatter;
import java.util.LinkedList;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.PairedKeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource.Distribution;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource.Modification;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Partition}.
 */
class PartitionTest {
    /** Default single pivot strategy. */
    private static final PivotingStrategy SP = PivotingStrategy.MEDIAN_OF_3;
    /** Default single pivot strategy. */
    private static final DualPivotingStrategy DP = DualPivotingStrategy.SORT_5;
    /** Default minimum quick select length. */
    private static final int QS = 3;
    /** Default minimum quick select length for dual pivot. */
    private static final int QS2 = 5;
    /** Default heap select shift. Using 31 disables length dependence. */
    private static final int HS = 31;
    /** Default heap select constant. */
    private static final int HC = 2;
    /** Default heap select mask shift constant. Using 31 disables length dependence. */
    private static final int MS = 31;
    /** Default sort select constant. */
    private static final int SC = 0;

    /**
     * Partition function. Used to test different implementations.
     */
    private interface DoubleRangePartitionFunction {
        /**
         * Partition the array such that range of indices {@code [ka, kb]} correspond to
         * their correctly sorted value in the equivalent fully sorted array. For all
         * indices {@code k} and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
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
     * Partition function. Used to test different implementations.
     */
    private interface DoublePartitionFunction {
        /**
         * Partition the array such that indices {@code k} correspond to their correctly
         * sorted value in the equivalent fully sorted array. For all indices {@code k}
         * and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * <p>This method allows variable length indices using a count of the indices to
         * process.
         *
         * @param a Values.
         * @param k Indices.
         * @param n Count of indices.
         */
        void partition(double[] a, int[] k, int n);
    }

    /**
     * Partition function. Used to test different implementations.
     */
    private interface DoublePartitionFunction2 {
        /**
         * Partition the array such that indices {@code k} correspond to their correctly
         * sorted value in the equivalent fully sorted array. For all indices {@code k}
         * and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * @param a Values.
         * @param k Indices.
         */
        void partition(double[] a, int... k);
    }

    @ParameterizedTest
    @MethodSource
    void testSortNaN(double[] values) {
        final double[] sorted = sort(values);
        final int last = Partition.sortNaN(values);
        // index of last non-NaN
        int i = sorted.length;
        while (--i >= 0) {
            if (!Double.isNaN(sorted[i])) {
                break;
            }
        }
        Assertions.assertEquals(i, last);
        // Check the data is the same
        Arrays.sort(values);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    static Stream<double[]> testSortNaN() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        final double nan = Double.NaN;
        builder.add(new double[0]);
        builder.add(new double[] {1.23});
        builder.add(new double[] {nan});
        builder.add(new double[] {nan, nan});
        builder.add(new double[] {nan, nan, nan});
        for (final int size : new int[] {2, 5}) {
            final double[] values = rng.doubles(size).toArray();
            builder.add(values.clone());
            // Random NaNs
            for (int n = 1; n < size; n++) {
                final double[] x = values.clone();
                Arrays.fill(x, 0, n, nan);
                for (int i = 0; i < 5; i++) {
                    builder.add(TestUtils.shuffle(rng, x).clone());
                }
            }
        }
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionMinMax"})
    void testPartitionMin(double[] values, int from, int to) {
        assertRangePartition(sort(values, from, to),
            (a, l, r, ka, kb) -> Partition.partitionMin(values, from, to),
            values, from, to, from, from);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionMinMax"})
    void testPartitionMax(double[] values, int from, int to) {
        assertRangePartition(sort(values, from, to),
            (a, l, r, ka, kb) -> Partition.partitionMax(values, from, to),
            values, from, to, to, to);
    }

    static Stream<Arguments> testPartitionMinMax() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(new double[] {1, 2, 3, 4, 5}, 0, 4));
        builder.add(Arguments.of(new double[] {5, 4, 3, 2, 1}, 0, 4));
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        for (final int size : new int[] {5, 10}) {
            final double[] values = rng.doubles(size).toArray();
            builder.add(Arguments.of(values.clone(), 0, size - 1));
            builder.add(Arguments.of(values.clone(), size >>> 1, size - 1));
            builder.add(Arguments.of(values.clone(), 1, size >>> 1));
        }
        builder.add(Arguments.of(new double[] {-0.0, 0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, -0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {-0.0, -0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, 0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, -0.0, 0.0, -0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {-0.0, 0.0, -0.0, 0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {0.0, -0.0, -0.0, 0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {-0.0, 0.0, 0.0, -0.0}, 0, 3));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = "testPartitionMinMax2")
    void testPartitionMin2IgnoreZeros(double[] values, int from, int to) {
        assertRangePartition(sort(values, from, to),
            (a, l, r, ka, kb) -> {
                replaceNegativeZeros(values, from, to);
                Partition.partitionMin2IgnoreZeros(values, from, to);
                restoreNegativeZeros(values, from, to);
            },
            values, from, to, from, from + 1);
    }

    @ParameterizedTest
    @MethodSource(value = "testPartitionMinMax2")
    void testPartitionMax2IgnoreZeros(double[] values, int from, int to) {
        assertRangePartition(sort(values, from, to),
            (a, l, r, ka, kb) -> {
                replaceNegativeZeros(values, from, to);
                Partition.partitionMax2IgnoreZeros(values, from, to);
                restoreNegativeZeros(values, from, to);
            },
            values, from, to, to - 1, to);
    }

    static Stream<Arguments> testPartitionMinMax2() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final double[] values = {-0.0, 0.0, 1};
        final double x = Double.NaN;
        final double y = 42;
        for (final double a : values) {
            for (final double b : values) {
                builder.add(Arguments.of(new double[] {a, b}, 0, 1));
                builder.add(Arguments.of(new double[] {x, a, b, y}, 1, 2));
                for (final double c : values) {
                    builder.add(Arguments.of(new double[] {a, b, c}, 0, 2));
                    builder.add(Arguments.of(new double[] {x, a, b, c, y}, 1, 3));
                    for (final double d : values) {
                        builder.add(Arguments.of(new double[] {a, b, c, d}, 0, 3));
                        builder.add(Arguments.of(new double[] {x, a, b, c, d, y}, 1, 4));
                    }
                }
            }
        }
        builder.add(Arguments.of(new double[] {-1, -1, -1, 4, 3, 2, 1, y}, 3, 6));
        builder.add(Arguments.of(new double[] {1, 2, 3, 4, 5}, 0, 4));
        builder.add(Arguments.of(new double[] {5, 4, 3, 2, 1}, 0, 4));
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        for (final int size : new int[] {5, 10}) {
            final double[] a = rng.doubles(size).toArray();
            builder.add(Arguments.of(a.clone(), 0, size - 1));
            builder.add(Arguments.of(a.clone(), size >>> 1, size - 1));
            builder.add(Arguments.of(a.clone(), 1, size >>> 1));
        }
        builder.add(Arguments.of(new double[] {-0.0, 0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, -0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {-0.0, -0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, 0.0}, 0, 1));
        builder.add(Arguments.of(new double[] {0.0, -0.0, 0.0, -0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {-0.0, 0.0, -0.0, 0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {0.0, -0.0, -0.0, 0.0}, 0, 3));
        builder.add(Arguments.of(new double[] {-0.0, 0.0, 0.0, -0.0}, 0, 3));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax", "testPartitionMinMax2"})
    void testPartitionMinK(double[] values, int from, int to) {
        final double[] sorted = sort(values, from, to);

        final double[] x = values.clone();
        replaceNegativeZeros(x, from, to);
        final DoubleRangePartitionFunction fun = (a, l, r, ka, kb) -> {
            Partition.partitionMinK(a, l, r, kb, kb - ka);
            restoreNegativeZeros(a, l, r);
        };

        for (int k = from; k <= to; k++) {
            assertRangePartition(sorted, fun, x.clone(), from, to, k, k);
            if (k > from) {
                // Sort an extra 1
                assertRangePartition(sorted, fun, x.clone(), from, to, k - 1, k);
                if (k > from + 1) {
                    // Sort all
                    // Test clipping with k < from
                    assertRangePartition(sorted, fun, x.clone(), from, to, from - 23, k);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax", "testPartitionMinMax2"})
    void testPartitionMaxK(double[] values, int from, int to) {
        final double[] sorted = sort(values, from, to);

        final double[] x = values.clone();
        replaceNegativeZeros(x, from, to);
        final DoubleRangePartitionFunction fun = (a, l, r, ka, kb) -> {
            Partition.partitionMaxK(a, l, r, ka, kb - ka);
            restoreNegativeZeros(a, l, r);
        };

        for (int k = from; k <= to; k++) {
            assertRangePartition(sorted, fun, x.clone(), from, to, k, k);
            if (k < to) {
                // Sort an extra 1
                assertRangePartition(sorted, fun, x.clone(), from, to, k, k + 1);
                if (k < to - 1) {
                    // Sort all
                    // Test clipping with k > to
                    assertRangePartition(sorted, fun, x.clone(), from, to, k, to + 23);
                }
            }
        }
    }

    static Stream<Arguments> testPartitionK() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(new double[] {1}, 0, 0));
        builder.add(Arguments.of(new double[] {3, 2, 1}, 1, 1));
        builder.add(Arguments.of(new double[] {2, 1}, 0, 1));
        builder.add(Arguments.of(new double[] {4, 3, 2, 1}, 1, 2));
        builder.add(Arguments.of(new double[] {-1, 0.0, -0.0, -0.0, 1}, 0, 4));
        builder.add(Arguments.of(new double[] {-1, 0.0, -0.0, -0.0, 1}, 0, 2));
        builder.add(Arguments.of(new double[] {1, 0.0, -0.0, -0.0, -1}, 0, 4));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 1, 6));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testHeapSelectPair(double[] values, int from, int to, int k1, int k2) {
        final double[] sorted = sort(values, from, to);
        Partition.heapSelect(values, from, to, k1, k2);
        Assertions.assertEquals(sorted[k1], values[k1]);
        Assertions.assertEquals(sorted[k2], values[k2]);
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    static Stream<Arguments> testHeapSelectPair() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 1, 2));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 2, 2));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 5, 7));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 1, 6));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 4, 4));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testHeapSelectRange(double[] values, int from, int to, int k1, int k2) {
        assertRangePartition(sort(values, from, to),
            Partition::heapSelectRange, values, from, to, k1, k2);
    }

    static Stream<Arguments> testHeapSelectRange() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 1, 2));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 2, 2));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 5, 7));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 1, 6));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 0, 3));
        builder.add(Arguments.of(new double[] {-1, 2, -3, 4, -4, 3, -2, 1}, 0, 7, 4, 7));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax", "testPartitionMinMax2"})
    void testSortMinK(double[] values, int from, int to) {
        final double[] sorted = sort(values, from, to);

        final double[] x = values.clone();
        replaceNegativeZeros(x, from, to);
        final DoubleRangePartitionFunction fun = (a, l, r, ka, kb) -> {
            Partition.sortMinK(a, l, r, kb);
            restoreNegativeZeros(a, l, r);
        };

        for (int k = from; k <= to; k++) {
            assertRangePartition(sorted, fun, x.clone(), from, to, k, k);
            if (k > from) {
                // Sort an extra 1
                assertRangePartition(sorted, fun, x.clone(), from, to, k - 1, k);
                if (k > from + 1) {
                    // Sort all
                    // Test clipping with k < from
                    assertRangePartition(sorted, fun, x.clone(), from, to, from - 23, k);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax", "testPartitionMinMax2"})
    void testSortMaxK(double[] values, int from, int to) {
        final double[] sorted = sort(values, from, to);

        final double[] x = values.clone();
        replaceNegativeZeros(x, from, to);
        final DoubleRangePartitionFunction fun = (a, l, r, ka, kb) -> {
            Partition.sortMaxK(a, l, r, ka);
            restoreNegativeZeros(a, l, r);
        };

        for (int k = from; k <= to; k++) {
            assertRangePartition(sorted, fun, x.clone(), from, to, k, k);
            if (k < to) {
                // Sort an extra 1
                assertRangePartition(sorted, fun, x.clone(), from, to, k, k + 1);
                if (k < to - 1) {
                    // Sort all
                    // Test clipping with k > to
                    assertRangePartition(sorted, fun, x.clone(), from, to, k, to + 23);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testHeapSelectRange"})
    void testSortSelectRange(double[] values, int from, int to, int k1, int k2) {
        assertRangePartition(sort(values, from, to),
            Partition::sortSelectRange, values, from, to, k1, k2);
    }

    /**
     * Return a copy of the {@code values} sorted.
     *
     * @param values Values.
     * @return the copy
     */
    private static double[] sort(double[] values) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted;
    }

    /**
     * Return a copy of the {@code values} sorted in the range {@code [from, to]}.
     *
     * @param values Values.
     * @param from From (inclusive).
     * @param to To (inclusive).
     * @return the copy
     */
    private static double[] sort(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        return sorted;
    }

    /**
     * Assert the function correctly partitions the range.
     *
     * @param sorted Expected sort result.
     * @param fun Partition function.
     * @param values Values.
     * @param from From (inclusive).
     * @param to To (inclusive).
     * @param ka Lower index to select.
     * @param kb Upper index to select.
     */
    private static void assertRangePartition(double[] sorted,
            DoubleRangePartitionFunction fun,
            double[] values, int from, int to, int ka, int kb) {
        Arrays.sort(sorted, from, to + 1);
        fun.partition(values, from, to, ka, kb);
        // Clip
        ka = ka < from ? from : ka;
        kb = kb > to ? to : kb;
        for (int i = ka; i <= kb; i++) {
            final int index = i;
            Assertions.assertEquals(sorted[i], values[i], () -> "index: " + index);
        }
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    @Test
    void testFloorLog2() {
        // Here expected = -Infinity; actual = -1
        Assertions.assertEquals(-1, Partition.floorLog2(0));
        Assertions.assertEquals(0, Partition.floorLog2(1));
        // Create a series of powers of 2, start at 2^1
        long p = 1;
        for (int i = 1;; i++) {
            p *= 2;
            if (p > Integer.MAX_VALUE) {
                break;
            }
            final int x = (int) p;
            Assertions.assertEquals(i - 1, Partition.floorLog2(x - 1));
            Assertions.assertEquals(i, Partition.floorLog2(x));
            Assertions.assertEquals(i, Partition.floorLog2(x + 1));
        }
    }

    @Test
    void testLog3() {
        // Reasonable behaviour at small x
        Assertions.assertEquals(0, Partition.log3(0));
        Assertions.assertEquals(0, Partition.log3(1));
        Assertions.assertEquals(1, Partition.log3(2));
        Assertions.assertEquals(1, Partition.log3(3));
        Assertions.assertEquals(1, Partition.log3(4));
        Assertions.assertEquals(1, Partition.log3(5));
        Assertions.assertEquals(1, Partition.log3(6));
        Assertions.assertEquals(1, Partition.log3(7));
        Assertions.assertEquals(2, Partition.log3(8));
        // log3(2^31-1) = 19.5588223...
        Assertions.assertEquals(19, Partition.log3(Integer.MAX_VALUE));
        // Create a series of powers of 3, start at 2^3
        long p = 3;
        for (int i = 2;; i++) {
            p *= 3;
            if (p > Integer.MAX_VALUE) {
                break;
            }
            final int x = (int) p;
            // Computes round(log3(x)) when x is close to a power of 3
            Assertions.assertEquals(i, Partition.log3(x - 1));
            Assertions.assertEquals(i, Partition.log3(x));
            Assertions.assertEquals(i, Partition.log3(x + 1));
            // Half-way point is within the bracket [i, i+1]
            final int y = (int) Math.floor(Math.pow(3, i + 0.5));
            Assertions.assertTrue(Partition.log3(y) >= i);
            Assertions.assertTrue(Partition.log3(y + 1) <= i + 1);
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.INDEX_SET)::partitionSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMPivotCache(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.PIVOT_CACHE)::partitionSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMSequential(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.SEQUENTIAL)::partitionSBM);
    }

    // Introselect versions use heap select configuration.
    // We test the different PairedKeyStrategy options alongside KeyStrategy options.

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISP(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.ORDERED_KEYS)
            .setPairedKeyStrategy(PairedKeyStrategy.PAIRED_KEYS)
            ::partitionISP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIBM(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.ORDERED_KEYS)
            .setPairedKeyStrategy(PairedKeyStrategy.PAIRED_KEYS)
            ::partitionIBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBM(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.ORDERED_KEYS)
            .setPairedKeyStrategy(PairedKeyStrategy.PAIRED_KEYS)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMScanningKey(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.SCANNING_KEY_SEARCHABLE_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.PAIRED_KEYS)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMSearchKey(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.SEARCH_KEY_SEARCHABLE_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.TWO_KEYS)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.INDEX_SET)
            .setPairedKeyStrategy(PairedKeyStrategy.SEARCHABLE_INTERVAL)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMKeyUpdating(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.KEY_UPDATING_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.UPDATING_INTERVAL)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMIndexSetUpdating(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.INDEX_SET_UPDATING_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.UPDATING_INTERVAL)
            ::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(1)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexSet2(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(2)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMIndexIterator(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS, HS, HC, SC).setKeyStrategy(KeyStrategy.INDEX_ITERATOR)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexIterator(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_ITERATOR)
            .setCompression(1)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexIterator4(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_ITERATOR)
            .setCompression(4)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDNF(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC, SC)::partitionIDNF);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPScanningKey(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.SCANNING_KEY_SEARCHABLE_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.PAIRED_KEYS)
            ::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPSearchKey(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.SEARCH_KEY_SEARCHABLE_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.TWO_KEYS)
            ::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.INDEX_SET)
            .setPairedKeyStrategy(PairedKeyStrategy.SEARCHABLE_INTERVAL)
            ::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPKeyUpdating(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.KEY_UPDATING_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.SEARCHABLE_INTERVAL)
            ::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPIndexSetUpdating(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.INDEX_SET_UPDATING_INTERVAL)
            .setPairedKeyStrategy(PairedKeyStrategy.SEARCHABLE_INTERVAL)
            ::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPCompressedIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(1)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPCompressedIndexSet2(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(2)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPIndexIterator(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(DP, QS2, HS, HC, MS, SC).setKeyStrategy(KeyStrategy.INDEX_ITERATOR)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPCompressedIndexIterator(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS, SC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_ITERATOR)
            .setCompression(1)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testSelect(double[] values, int[] indices) {
        assertPartition(values, indices, Partition::select);
    }

    static void assertPartitionPaired(double[] values, int[] indices, DoublePartitionFunction2 function) {
        // Create a paired version of the indices.
        // We apply the partition function to this and test the result as if values
        // had been partitioned using indices.
        final BitSet bs = new BitSet();
        for (final int i : indices) {
            bs.set(i);
        }
        final int[] unique = bs.stream().toArray();
        // compress pairs
        int n = 1;
        for (int i = 1; i < unique.length; i++) {
            final int k = unique[i];
            if (k - 1 == unique[n - 1]) {
                // Mark as pair with sign bit
                unique[n - 1] |= Integer.MIN_VALUE;
                continue;
            }
            unique[n++] = k;
        }
        final int[] k = Arrays.copyOf(unique, n);
        TestUtils.shuffle(RandomSource.XO_RO_SHI_RO_128_PP.create(0xdeadbeef), k);
        assertPartition(values, indices, (a, ignoredIndices, ignoredN) -> function.partition(a, k));
    }

    static void assertPartition(double[] values, int[] indices, DoublePartitionFunction function) {
        final double[] data = values.clone();
        final double[] sorted = values.clone();
        Arrays.sort(sorted);
        // Indices may be destructively modified
        function.partition(data, indices.clone(), indices.length);
        if (indices.length == 0) {
            return;
        }
        for (final int k : indices) {
            Assertions.assertEquals(sorted[k], data[k], () -> "k[" + k + "]");
        }
        // Check partial ordering
        Arrays.sort(indices);
        int i = 0;
        for (final int k : indices) {
            final double value = sorted[k];
            while (i < k) {
                final int j = i;
                Assertions.assertTrue(Double.compare(data[i], value) <= 0,
                    () -> j + " < " + k + " : " + data[j] + " < " + value);
                i++;
            }
        }
        final int k = indices[indices.length - 1];
        final double value = sorted[k];
        while (i < data.length) {
            final int j = i;
            Assertions.assertTrue(Double.compare(data[i], value) >= 0,
                () -> k + " < " + j);
            i++;
        }
        Arrays.sort(data);
        Assertions.assertArrayEquals(sorted, data, "Data destroyed");
    }

    static Stream<Arguments> testPartition() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        // Sizes above and below the threshold for partitioning
        for (final int size : new int[] {5, 50, 500}) {
            final double[] values = IntStream.range(0, size).asDoubleStream().toArray();
            final double[] zeros = values.clone();
            Arrays.fill(zeros, 0, size >>> 2, -0.0);
            Arrays.fill(zeros, size >>> 2, size >>> 1, 0.0);
            for (final int k : new int[] {1, 2, 3, size}) {
                for (int i = 0; i < 15; i++) {
                    // Note: Duplicate indices do not matter
                    final int[] indices = rng.ints(k, 0, size).toArray();
                    builder.add(Arguments.of(
                        TestUtils.shuffle(rng, values.clone()),
                        indices));
                    builder.add(Arguments.of(
                        TestUtils.shuffle(rng, zeros.clone()),
                        indices));
                }
            }
            // Test sequential processing by creating potential ranges
            // after an initial low point. This should be high enough
            // so any range analysis that joins indices will leave the initial
            // index as a single point.
            final int limit = 50;
            if (size > limit) {
                for (int i = 0; i < 10; i++) {
                    final int[] indices = rng.ints(size - limit, limit, size).toArray();
                    // This sets a low index
                    indices[rng.nextInt(indices.length)] = rng.nextInt(0, limit >>> 1);
                    builder.add(Arguments.of(
                        TestUtils.shuffle(rng, values.clone()),
                        indices));
                }
            }
            // min; max; min/max
            builder.add(Arguments.of(values.clone(), new int[] {0}));
            builder.add(Arguments.of(values.clone(), new int[] {size - 1}));
            builder.add(Arguments.of(values.clone(), new int[] {0, size - 1}));
            builder.add(Arguments.of(zeros.clone(), new int[] {0}));
            builder.add(Arguments.of(zeros.clone(), new int[] {size - 1}));
            builder.add(Arguments.of(zeros.clone(), new int[] {0, size - 1}));
        }
        final double nan = Double.NaN;
        builder.add(Arguments.of(new double[] {}, new int[0]));
        builder.add(Arguments.of(new double[] {nan}, new int[] {0}));
        builder.add(Arguments.of(new double[] {-0.0, nan}, new int[] {1}));
        builder.add(Arguments.of(new double[] {nan, nan, nan}, new int[] {2}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0, nan}, new int[] {3}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0, nan}, new int[] {1, 2}));
        builder.add(Arguments.of(new double[] {nan, 0.0, 1, -0.0, nan}, new int[] {1, 3}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0}, new int[] {0, 2}));
        builder.add(Arguments.of(new double[] {nan, 1.23, 0.0, -4.56, -0.0, nan}, new int[] {0, 1, 3}));
        // Dual-pivot with a large middle region (> 5 / 8) requires equal elements loop
        final int n = 128;
        final double[] x = IntStream.range(0, n).asDoubleStream().toArray();
        // Put equal elements in the central region:
        //          2/16      6/16             10/16      14/16
        // |  <P1    |    P1   |   P1< & < P2    |    P2    |    >P2    |
        final int sixteenth = n / 16;
        final int i2 = 2 * sixteenth;
        final int i6 = 6 * sixteenth;
        final double p1 = x[i2];
        final double p2 = x[n - i2];
        // Lots of values equal to the pivots
        Arrays.fill(x, i2, i6, p1);
        Arrays.fill(x, n - i6, n - i2, p2);
        // Equal value in between the pivots
        Arrays.fill(x, i6, n - i6, (p1 + p2) / 2);
        // Shuffle this and partition in the middle.
        // Use a fix seed to ensure we hit coverage with only 5 loops.
        rng = RandomSource.XO_SHI_RO_128_PP.create(-8111061151820577011L);
        for (int i = 0; i < 5; i++) {
            builder.add(Arguments.of(TestUtils.shuffle(rng, x.clone()), new int[] {50}));
        }
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortSBM(double[] values) {
        assertSort(values,
            new Partition(SP, 3)::sortSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testHeapSortUsingHeapSelectRange(double[] values) {
        assumeNonNaN(values);
        Assumptions.assumeTrue(values.length > 0);
        assertSort(values, x -> {
            replaceNegativeZeros(x, 0, x.length - 1);
            Partition.heapSelectRange(x, 0, x.length - 1, 0, x.length - 1);
            restoreNegativeZeros(x, 0, x.length - 1);
        });
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testHeapSort(double[] values) {
        assumeNonNaN(values);
        Assumptions.assumeTrue(values.length > 0);
        assertSort(values, x -> {
            replaceNegativeZeros(x, 0, x.length - 1);
            Partition.heapSort(x, 0, x.length - 1);
            restoreNegativeZeros(x, 0, x.length - 1);
        });
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortISP(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortISP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortIBM(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortIBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortISBM(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortIDNF1(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortIDNF1);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortIDNF2(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortIDNF2);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortIDNF3(double[] values) {
        assertSort(values,
            new Partition(SP, QS)::sortIDNF3);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortIDP(double[] values) {
        assertSort(values,
            new Partition(DP, QS2)::sortIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortSelect(double[] values) {
        // This tests that the select partitioning function performs
        // a full sort when the IndexInterval is saturated.
        assertSort(values, a -> {
            final int right = Partition.sortNaN(a);
            if (right < 1) {
                return;
            }
            replaceNegativeZeros(a, 0, right);
            Partition.select(a, 0, right, IndexIntervals.interval(0, right), 100, Partition.SORTSELECT_SIZE);
            restoreNegativeZeros(a, 0, right);
        });
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testInsertionSort(double[] values) {
        assumeNonNaN(values);
        assertSort(values, x -> {
            replaceNegativeZeros(x, 0, x.length - 1);
            Sorting.sort(x, 0, x.length - 1, false);
            restoreNegativeZeros(x, 0, x.length - 1);
        });
        if (values.length < 2) {
            return;
        }
        // Check internal sort
        // Set pivot at lower end
        values[0] = Arrays.stream(values).min().getAsDouble();
        // check internal sort
        assertSort(values, x -> {
            replaceNegativeZeros(x, 1, x.length - 1);
            Sorting.sort(x, 1, x.length - 1, false);
            restoreNegativeZeros(x, 1, x.length - 1);
        });
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testInsertionSort5(double[] values) {
        // Cannot handle NaN or -0.0
        // Negative zeros are swapped for a proxy
        assumeNonNaN(values);
        final double[] data = Arrays.copyOf(values, 5);
        assertSort(data, x -> {
            replaceNegativeZeros(x, 0, x.length - 1);
            Sorting.sort5(x, 0, 1, 2, 3, 4);
            restoreNegativeZeros(x, 0, x.length - 1);
        });
    }

    @Test
    void testSortZero() {
        final double a = -0.0;
        final double b = 0.0;
        final double[][] values = new double[][] {
            {a, a},
            {a, b},
            {b, a},
            {b, b},
            {a, a, a},
            {a, a, b},
            {a, b, a},
            {a, b, b},
            {b, a, a},
            {b, a, b},
            {b, b, a},
            {b, b, b},
            {a, a, a, a},
            {a, a, a, b},
            {a, a, b, a},
            {a, a, b, b},
            {a, b, a, a},
            {a, b, a, b},
            {a, b, b, a},
            {a, b, b, b},
            {b, a, a, a},
            {b, a, a, b},
            {b, a, b, a},
            {b, a, b, b},
            {b, b, a, a},
            {b, b, a, b},
            {b, b, b, a},
            {b, b, b, b},
        };
        for (final double[] v : values) {
            assertSort(v, x -> Partition.sortZero(x, 0, x.length - 1));
        }
    }

    private static void assertSort(double[] values, Consumer<double[]> function) {
        final double[] data = values.clone();
        final double[] sorted = values.clone();
        Arrays.sort(sorted);
        function.accept(data);
        Assertions.assertArrayEquals(sorted, data);
    }

    static Stream<double[]> testSort() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        // Sizes above and below the threshold for partitioning
        for (final int size : new int[] {5, 50}) {
            double[] a = new double[size];
            Arrays.fill(a, 1.23);
            builder.add(a.clone());
            for (int ii = 0; ii < size; ii++) {
                a[ii] = ii;
            }
            builder.add(a.clone());
            for (int ii = 0; ii < size; ii++) {
                a[ii] = size - ii;
            }
            builder.add(a.clone());
            for (int i = 0; i < 5; i++) {
                a = rng.doubles(size).toArray();
                builder.add(a.clone());
                final int j = rng.nextInt(size);
                final int k = rng.nextInt(size);
                a[j] = Double.NaN;
                a[k] = Double.NaN;
                builder.add(a.clone());
                a[j] = -0.0;
                a[k] = 0.0;
                builder.add(a.clone());
                for (int z = 0; z < size; z++) {
                    a[z] = rng.nextBoolean() ? -0.0 : 0.0;
                }
                builder.add(a.clone());
                a[j] = -rng.nextDouble();
                a[k] = rng.nextDouble();
                builder.add(a.clone());
            }
        }
        final double nan = Double.NaN;
        builder.add(new double[] {});
        builder.add(new double[] {nan});
        builder.add(new double[] {-0.0, nan});
        builder.add(new double[] {nan, nan, nan});
        builder.add(new double[] {nan, 0.0, -0.0, nan});
        builder.add(new double[] {nan, 0.0, -0.0});
        builder.add(new double[] {nan, 0.0, 1, -0.0});
        builder.add(new double[] {nan, 1.23, 0.0, -4.56, -0.0, nan});
        return builder.build();
    }

    /**
     * Test key analysis.
     * The key analysis code decides the partition strategy. Currently this
     * supports recommendations for processing keys or ranges of keys in ascending
     * order based on separation between points, and the point when data partitioning
     * switches to a full sort.
     *
     * @param size Length of the data to partition.
     * @param k Indices (non-zero length).
     * @param n Count of indices (either {@code 1 <= n <= k.length} or -1).
     * @param minSeparation Minimum separation between points.
     * @param minSelectSize Minimum selection size for insertion sort rather than selection.
     * @param expected Expected keys (up to the end of the indices or the marker {@link Integer#MIN_VALUE})
     * @param cacheRange {@code [L, R]} bounds of returned {@link PivotCache}, or null.
     */
    @ParameterizedTest
    @MethodSource
    void testKeyAnalysis(int size, int[] k, int n, int minSeparation, int minSelectSize,
            int[] expected, int[] cacheRange) {
        // Set the number of keys
        n = n < 0 ? k.length : n;
        final PivotCache pivotCache = new Partition(minSelectSize)
            .keyAnalysis(size, k, n, minSeparation);
        // Truncate to the marker
        int m = 0;
        while (m < n && k[m] != Integer.MIN_VALUE) {
            m++;
        }
        if (m == 0) {
            // Full sort recommendation
            Assertions.assertArrayEquals(expected, new int[] {Integer.MIN_VALUE});
        } else {
            final int[] actual = Arrays.copyOf(k, m);
            Assertions.assertArrayEquals(expected, actual,
                () -> Arrays.toString(actual));
        }
        if (cacheRange == null) {
            Assertions.assertNull(pivotCache);
        } else {
            Assertions.assertEquals(cacheRange[0], pivotCache.left());
            Assertions.assertEquals(cacheRange[1], pivotCache.right());
        }
    }

    static Stream<Arguments> testKeyAnalysis() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final int allK = -1;
        final int[] noCache = null;
        final int[] fullSort = new int[] {Integer.MIN_VALUE};
        builder.add(Arguments.of(100, new int[] {3}, allK, 2, 0,
            new int[] {3}, noCache));
        builder.add(Arguments.of(100, new int[] {3, 4, 5}, allK, 1, 0,
            new int[] {3, 5}, noCache));
        builder.add(Arguments.of(100, new int[] {3, 4, 5, 8}, allK, 2, 0,
            new int[] {3, 5, ~8}, new int[] {8, 8}));
        builder.add(Arguments.of(100, new int[] {3, 4, 5, 6, 7, 8}, allK, 1, 0,
            new int[] {3, 8}, noCache));
        builder.add(Arguments.of(100, new int[] {3, 4, 7, 8}, allK, 1, 0,
            new int[] {3, 4, 7, 8}, new int[] {7, 8}));
        builder.add(Arguments.of(100, new int[] {3, 4, 7, 8, 99}, allK, 1, 0,
            new int[] {3, 4, 7, 8, ~99}, new int[] {7, 99}));
        // Full sort recommendation: cases not large enough
        builder.add(Arguments.of(20, new int[] {3, 5, 8, 17}, allK, 3, 0,
            new int[] {3, 8, ~17}, new int[] {17, 17}));
        builder.add(Arguments.of(20, new int[] {3, 5, 8, 17}, allK, 3, 10,
            new int[] {3, 8, ~17}, new int[] {17, 17}));
        builder.add(Arguments.of(20, new int[] {3, 5, 8, 17}, allK, 9, 0,
            new int[] {3, 17}, noCache));
        // Full sort based on a single range to the end (due to high min separation)
        builder.add(Arguments.of(20, new int[] {3, 5, 8, 17}, allK, 10, 10,
            fullSort, noCache));
        // Full sort based on min select size
        builder.add(Arguments.of(20, new int[] {10, 11}, allK, 1, 20,
            fullSort, noCache));
        // No min separation - process each index
        builder.add(Arguments.of(100, new int[] {3, 4, 5}, allK, 0, 0,
            new int[] {~3, ~4, ~5}, new int[] {4, 5}));
        builder.add(Arguments.of(100, new int[] {3, 4, 7, 8}, allK, 0, 0,
            new int[] {~3, ~4, ~7, ~8}, new int[] {4, 8}));
        // Duplicate keys
        builder.add(Arguments.of(100, new int[] {0, 1, 2, 2, 3, 3}, allK, 0, 0,
            new int[] {~0, ~1, ~2, ~3}, new int[] {1, 3}));
        builder.add(Arguments.of(100, new int[] {0, 1, 2, 2, 3, 3}, allK, 1, 0,
            new int[] {0, 3}, noCache));
        builder.add(Arguments.of(100, new int[] {0, 1, 2, 2, 3, 3, 8, 8, 8}, allK, 2, 0,
            new int[] {0, 3, ~8}, new int[] {8, 8}));
        builder.add(Arguments.of(100, new int[] {9, 6, 7, 8, 2, 1, 1, 3}, allK, 2, 0,
            new int[] {1, 3, 6, 9}, new int[] {6, 9}));
        // TODO: more cases

        // Repeat the contents of the stream with any case not using the full length of the data.
        // by padding with random indices (these should be ignored)
        final Stream.Builder<Arguments> builder2 = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        builder.build().forEach(arg -> {
            builder2.add(arg);
            // unpack
            final Object[] o = arg.get();
            final int size = (int) o[0];
            final int[] k = (int[]) o[1];
            final int n = (int) o[2];
            if (n < 0) {
                final Object[] o2 = o.clone();
                // Add extra
                final int extra = rng.nextInt(3, 10);
                final int len = k.length;
                // Extra are zeros
                final int[] k2 = Arrays.copyOf(k, len + extra);
                o2[1] = k2.clone();
                o2[2] = len;
                builder2.add(Arguments.of(o2));
                // Deliberately add indices not in the original
                final Object[] o3 = o2.clone();
                final int max = Arrays.stream(k).max().getAsInt();
                for (int i = len; i < k2.length; i++) {
                    k2[i] = rng.nextInt(max, size);
                }
                o3[1] = k2;
                builder2.add(Arguments.of(o3));
            }
        });
        return builder2.build();
    }

    /**
     * Assume the data are non-NaN, otherwise skip the test.
     *
     * @param a Data.
     */
    private void assumeNonNaN(double[] a) {
        for (int i = 0; i < a.length; i++) {
            Assumptions.assumeFalse(Double.isNaN(a[i]));
        }
    }

    /**
     * Replace negative zeros with a proxy. Uses -{@link Double#MIN_VALUE} as the proxy.
     *
     * @param a Data.
     * @param from Lower bound (inclusive).
     * @param to Upper bound (inclusive).
     */
    private static void replaceNegativeZeros(double[] a, int from, int to) {
        for (int i = from; i <= to; i++) {
            if (Double.doubleToRawLongBits(a[i]) == Long.MIN_VALUE) {
                a[i] = -Double.MIN_VALUE;
            }
        }
    }
    /**
     * Restore proxy negative zeros.
     *
     * @param a Data.
     * @param from Lower bound (inclusive).
     * @param to Upper bound (inclusive).
     */
    private static void restoreNegativeZeros(double[] a, int from, int to) {
        for (int i = from; i <= to; i++) {
            if (a[i] == -Double.MIN_VALUE) {
                a[i] = -0.0;
            }
        }
    }

    @ParameterizedTest
    @MethodSource
    void testSearch(int[] keys, int left, int right) {
        // Clip to correct range
        final int l = left < 0 ? 0 : left;
        final int r = right < 0 ? keys.length - 1 : right;
        for (int i = l; i <= r; i++) {
            final int k = keys[i];
            // Unspecified index when key is present
            Assertions.assertEquals(k, keys[Partition.searchLessOrEqual(keys, l, r, k)], "leq");
            Assertions.assertEquals(k, keys[Partition.searchGreaterOrEqual(keys, l, r, k)], "geq");
        }
        // Search above/below keys
        Assertions.assertEquals(l - 1, Partition.searchLessOrEqual(keys, l, r, keys[l] - 44), "leq below");
        Assertions.assertEquals(r, Partition.searchLessOrEqual(keys, l, r, keys[r] + 44), "leq above");
        Assertions.assertEquals(l, Partition.searchGreaterOrEqual(keys, l, r, keys[l] - 44), "geq below");
        Assertions.assertEquals(r + 1, Partition.searchGreaterOrEqual(keys, l, r, keys[r] + 44), "geq above");
        // Search between neighbour keys
        for (int i = l + 1; i <= r; i++) {
            // Bound: keys[i-1] < k < keys[i]
            final int k1 = keys[i - 1];
            final int k2 = keys[i];
            for (int k = k1 + 1; k < k2; k++) {
                Assertions.assertEquals(i - 1, Partition.searchLessOrEqual(keys, l, r, k), "leq between");
                Assertions.assertEquals(i, Partition.searchGreaterOrEqual(keys, l, r, k), "geq between");
            }
        }
    }

    static Stream<Arguments> testSearch() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final int allIndices = -1;
        builder.add(Arguments.of(new int[] {1}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 2}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 10}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 2, 3}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 4, 7}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 4, 5, 7}, allIndices, allIndices));
        // Duplicates. These match binary search when found.
        builder.add(Arguments.of(new int[] {1, 1, 1, 1, 1, 1}, allIndices, allIndices));
        builder.add(Arguments.of(new int[] {1, 1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5}, allIndices, allIndices));
        // Part of the range
        builder.add(Arguments.of(new int[] {1, 4, 5, 7, 13, 15}, 2, 4));
        builder.add(Arguments.of(new int[] {1, 4, 5, 7, 13, 15}, 0, 3));
        builder.add(Arguments.of(new int[] {1, 4, 5, 7, 13, 15}, 3, 5));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testPartitionDP(double[] a, int pivot1, int pivot2, int k0, int[] bounds) {
        final int r = a.length - 1;
        final int[] b = new int[3];
        Assertions.assertEquals(k0, Partition.partitionDP(a, 0, r, pivot1, pivot2, b));
        Assertions.assertArrayEquals(bounds, b);
    }

    static Stream<Arguments> testPartitionDP() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Test less-than fast-forward bounds check - all values are < the pivots
        //builder.add(Arguments.of(new double[] {3, 4, 10, 12, 5, 6}, 2, 3, 4, new int[] {5, 5, 5}));
        builder.add(Arguments.of(new double[] {3, 4, 10, 12, 5, 6}, 2, 3, 4, new int[] {5, 4, 5}));
        // Test greater-than fast-forward bounds check - all values are > the pivots
        //builder.add(Arguments.of(new double[] {3, 4, 1, 2, 5, 6}, 2, 3, 0, new int[] {1, 1, 1}));
        builder.add(Arguments.of(new double[] {3, 4, 1, 2, 5, 6}, 2, 3, 0, new int[] {1, 0, 1}));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testSelectRange(double[] a, int k1, int kn) {
        final double[] copy = a.clone();
        Arrays.sort(copy);
        Partition.select(a, 0, a.length - 1, k1, kn, 1000000);
        for (int i = k1; i <= kn; i++) {
            Assertions.assertEquals(copy[i], a[i]);
        }
        Arrays.sort(a);
        Assertions.assertArrayEquals(copy, a, "data destroyed");
    }

    static Stream<Arguments> testSelectRange() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Reverse data. Makes it simple to detect failed range selection
        final double[] a = IntStream.range(0, 50).asDoubleStream().toArray();
        for (int i = -1, j = a.length; ++i < --j;) {
            final double v = a[i];
            a[i] = a[j];
            a[j] = v;
        }
        builder.add(Arguments.of(a, 1, 1));
        builder.add(Arguments.of(a, 1, 2));
        builder.add(Arguments.of(a, 10, 12));
        builder.add(Arguments.of(a, 10, 42));
        builder.add(Arguments.of(a, 1, 48));
        builder.add(Arguments.of(a, 48, 49));
        return builder.build();
    }

    /**
     * This is not a test. It runs the introselect algorithm as a full sort on the specified
     * data. A histogram of the level of recursion required to visit all regions is recorded
     * to file.
     */
    @ParameterizedTest
    @MethodSource
    @Disabled("Used for testing")
    void testRecursion(Distribution dist, Modification mod, int length, int range, boolean dualPivot) {
        final int maxDepth = 2048;
        final int[] h = new int[maxDepth + 1];
        // Use the defaults.
        // If the single pivot strategy is changed from MEDIAN_OF_3 to DYNAMIC
        // this avoid excess recursion.
        final Partition p = new Partition(
            Partition.PIVOTING_STRATEGY,
            //PivotingStrategy.MEDIAN_OF_3, // Use this to see excess recursion
            Partition.DUAL_PIVOTING_STRATEGY,
            Partition.MIN_QUICKSELECT_SIZE,
            Partition.HEAPSELECT_SHIFT,
            Partition.HEAPSELECT_CONSTANT,
            Partition.HEAPSELECT_MASK_SHIFT,
            Partition.SORTSELECT_CONSTANT);
        p.setRecursionConsumer(i -> h[maxDepth - i]++);
        final AbstractDataSource source = new AbstractDataSource() {
            @Override
            protected int getLength() {
                return length;
            }
        };
        source.setDistribution(dist);
        source.setModification(mod);
        source.setRange(range);
        source.setup();

        // Sort the data. This will record the recursion depth when a region is complete.
        for (int i = 0; i < source.size(); i++) {
            final double[] x = source.getData(i);
            if (dualPivot) {
                p.introselect(Partition::partitionDP, x, 0, x.length - 1,
                    IndexIntervals.anyIndex(), 0, x.length - 1, maxDepth);
            } else {
                p.introselect(Partition::partitionSBM, x, 0, x.length - 1,
                    IndexIntervals.anyIndex(), 0, x.length - 1, maxDepth);
            }
        }

        // Bracket the histogram. Assume at least 1 non-zero value.
        int hi = h.length;
        do {
            --hi;
        } while (h[hi] == 0);

        // Summary statistics
        long s = 0;
        long ss = 0;
        long n = 0;
        for (int i = 0; i < h.length; i++) {
            final int c = h[i];
            if (c != 0) {
                n += c;
                s += (long) i * c;
                ss += (long) i * i * c;
            }
        }
        final double mean = s / (double) n;
        double variance = ss - ((double) s * s) / n;
        if (variance > 0) {
            variance = variance / (n - 1);
        } else {
            variance = Double.isFinite(variance) ? 0.0 : Double.NaN;
        }
        final String name = dualPivot ? "DP" : "SP";
        final String distName = dist == null ? "ALL" : dist.name();
        final String modName = mod == null ? "ALL" : mod.name();
        // Flag when the method used excessive recursion.
        // Note that recursion only occurs down to a small length which is finished with a sort.
        final double expected = Math.log((length + range * 0.5) / Partition.MIN_QUICKSELECT_SIZE) /
            Math.log(dualPivot ? 3 : 2);
        String excess = "";
        for (double m = mean; m > expected && m > mean - 10; m -= 1) {
            excess += "*";
        }
        TestUtils.printf("%s %10s %15s %d-%d : n=%11d  mean=%10.6f  std=%10.6f   max=%4d : expected=%10.6f %s%n",
            name, distName, modName, length, length + range,
            n, mean, Math.sqrt(variance), hi, expected, excess);

        // Record the histogram
        final String dir = System.getProperty("java.io.tmpdir");
        final Path path = Path.of(dir, String.format("%s_%s_%s_%d-%d.txt",
            name, distName, modName, length, length + range));
        try (BufferedWriter bw = Files.newBufferedWriter(path);
            Formatter f = new Formatter(bw)) {
            for (int i = 0; i <= hi; i++) {
                f.format("%d %d%n", i, h[i]);
            }
        } catch (final IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static Stream<Arguments> testRecursion() {
        TestUtils.printf("Save directory: %s%n", System.getProperty("java.io.tmpdir"));

        final Stream.Builder<Arguments> builder = Stream.builder();
        //final int length = 10000023;
        final int length = 1023;
        final int range = 2;

        // All
        //builder.add(Arguments.of(null, null, length, range, true));
        //builder.add(Arguments.of(null, null, length, range, false));

        // Individual distribution / modification
        // Both single-pivot method (using DYNAMIC pivoting strategy) and dual-pivot
        // method have a mean recursion just above the theoretical max recursion depth.
        for (final Boolean dp : new Boolean[] {Boolean.TRUE, Boolean.FALSE}) {
            for (final Distribution dist : Distribution.values()) {
                for (final Modification mod : Modification.values()) {
                    builder.add(Arguments.of(dist, mod, length, range, dp));
                }
            }
        }

        return builder.build();
    }

    /**
     * This is not a test. It outputs the Bentley and McIlroy test data, optionally
     * with a single round of partitioning performed. This allows visualising the
     * change to the data made by the partition algorithm.
     */
    @ParameterizedTest
    @MethodSource
    @Disabled("Used for testing")
    void testData(int length, int seed, int partition) {
        final AbstractDataSource source = new AbstractDataSource() {
            @Override
            protected int getLength() {
                return length;
            }
        };
        source.setRange(0);
        source.setModification(Modification.COPY);
        source.setSeed(seed);
        source.setRngSeed(0xdeadbeef);
        source.setup();

        // Get the data
        final double[][] data = IntStream.range(0, source.size())
            .mapToObj(source::getDataSample).toArray(double[][]::new);

        // Optional: Run a single round of partitioning on the data.
        final LinkedList<String> pivots = new LinkedList<>();
        final int left = 0;
        final int right = length - 1;
        final int[] bounds = new int[3];
        if (partition == 1) {
            for (final double[] d : data) {
                int p = Partition.PIVOTING_STRATEGY.pivotIndex(d, left, right);
                p = Partition.partitionSBM(d, left, right, p, bounds);
                pivots.add(formatPivotRange(p, bounds[0]));
            }
        } else if (partition == 2) {
            for (final double[] d : data) {
                int p = Partition.DUAL_PIVOTING_STRATEGY.pivotIndex(d, left, right, bounds);
                p = Partition.partitionDP(d, left, right, p, bounds[0], bounds);
                pivots.add(formatPivotRange(p, bounds[0]) + ":" + formatPivotRange(bounds[1], bounds[2]));
            }
        }

        // Print header (distributions are in enum order)
        TestUtils.printf("m %d%n", seed);
        TestUtils.printf("i");
        for (final Distribution d : Distribution.values()) {
            TestUtils.printf(" %s", d);
            if (!pivots.isEmpty()) {
                TestUtils.printf(pivots.pop());
            }
        }
        TestUtils.printf("%n");

        // Sort the data. This will record the recursion depth when a region is complete.
        for (int j = 0; j < length; j++) {
            TestUtils.printf("%d", j);
            for (int i = 0; i < data.length; i++) {
                TestUtils.printf(" %s", data[i][j]);
            }
            TestUtils.printf("%n");
        }
    }

    private static String formatPivotRange(int lo, int hi) {
        return lo == hi ? Integer.toString(lo) : lo + "-" + hi;
    }

    static Stream<Arguments> testData() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(128, 32, 0));
        return builder.build();
    }
}
