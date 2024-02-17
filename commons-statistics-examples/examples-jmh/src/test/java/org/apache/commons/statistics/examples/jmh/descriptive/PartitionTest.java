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
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
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
        final double[] sorted = values.clone();
        Arrays.sort(sorted);
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
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        Partition.partitionMin(values, from, to);
        Assertions.assertEquals(sorted[from], values[from]);
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
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
    @MethodSource(value = {"testPartitionMinMax"})
    void testPartitionMax(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        Partition.partitionMax(values, from, to);
        Assertions.assertEquals(sorted[to], values[to]);
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax"})
    void testPartitionMinK(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        for (int k = from; k <= to; k++) {
            final int target = k;
            double[] x = values.clone();
            replaceNegativeZeros(x, from, to);
            Partition.partitionMinK(x, from, to, k, 0);
            restoreNegativeZeros(x, from, to);
            Assertions.assertEquals(sorted[k], x[k], () -> Integer.toString(target));
            // Check the data is the same
            Arrays.sort(x, from, to + 1);
            Assertions.assertArrayEquals(sorted, x, "Data destroyed");
            if (k > from) {
                // Sort an extra 1
                x = values.clone();
                replaceNegativeZeros(x, from, to);
                Partition.partitionMinK(x, from, to, k, 1);
                restoreNegativeZeros(x, from, to);
                for (int i = k - 1; i <= k; i++) {
                    Assertions.assertEquals(sorted[i], x[i], () -> (target - 1) + " to " + target);
                }
                // Check the data is the same
                Arrays.sort(x, from, to + 1);
                Assertions.assertArrayEquals(sorted, x, "Data destroyed");
                if (k > from + 1) {
                    // Sort all
                    x = values.clone();
                    replaceNegativeZeros(x, from, to);
                    // Test clipping with size of range below k too large: it should be k - from
                    Partition.partitionMinK(x, from, to, k, k - from + 42);
                    restoreNegativeZeros(x, from, to);
                    for (int i = from; i <= k; i++) {
                        Assertions.assertEquals(sorted[i], x[i], () -> "Full sort to " + Integer.toString(target));
                    }
                    // Check the data is the same
                    Arrays.sort(x, from, to + 1);
                    Assertions.assertArrayEquals(sorted, x, "Data destroyed");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartitionK", "testPartitionMinMax"})
    void testPartitionMaxK(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        for (int k = from; k <= to; k++) {
            final int target = k;
            double[] x = values.clone();
            replaceNegativeZeros(x, from, to);
            Partition.partitionMaxK(x, from, to, k, 0);
            restoreNegativeZeros(x, from, to);
            Assertions.assertEquals(sorted[k], x[k], () -> Integer.toString(target));
            // Check the data is the same
            Arrays.sort(x, from, to + 1);
            Assertions.assertArrayEquals(sorted, x, "Data destroyed");
            if (k < to) {
                // Sort an extra 1
                x = values.clone();
                replaceNegativeZeros(x, from, to);
                Partition.partitionMaxK(x, from, to, k, 1);
                restoreNegativeZeros(x, from, to);
                for (int i = k; i <= k + 1; i++) {
                    Assertions.assertEquals(sorted[i], x[i], () -> target + " to " + (target + 1));
                }
                // Check the data is the same
                Arrays.sort(x, from, to + 1);
                Assertions.assertArrayEquals(sorted, x, "Data destroyed");
                if (k < to - 1) {
                    // Sort all
                    x = values.clone();
                    replaceNegativeZeros(x, from, to);
                    // Test clipping with size of range above k too large: it should be to - k
                    Partition.partitionMaxK(x, from, to, k, to - k + 42);
                    restoreNegativeZeros(x, from, to);
                    for (int i = k; i <= to; i++) {
                        Assertions.assertEquals(sorted[i], x[i], () -> "Full sort from " + Integer.toString(target));
                    }
                    // Check the data is the same
                    Arrays.sort(x, from, to + 1);
                    Assertions.assertArrayEquals(sorted, x, "Data destroyed");
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
    void testPartitionMin2(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        Partition.partitionMin2(values, from, to);
        Assertions.assertEquals(sorted[from], values[from]);
        if (to - from > 1) {
            Assertions.assertEquals(sorted[from + 1], values[from + 1]);
        }
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    @ParameterizedTest
    @MethodSource(value = "testPartitionMin2")
    void testPartitionMin2IgnoreZeros(double[] values, int from, int to) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        replaceNegativeZeros(values, from, to);
        Partition.partitionMin2IgnoreZeros(values, from, to);
        restoreNegativeZeros(values, from, to);
        Assertions.assertEquals(sorted[from], values[from]);
        if (to - from > 1) {
            Assertions.assertEquals(sorted[from + 1], values[from + 1]);
        }
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    static Stream<Arguments> testPartitionMin2() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final double[] values = {-0.0, 0.0, 1};
        final double x = Double.NaN;
        final double y = 42;
        for (final double a : values) {
            builder.add(Arguments.of(new double[] {a}, 0, 0));
            builder.add(Arguments.of(new double[] {x, a, y}, 1, 1));
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
    @MethodSource
    void testHeapSelect(double[] values, int from, int to, int k1, int k2) {
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        Partition.heapSelect(values, from, to, k1, k2);
        Assertions.assertEquals(sorted[k1], values[k1]);
        Assertions.assertEquals(sorted[k2], values[k2]);
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
    }

    static Stream<Arguments> testHeapSelect() {
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
        final double[] sorted = values.clone();
        Arrays.sort(sorted, from, to + 1);
        Partition.heapSelectRange(values, from, to, k1, k2);
        for (int i = k1; i <= k2; i++) {
            Assertions.assertEquals(sorted[i], values[i]);
        }
        // Check the data is the same
        Arrays.sort(values, from, to + 1);
        Assertions.assertArrayEquals(sorted, values, "Data destroyed");
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

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionKSBM(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.PIVOT_CACHE)::partitionKSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionK1SBM(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.PIVOT_CACHE)::partitionK1SBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionPairedSBM(double[] values, int[] indices) {
        assertPartitionPaired(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.INDEX_SET)::partitionPairedSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionPairedSBMPivotCache(double[] values, int[] indices) {
        assertPartitionPaired(values, indices,
            new Partition(SP, QS).setKeyStrategy(KeyStrategy.PIVOT_CACHE)::partitionPairedSBM);
    }

    // Introselect versions use heap select configuration.

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBM(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS, HS, HC).setKeyStrategy(KeyStrategy.ORDERED_KEYS)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMScanningKey(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS, HS, HC).setKeyStrategy(KeyStrategy.SCANNING_KEY_INTERVAL)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMSearchKey(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS, HS, HC).setKeyStrategy(KeyStrategy.SEARCH_KEY_INTERVAL)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(SP, QS, HS, HC).setKeyStrategy(KeyStrategy.INDEX_SET)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(1)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionISBMCompressedIndexSet2(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(SP, QS, HS, HC)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(2)::partitionISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPScanningKey(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(DP, QS2, HS, HC, MS).setKeyStrategy(KeyStrategy.SCANNING_KEY_INTERVAL)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPSearchKey(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(DP, QS2, HS, HC, MS).setKeyStrategy(KeyStrategy.SEARCH_KEY_INTERVAL)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices,
            new Partition(DP, QS2, HS, HC, MS).setKeyStrategy(KeyStrategy.INDEX_SET)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPCompressedIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(1)::partitionIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionIDPCompressedIndexSet2(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(DP, QS2, HS, HC, MS)
            .setKeyStrategy(KeyStrategy.COMPRESSED_INDEX_SET)
            .setCompression(2)::partitionIDP);
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
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
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
        builder.add(Arguments.of(new double[] {nan, nan, nan}, new int[] {2}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0, nan}, new int[] {3}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0, nan}, new int[] {1, 2}));
        builder.add(Arguments.of(new double[] {nan, 0.0, 1, -0.0, nan}, new int[] {1, 3}));
        builder.add(Arguments.of(new double[] {nan, 0.0, -0.0}, new int[] {0, 2}));
        builder.add(Arguments.of(new double[] {nan, 1.23, 0.0, -4.56, -0.0, nan}, new int[] {0, 1, 3}));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortRangeSBM(double[] values) {
        assertSort(values,
            new Partition(SP, 3)::sortRangeSBM);
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
            replaceNegativeZeros(a, 0, right);
            Partition.select(a, 0, right, IndexIntervals.anyIndex(), 0, right, 100);
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
        builder.add(new double[] {});
        builder.add(new double[] {Double.NaN});
        builder.add(new double[] {Double.NaN, Double.NaN, Double.NaN});
        builder.add(new double[] {Double.NaN, 0.0, -0.0, Double.NaN});
        builder.add(new double[] {Double.NaN, 0.0, -0.0});
        builder.add(new double[] {Double.NaN, 1.23, 0.0, -4.56, -0.0, Double.NaN});
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

    @ParameterizedTest
    @MethodSource
    void testCreateIndexSetForPairedIndices(int[] k, int n) {
        final int[] copy = k.clone();
        n = n < 0 ? k.length : n;
        final IndexSet set = Partition.createIndexSetForPairedIndices(k, n);
        final int min = Arrays.stream(copy).limit(n).map(i -> i & Integer.MAX_VALUE).min().getAsInt();
        final int max = Arrays.stream(copy).limit(n).map(i -> (i & Integer.MAX_VALUE) + (i < 0 ? 1 : 0)).max().getAsInt();
        Assertions.assertEquals(min, k[0] & Integer.MAX_VALUE, "Invalid min");
        Assertions.assertEquals(max, (k[n - 1] & Integer.MAX_VALUE) + (k[n - 1] < 0 ? 1 : 0), "Invalid max");
        // Check for destroyed data.
        // This is only relevant if the indices have an internal range.
        if (max - min > 1) {
            Arrays.sort(copy, 0, n);
            Arrays.sort(k, 0, n);
            Assertions.assertArrayEquals(copy, k, "Indices destroyed");
        }
        // Quick check we can write to the IndexSet with the entire range
        Assertions.assertFalse(set.get(min));
        set.add(min);
        Assertions.assertTrue(set.get(min));
        if (max != min) {
            Assertions.assertFalse(set.get(max));
            set.add(max);
            Assertions.assertTrue(set.get(max));
        }
    }

    static Stream<Arguments> testCreateIndexSetForPairedIndices() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final int allIndices = -1;
        final int signBit = Integer.MIN_VALUE;
        builder.add(Arguments.of(new int[] {1, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 1, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[] {1, 1, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {1, 2}, allIndices));
        builder.add(Arguments.of(new int[] {2, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 2, 3}, allIndices));
        builder.add(Arguments.of(new int[] {1, 3, 2}, allIndices));
        builder.add(Arguments.of(new int[] {2, 1, 3}, allIndices));
        builder.add(Arguments.of(new int[] {2, 3, 1}, allIndices));
        builder.add(Arguments.of(new int[] {3, 1, 2}, allIndices));
        builder.add(Arguments.of(new int[] {3, 2, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 2, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[] {2, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[] {1, 2, 3, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {1, 3, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2, 1, 3, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2, 3, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {3, 1, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {3, 2, 1, 99, 98, 97}, 3));
        // Paired keys. Replace highest value from above with the next value and a sign bit.
        builder.add(Arguments.of(new int[] {1, 1 | signBit}, allIndices));
        builder.add(Arguments.of(new int[] {1 | signBit, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 2, 2 | signBit}, allIndices));
        builder.add(Arguments.of(new int[] {1, 2 | signBit, 2}, allIndices));
        builder.add(Arguments.of(new int[] {2, 1, 2 | signBit}, allIndices));
        builder.add(Arguments.of(new int[] {2, 2 | signBit, 1}, allIndices));
        builder.add(Arguments.of(new int[] {2 | signBit, 1, 2}, allIndices));
        builder.add(Arguments.of(new int[] {2 | signBit, 2, 1}, allIndices));
        builder.add(Arguments.of(new int[] {1, 1 | signBit, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[] {1 | signBit, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[] {1, 2, 2 | signBit, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {1, 2 | signBit, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2, 1, 2 | signBit, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2, 2 | signBit, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2 | signBit, 1, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[] {2 | signBit, 2, 1, 99, 98, 97}, 3));
        // Case that created an index-out-of-bound error during benchmarking.
        // The max value is first key.
        builder.add(Arguments.of(new int[] {9874, 6495, 535, 9431, 2961, 5073, 9839, 5712, 9803, 1125, 6733, 2558, 1230, 35, 7378, 1114, 7142,
            9542, 7654, 8722, 4403, 3435, 7350, 4674, 7147, 8806, 4040, 8959, 4945, 8849, 1647, 6601, 6654, 3229, 531,
            5057, 9783, 4693, 8818, 7415, 6659, 9513, 6543, 8084, 4112, 1139, 3804, 4008, 6225, 2231, 139, 6731, 4562,
            6717, 7598, 3149, 2843, 4073, 470, 3568, 9270, 6213, 9185, 34, 2084, 415, 2943, 2211, 9103, 7432, 8011,
            6210, 5058, 3934, 8889, 9359, 2303, 8148, 5808, 1885, 5769, 7043, 653, 4198, 9758, 8659, 7348, 7373, 7081,
            43, 747, 1695, 3779, 3676, 5985, 3035, 6966, 2081, 5390, 6807}, allIndices));

        return builder.build();
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
        for (final int k : keys) {
            // Same as binary search when the key is present
            // (binarySearch has exclusive upper bound)
            final int expected = Arrays.binarySearch(keys, l, r + 1, k);
            Assertions.assertEquals(expected, Partition.searchLessOrEqual(keys, l, r, k), "leq");
            Assertions.assertEquals(expected, Partition.searchGreaterOrEqual(keys, l, r, k), "geq");
        }
        // Search above/below keys
        Assertions.assertEquals(l - 1, Partition.searchLessOrEqual(keys, l, r, keys[l] - 44), "leq below");
        Assertions.assertEquals(r, Partition.searchLessOrEqual(keys, l, r, keys[r] + 44), "leq above");
        Assertions.assertEquals(l, Partition.searchGreaterOrEqual(keys, l, r, keys[l] - 44), "geq below");
        Assertions.assertEquals(r + 1, Partition.searchGreaterOrEqual(keys, l, r, keys[r] + 44), "geq above");
        // Search between neighbour keys
        for (int i = 1; i < keys.length; i++) {
            // Bound: keys[i-1] < k < keys[i]
            final int k1 = keys[i - 1];
            final int k2 = keys[i] - 1;
            for (int k = k1 + 1; k <= k2; k++) {
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
        builder.add(Arguments.of(new double[] {3, 4, 10, 12, 5, 6}, 2, 3, 4, new int[] {5, 5, 5}));
        // Test greater-than fast-forward bounds check - all values are > the pivots
        builder.add(Arguments.of(new double[] {3, 4, 1, 2, 5, 6}, 2, 3, 0, new int[] {1, 1, 1}));
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
            Partition.HEAPSELECT_MASK_SHIFT);
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
        // Note that recursion only occurs down to a small length which are finished with a sort.
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
        final int length = 10000023;
        final int range = 0;

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
}
