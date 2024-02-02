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
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Partition}.
 */
class PartitionTest {
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
        final int[] upper = {0};
        for (int k = from; k <= to; k++) {
            final int target = k;
            double[] x = values.clone();
            int lower = Partition.partitionMinK(x, from, to, k, 0, upper);
            Assertions.assertTrue(lower <= k && upper[0] >= k);
            for (int i = lower; i <= upper[0]; i++) {
                Assertions.assertEquals(sorted[i], x[i], () -> Integer.toString(target));
            }
            // Check the data is the same
            Arrays.sort(x, from, to + 1);
            Assertions.assertArrayEquals(sorted, x, "Data destroyed");
            if (k > from) {
                // Sort an extra 1
                x = values.clone();
                lower = Partition.partitionMinK(x, from, to, k, 1, upper);
                Assertions.assertTrue(lower <= k - 1 && upper[0] >= k);
                for (int i = lower; i <= upper[0]; i++) {
                    Assertions.assertEquals(sorted[i], x[i], () -> (target - 1) + " to " + target);
                }
                // Check the data is the same
                Arrays.sort(x, from, to + 1);
                Assertions.assertArrayEquals(sorted, x, "Data destroyed");
                if (k > from + 1) {
                    // Sort all
                    x = values.clone();
                    lower =  Partition.partitionMinK(x, from, to, k, k - from, upper);
                    Assertions.assertTrue(lower == from && upper[0] >= k);
                    for (int i = lower; i <= upper[0]; i++) {
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
        final int[] upper = {0};
        for (int k = from; k <= to; k++) {
            final int target = k;
            double[] x = values.clone();
            int lower = Partition.partitionMaxK(x, from, to, k, 0, upper);
            Assertions.assertTrue(lower <= k && upper[0] >= k);
            for (int i = lower; i <= upper[0]; i++) {
                Assertions.assertEquals(sorted[i], x[i], () -> Integer.toString(target));
            }
            // Check the data is the same
            Arrays.sort(x, from, to + 1);
            Assertions.assertArrayEquals(sorted, x, "Data destroyed");
            if (k < to) {
                // Sort an extra 1
                x = values.clone();
                lower = Partition.partitionMaxK(x, from, to, k, 1, upper);
                Assertions.assertTrue(lower <= k && upper[0] >= k + 1);
                for (int i = lower; i <= upper[0]; i++) {
                    Assertions.assertEquals(sorted[i], x[i], () -> target + " to " + (target + 1));
                }
                // Check the data is the same
                Arrays.sort(x, from, to + 1);
                Assertions.assertArrayEquals(sorted, x, "Data destroyed");
                if (k < to - 1) {
                    // Sort all
                    x = values.clone();
                    lower = Partition.partitionMaxK(x, from, to, k, to - k, upper);
                    Assertions.assertTrue(lower <= k && upper[0] == to);
                    for (int i = lower; i <= upper[0]; i++) {
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
        return builder.build();
    }

//    @ParameterizedTest
//    @MethodSource
//    void testSelect(double[] values) {
//        final double[] sorted = values.clone();
//        Arrays.sort(sorted);
//        final Partition selector = new Partition();
//        final double[] kp1 = new double[1];
//        for (int i = 0; i < sorted.length; i++) {
//            final int k = i;
//            double[] x = values.clone();
//            Assertions.assertEquals(sorted[k], selector.selectSP(x, k, null), () -> "k[" + k + "]");
//            Arrays.sort(x);
//            Assertions.assertArrayEquals(sorted, x, () -> "Data destroyed: k[" + k + "]");
//            if (k + 1 < sorted.length) {
//                x = values.clone();
//                Assertions.assertEquals(sorted[k], selector.selectSP(x, k, kp1), () -> "k[" + k + "] with k+1");
//                Assertions.assertEquals(sorted[k + 1], kp1[0], () -> "k+1[" + (k + 1) + "]");
//                Arrays.sort(x);
//                Assertions.assertArrayEquals(sorted, x, () -> "Data destroyed: k[" + k + "] with k+1");
//            }
//        }
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSelect"})
//    void testSelectSPN(double[] values) {
//        final double[] sorted = values.clone();
//        Arrays.sort(sorted);
//        final Partition selector = new Partition();
//        final double[] kp1 = new double[1];
//        for (int i = 0; i < sorted.length; i++) {
//            final int k = i;
//            double[] x = values.clone();
//            Assertions.assertEquals(sorted[k], selector.selectSPN(x, k, null), () -> "k[" + k + "]");
//            Arrays.sort(x);
//            Assertions.assertArrayEquals(sorted, x, () -> "Data destroyed: k[" + k + "]");
//            if (k + 1 < sorted.length) {
//                x = values.clone();
//                Assertions.assertEquals(sorted[k], selector.selectSPN(x, k, kp1), () -> "k[" + k + "] with k+1");
//                Assertions.assertEquals(sorted[k + 1], kp1[0], () -> "k+1[" + (k + 1) + "]");
//                Arrays.sort(x);
//                Assertions.assertArrayEquals(sorted, x, () -> "Data destroyed: k[" + k + "] with k+1");
//            }
//        }
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSelect"})
//    void testSelectSPWithHeap(double[] values) {
//        final double[] sorted = values.clone();
//        Arrays.sort(sorted);
//        final Partition selector = new Partition();
//        final double[] kp1 = new double[1];
//        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
//        final int[] indices = IntStream.range(0, sorted.length).toArray();
//        for (int n = 0; n < 3; n++) {
//            TestUtils.shuffle(rng, indices);
//            final double[] x = values.clone();
//            final int[] pivotsHeap = Partition.createPivotsHeap(sorted.length);
//            for (int i = 0; i < sorted.length; i++) {
//                final int k = indices[i];
//                Assertions.assertEquals(sorted[k], selector.selectSPH(x, pivotsHeap, k, null), () -> "k[" + k + "]");
//                if (k + 1 < sorted.length) {
//                    Assertions.assertEquals(sorted[k], selector.selectSPH(x, pivotsHeap, k, kp1), () -> "k[" + k + "] with k+1");
//                    Assertions.assertEquals(sorted[k + 1], kp1[0], () -> "k+1[" + (k + 1) + "]");
//                }
//            }
//            Arrays.sort(x);
//            Assertions.assertArrayEquals(sorted, x, "Data destroyed");
//        }
//    }
//
//    static Stream<double[]> testSelect() {
//        final Stream.Builder<double[]> builder = Stream.builder();
//        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
//        // Sizes above and below the threshold for partitioning
//        for (final int size : new int[] {5, 50}) {
//            final double[] values = IntStream.range(0, size).asDoubleStream().toArray();
//            final double[] zeros = values.clone();
//            final double[] nans = values.clone();
//            Arrays.fill(zeros, 0, size >>> 2, -0.0);
//            Arrays.fill(zeros, size >>> 2, size >>> 1, 0.0);
//            Arrays.fill(nans, 0, 2, Double.NaN);
//            for (int i = 0; i < 25; i++) {
//                builder.add(TestUtils.shuffle(rng, values.clone()));
//                builder.add(TestUtils.shuffle(rng, zeros.clone()));
//            }
//            for (int i = 0; i < 5; i++) {
//                builder.add(TestUtils.shuffle(rng, nans.clone()));
//            }
//        }
//        return builder.build();
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionSP(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionSP);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionSPN(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionSPN);
//    }
//
    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMIndexSet(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(KeyStrategy.INDEX_SET)::partitionSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMPivotCache(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(KeyStrategy.PIVOT_CACHE)::partitionSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionSBMSequential(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(KeyStrategy.SEQUENTIAL)::partitionSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionKSBM(double[] values, int[] indices) {
        assertPartition(values, indices, new Partition(KeyStrategy.PIVOT_CACHE)::partitionKSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionPairedSBM(double[] values, int[] indices) {
        assertPartitionPaired(values, indices, new Partition(KeyStrategy.INDEX_SET)::partitionPairedSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testPartition"})
    void testPartitionPairedSBMPivotCache(double[] values, int[] indices) {
        assertPartitionPaired(values, indices, new Partition(KeyStrategy.PIVOT_CACHE)::partitionPairedSBM);
    }

//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionBM(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionBM);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionDP(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionDP);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionDP5(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionDP5);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testPartition"})
//    void testPartitionDNF(double[] values, int[] indices) {
//        assertPartition(values, indices, new Partition()::partitionDNF);
//    }
//
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
                    () -> j + " < " + k);
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
            new Partition(PivotingStrategy.DYNAMIC, 3, KeyStrategy.INDEX_SET)::sortRangeSBM);
    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSort"})
//    void testSortSP(double[] values) {
//        assertSort(values, new Partition()::sortSP);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSort"})
//    void testSortBM(double[] values) {
//        assertSort(values, new Partition()::sortBM);
//    }
//
    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testSortSBM(double[] values) {
        assertSort(values,
            new Partition(PivotingStrategy.DYNAMIC, 3, KeyStrategy.INDEX_SET)::sortSBM);
    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSort"})
//    void testSortDP(double[] values) {
//        assertSort(values, new Partition(PivotingStrategy.DYNAMIC, 3)::sortDP);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSort"})
//    void testSortDP5(double[] values) {
//        // Requires at least 5 points
//        assertSort(values, new Partition(PivotingStrategy.DYNAMIC, 5)::sortDP5);
//    }
//
//    @ParameterizedTest
//    @MethodSource(value = {"testSort"})
//    void testSortDNF(double[] values) {
//        assertSort(values, new Partition()::sortDNF);
//    }
//
    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testInsertionSort(double[] values) {
        // Cannot handle NaN or signed zeros
        Assumptions.assumeFalse(Arrays.stream(values)
            .filter(x -> x == 0 || Double.isNaN(x)).findAny().isPresent());
        assertSort(values, x -> Sorting.sort(x, 0, x.length - 1, false));
        if (values.length < 2) {
            return;
        }
        // Check internal sort
        // Set pivot at lower end
        values[0] = Arrays.stream(values).min().getAsDouble();
        // check internal sort
        assertSort(values, x -> Sorting.sort(x, 1, x.length - 1, true));
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort"})
    void testInsertionSort5(double[] values) {
        // Cannot handle NaN or signed zeros
        Assumptions.assumeFalse(Arrays.stream(values)
            .filter(x -> x == 0 || Double.isNaN(x)).findAny().isPresent());
        final double[] data = Arrays.copyOf(values, 5);
        assertSort(data, x -> Sorting.sort5(x, 0, 1, 2, 3, 4));
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
        builder.add(Arguments.of(new int[]{1, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 1, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[]{1, 1, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{1, 2}, allIndices));
        builder.add(Arguments.of(new int[]{2, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 2, 3}, allIndices));
        builder.add(Arguments.of(new int[]{1, 3, 2}, allIndices));
        builder.add(Arguments.of(new int[]{2, 1, 3}, allIndices));
        builder.add(Arguments.of(new int[]{2, 3, 1}, allIndices));
        builder.add(Arguments.of(new int[]{3, 1, 2}, allIndices));
        builder.add(Arguments.of(new int[]{3, 2, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 2, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[]{2, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[]{1, 2, 3, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{1, 3, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2, 1, 3, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2, 3, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{3, 1, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{3, 2, 1, 99, 98, 97}, 3));
        // Paired keys. Replace highest value from above with the next value and a sign bit.
        builder.add(Arguments.of(new int[]{1, 1 | signBit}, allIndices));
        builder.add(Arguments.of(new int[]{1 | signBit, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 2, 2 | signBit}, allIndices));
        builder.add(Arguments.of(new int[]{1, 2 | signBit, 2}, allIndices));
        builder.add(Arguments.of(new int[]{2, 1, 2 | signBit}, allIndices));
        builder.add(Arguments.of(new int[]{2, 2 | signBit, 1}, allIndices));
        builder.add(Arguments.of(new int[]{2 | signBit, 1, 2}, allIndices));
        builder.add(Arguments.of(new int[]{2 | signBit, 2, 1}, allIndices));
        builder.add(Arguments.of(new int[]{1, 1 | signBit, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[]{1 | signBit, 1, 99, 98, 97}, 2));
        builder.add(Arguments.of(new int[]{1, 2, 2 | signBit, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{1, 2 | signBit, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2, 1, 2 | signBit, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2, 2 | signBit, 1, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2 | signBit, 1, 2, 99, 98, 97}, 3));
        builder.add(Arguments.of(new int[]{2 | signBit, 2, 1, 99, 98, 97}, 3));
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
}
