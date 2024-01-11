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
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.PermutationSampler;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Sorting}.
 */
class SortingTest {

    // double[]

    @ParameterizedTest
    @MethodSource(value = {"testDoubleSort"})
    void testDoubleInsertionSort(double[] values) {
        assertDoubleSort(values, x -> Sorting.sort(x, 0, x.length - 1, false));
        if (values.length < 2) {
            return;
        }
        // Check internal sort
        // Set pivot at lower end
        values[0] = Arrays.stream(values).min().getAsDouble();
        // check internal sort
        assertDoubleSort(values, x -> Sorting.sort(x, 1, x.length - 1, true));
    }

    @ParameterizedTest
    @MethodSource(value = {"testDoubleSort"})
    void testDoubleInsertionSort3(double[] values) {
        final double[] data = Arrays.copyOf(values, 3);
        assertDoubleSort(data, x -> Sorting.sort3(x, 0, 1, 2));
    }

    @ParameterizedTest
    @MethodSource(value = {"testDoubleSort"})
    void testDoubleInsertionSort5(double[] values) {
        final double[] data = Arrays.copyOf(values, 5);
        assertDoubleSort(data, x -> Sorting.sort5(x, 0, 1, 2, 3, 4));
    }

    @ParameterizedTest
    @MethodSource(value = {"testDoubleInsertionSort3Internal"})
    void testDoubleInsertionSort3Internal(double[] values, int[] indices) {
        final int a = indices[0];
        final int b = indices[1];
        final int c = indices[2];
        assertDoubleSortInternal(values, x -> Sorting.sort3(x, a, b, c), indices);
    }

    @ParameterizedTest
    @MethodSource(value = {"testDoubleInsertionSort5Internal"})
    void testDoubleInsertionSort5Internal(double[] values, int[] indices) {
        final int a = indices[0];
        final int b = indices[1];
        final int c = indices[2];
        final int d = indices[3];
        final int e = indices[4];
        assertDoubleSortInternal(values, x -> Sorting.sort5(x, a, b, c, d, e), indices);
    }

    /**
     * Assert that the sort {@code function} computes the same result as
     * {@link Arrays#sort(double[])}. Ignores signed zeros.
     *
     * @param values Data.
     * @param function Sort function.
     */
    private static void assertDoubleSort(double[] values, Consumer<double[]> function) {
        final double[] expected = values.clone();
        Arrays.sort(expected);
        final double[] actual = values.clone();
        function.accept(actual);
        assertDoubleSort(expected, actual);
    }

    /**
     * Assert that the {@code expected} and {@code actual} sort are the same. Ignores
     * signed zeros.
     *
     * @param expected Expected sort.
     * @param actual Actual sort.
     */
    private static void assertDoubleSort(double[] expected, double[] actual) {
        // Detect signed zeros
        int c = 0;
        for (int i = 0; i < expected.length; i++) {
            if (Double.compare(-0.0, expected[i]) == 0) {
                c++;
            }
        }
        // Check
        if (c != 0) {
            // Replace signed zeros
            final double[] e = replaceSignedZeros(expected.clone());
            final double[] a = replaceSignedZeros(actual.clone());
            Assertions.assertArrayEquals(e, a, "Sort with +0.0");
            // Sort the signed zeros correctly
            Arrays.sort(actual);
            // Check the same number of signed zeros are present
            Assertions.assertArrayEquals(expected, actual, "Signed zeros destroyed");
        } else {
            Assertions.assertArrayEquals(expected, actual, "Invalid sort");
        }
    }

    /**
     * Assert that the sort {@code function} computes the same result as
     * {@link Arrays#sort(double[])} run on the provided {@code indices}. Ignores signed
     * zeros.
     *
     * @param values Data.
     * @param function Sort function.
     * @param indices Indices.
     */
    private static void assertDoubleSortInternal(double[] values, Consumer<double[]> function, int... indices) {
        Assertions.assertFalse(containsDuplicates(indices), () -> "Duplicate indices: " + Arrays.toString(indices));
        // Pick out the data to sort
        final double[] expected = extractIndices(values, indices);
        Arrays.sort(expected);
        final double[] data = values.clone();
        function.accept(data);
        // Pick out the data that was sorted
        final double[] actual = extractIndices(data, indices);
        assertDoubleSort(expected, actual);
        // Check outside the sorted indices
        OUTSIDE: for (int i = 0; i < values.length; i++) {
            for (final int ignore : indices) {
                if (i == ignore) {
                    continue OUTSIDE;
                }
            }
            Assertions.assertEquals(values[i], data[i]);
        }
    }

    static Stream<double[]> testDoubleSort() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        for (final int size : new int[] {5, 10}) {
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
                // Pick a different index
                final int k = (j + rng.nextInt(size - 1)) % size;
                a[j] = -0.0;
                a[k] = 0.0;
                builder.add(a.clone());
                for (int z = 0; z < size; z++) {
                    a[z] = rng.nextBoolean() ? -0.0 : 0.0;
                }
            }
        }
        return builder.build();
    }

    static Stream<Arguments> testDoubleInsertionSort3Internal() {
        return testDoubleInsertionSortInternal(3);
    }

    static Stream<Arguments> testDoubleInsertionSort5Internal() {
        return testDoubleInsertionSortInternal(5);
    }

    static Stream<Arguments> testDoubleInsertionSortInternal(int k) {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        for (final int size : new int[] {k, 2 * k, 4 * k}) {
            double[] a = rng.doubles(size).toArray();
            final PermutationSampler s = new PermutationSampler(rng, size, k);
            for (int i = 0; i < 5; i++) {
                a = rng.doubles(size).toArray();
                final int[] indices = s.sample();
                builder.add(Arguments.of(a.clone(), indices));
                a[indices[0]] = -0.0;
                a[indices[1]] = 0.0;
                builder.add(Arguments.of(a.clone(), indices));
                for (final int z : indices) {
                    a[z] = rng.nextBoolean() ? -0.0 : 0.0;
                }
                builder.add(Arguments.of(a.clone(), indices));
            }
        }
        return builder.build();
    }

    private static double[] extractIndices(double[] values, int[] indices) {
        final double[] data = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            data[i] = values[indices[i]];
        }
        return data;
    }

    // int[]

    @ParameterizedTest
    @MethodSource(value = {"testIntSort"})
    void testIntInsertionSort(int[] values) {
        assertIntSort(values, x -> Sorting.sort(x, 0, x.length - 1, false));
        if (values.length < 2) {
            return;
        }
        // Check internal sort
        // Set pivot at lower end
        values[0] = Arrays.stream(values).min().getAsInt();
        // check internal sort
        assertIntSort(values, x -> Sorting.sort(x, 1, x.length - 1, true));
    }

    @ParameterizedTest
    @MethodSource(value = {"testIntSort"})
    void testIntInsertionSort3(int[] values) {
        final int[] data = Arrays.copyOf(values, 3);
        assertIntSort(data, x -> Sorting.sort3(x, 0, 1, 2));
    }

    @ParameterizedTest
    @MethodSource(value = {"testIntSort"})
    void testIntInsertionSort5(int[] values) {
        final int[] data = Arrays.copyOf(values, 5);
        assertIntSort(data, x -> Sorting.sort5(x, 0, 1, 2, 3, 4));
    }

    @ParameterizedTest
    @MethodSource(value = {"testIntInsertionSort3Internal"})
    void testIntInsertionSort3Internal(int[] values, int[] indices) {
        final int a = indices[0];
        final int b = indices[1];
        final int c = indices[2];
        assertIntSortInternal(values, x -> Sorting.sort3(x, a, b, c), indices);
    }

    @ParameterizedTest
    @MethodSource(value = {"testIntInsertionSort5Internal"})
    void testIntInsertionSort5Internal(int[] values, int[] indices) {
        final int a = indices[0];
        final int b = indices[1];
        final int c = indices[2];
        final int d = indices[3];
        final int e = indices[4];
        assertIntSortInternal(values, x -> Sorting.sort5(x, a, b, c, d, e), indices);
    }

    /**
     * Assert that the sort {@code function} computes the same result as
     * {@link Arrays#sort(int[])}. Ignores signed zeros.
     *
     * @param values Data.
     * @param function Sort function.
     */
    private static void assertIntSort(int[] values, Consumer<int[]> function) {
        final int[] expected = values.clone();
        Arrays.sort(expected);
        final int[] actual = values.clone();
        function.accept(actual);
        assertIntSort(expected, actual);
    }

    /**
     * Assert that the {@code expected} and {@code actual} sort are the same. Ignores
     * signed zeros.
     *
     * @param expected Expected sort.
     * @param actual Actual sort.
     */
    private static void assertIntSort(int[] expected, int[] actual) {
        Assertions.assertArrayEquals(expected, actual, "Invalid sort");
    }

    /**
     * Assert that the sort {@code function} computes the same result as
     * {@link Arrays#sort(int[])} run on the provided {@code indices}. Ignores signed
     * zeros.
     *
     * @param values Data.
     * @param function Sort function.
     * @param indices Indices.
     */
    private static void assertIntSortInternal(int[] values, Consumer<int[]> function, int... indices) {
        Assertions.assertFalse(containsDuplicates(indices), () -> "Duplicate indices: " + Arrays.toString(indices));
        // Pick out the data to sort
        final int[] expected = extractIndices(values, indices);
        Arrays.sort(expected);
        final int[] data = values.clone();
        function.accept(data);
        // Pick out the data that was sorted
        final int[] actual = extractIndices(data, indices);
        assertIntSort(expected, actual);
        // Check outside the sorted indices
        OUTSIDE: for (int i = 0; i < values.length; i++) {
            for (final int ignore : indices) {
                if (i == ignore) {
                    continue OUTSIDE;
                }
            }
            Assertions.assertEquals(values[i], data[i]);
        }
    }

    static Stream<int[]> testIntSort() {
        final Stream.Builder<int[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        for (final int size : new int[] {5, 10}) {
            int[] a = new int[size];
            Arrays.fill(a, 42);
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
                a = rng.ints(size).toArray();
                builder.add(a.clone());
            }
        }
        return builder.build();
    }

    static Stream<Arguments> testIntInsertionSort3Internal() {
        return testIntInsertionSortInternal(3);
    }

    static Stream<Arguments> testIntInsertionSort5Internal() {
        return testIntInsertionSortInternal(5);
    }

    static Stream<Arguments> testIntInsertionSortInternal(int k) {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        for (final int size : new int[] {k, 2 * k, 4 * k}) {
            int[] a = rng.ints(size).toArray();
            final PermutationSampler s = new PermutationSampler(rng, size, k);
            for (int i = 0; i < 5; i++) {
                a = rng.ints(size).toArray();
                final int[] indices = s.sample();
                builder.add(Arguments.of(a.clone(), indices));
            }
        }
        return builder.build();
    }

    private static int[] extractIndices(int[] values, int[] indices) {
        final int[] data = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            data[i] = values[indices[i]];
        }
        return data;
    }

    // Sorting unique indices

    @ParameterizedTest
    @MethodSource(value = {"testSortUnique"})
    void testSortUniqueArray(int[] values, int n) {
        assertSortUnique(values.length, values, n < 0 ? values.length : n);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSortUnique"})
    void testSortUniqueIndexSet(int[] values, int n) {
        assertSortUnique(0, values, n < 0 ? values.length : n);
    }

    private static void assertSortUnique(int threshold, int[] values, int n) {
        final int[] x = values.clone();
        final int[] expected = Arrays.stream(values).limit(n)
            .distinct().sorted().toArray();
        final IndexSet set = Sorting.sortUnique(threshold, x, n);
        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], x[i]);
        }
        if (n > 0) {
            final int end = expected.length - 1;
            final int max = x[n - 1];
            if (expected.length < n) {
                Assertions.assertEquals(expected[end], ~max, "twos-complement max value");
            } else {
                Assertions.assertEquals(expected[end], max, "max value");
            }
        }
        for (int i = expected.length; i < n; i++) {
            Assertions.assertTrue(x[i] < 0, "Duplicate not set to negative");
        }

        if (x.length <= threshold) {
            Assertions.assertNull(set);
        } else if (n > 1) {
            // Check the IndexSet contains all the indices
            final int[] a = new int[expected.length];
            final int[] c = {0};
            set.forEach(i -> a[c[0]++] = i);
            Assertions.assertArrayEquals(expected, a);
        }
    }

    static Stream<Arguments> testSortUnique() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Use length -1 to use the array length
        builder.add(Arguments.of(new int[0], -1));
        builder.add(Arguments.of(new int[3], -1));
        builder.add(Arguments.of(new int[3], -1));
        builder.add(Arguments.of(new int[] {1, 2, 3}, -1));
        builder.add(Arguments.of(new int[] {1, 1, 1}, -1));
        builder.add(Arguments.of(new int[] {42}, -1));
        builder.add(Arguments.of(new int[] {42, 5, 7}, -1));
        builder.add(Arguments.of(new int[] {42, 5, 7, 7, 4}, -1));
        // Truncated indices
        builder.add(Arguments.of(new int[] {42, 5, 7, 7, 4}, 3));
        return builder.build();
    }

    // Helper methods

    private static boolean containsDuplicates(int[] indices) {
        for (int i = 0; i < indices.length; i++) {
            for (int j = 0; j < i; j++) {
                if (indices[i] == indices[j]) {
                    return true;
                }
            }
        }
        return false;
    }

    private static double[] extractNonduplicateIndices(double[] values, int[] indices) {
        final double[] data = new double[indices.length];
        int c = 0;
        NEXT_INDEX: for (int i = 0; i < indices.length; i++) {
            for (int j = 0; j < i; j++) {
                if (indices[i] == indices[j]) {
                    continue NEXT_INDEX;
                }
            }
            data[c++] = values[indices[i]];
        }
        return Arrays.copyOf(data, c);
    }

    private static double[] replaceSignedZeros(double[] values) {
        for (int i = 0; i < values.length; i++) {
            if (Double.compare(-0.0, values[i]) == 0) {
                values[i] = 0;
            }
        }
        return values;
    }
}
