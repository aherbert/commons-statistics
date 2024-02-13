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
import java.util.Formatter;
import java.util.stream.Stream;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link DualPivotingStrategy}.
 */
class DualPivotingStrategyTest {
    @ParameterizedTest
    @MethodSource
    void testMedians(double[] a) {
        assertPivots(a, DualPivotingStrategy.MEDIANS);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5", "testSort5IsSorted"})
    void testSort5(double[] a) {
        assertPivots(a, DualPivotingStrategy.SORT_5);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5", "testSort5IsSorted"})
    void testSort5B(double[] a) {
        assertPivots(a, DualPivotingStrategy.SORT_5B);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5", "testSort5IsSorted"})
    void testSort5C(double[] a) {
        assertPivots(a, DualPivotingStrategy.SORT_5C);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort5J(double[] a) {
        // Does not work well for small length
        Assumptions.assumeTrue(a.length > 40);
        assertPivots(a, DualPivotingStrategy.SORT_5J);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort5of3(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 14);
        assertPivots(a, DualPivotingStrategy.SORT_5_OF_3);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort4of3(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 11);
        assertPivots(a, DualPivotingStrategy.SORT_4_OF_3);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort3of3(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 8);
        assertPivots(a, DualPivotingStrategy.SORT_3_OF_3);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort5of5(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 24);
        assertPivots(a, DualPivotingStrategy.SORT_5_OF_5);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort7(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 6);
        assertPivots(a, DualPivotingStrategy.SORT_7);
    }

    @ParameterizedTest
    @MethodSource(value = {"testSort5IsSorted"})
    void testSort8(double[] a) {
        // Does not work for small length
        Assumptions.assumeTrue(a.length > 7);
        assertPivots(a, DualPivotingStrategy.SORT_8);
    }

    private static void assertPivots(double[] a, DualPivotingStrategy s) {
        final double[] copy = a.clone();
        final int[] k = s.getSampledIndices(0, a.length - 1);
        // Extract data
        final double[] x = new double[k.length];
        for (int i = 0; i < k.length; i++) {
            x[i] = a[k[i]];
        }
        final int[] pivot2 = {-1};
        final int p1 = s.pivotIndex(a, 0, a.length - 1, pivot2);
        final int p2 = pivot2[0];
        Assertions.assertTrue(a[p1] <= a[p2], "pivots not sorted");
        // Extract data after
        final double[] y = new double[k.length];
        for (int i = 0; i < k.length; i++) {
            y[i] = a[k[i]];
        }
        // Test the effect on the data
        final int effect = s.samplingEffect();
        if (effect == DualPivotingStrategy.SORT) {
            Arrays.sort(x);
            Assertions.assertArrayEquals(x, y, "Data at indices not sorted");
        } else if (effect == DualPivotingStrategy.UNCHANGED) {
            Assertions.assertArrayEquals(x, y, "Data at indices changed");
        } else if (effect == DualPivotingStrategy.PARTIAL_SORT) {
            Arrays.sort(x);
            Arrays.sort(y);
            Assertions.assertArrayEquals(x, y, "Data destroyed");
        }
        // Flip data, pivot values should be the same
        for (int i = 0, j = k.length - 1; i < j; i++, j--) {
            final double v = copy[k[i]];
            copy[k[i]] = copy[k[j]];
            copy[k[j]] = v;
        }
        final int p1a = s.pivotIndex(copy, 0, a.length - 1, pivot2);
        final int p2a = pivot2[0];
        Assertions.assertEquals(a[p1], copy[p1a], "Pivot 1 changed");
        Assertions.assertEquals(a[p2], copy[p2a], "Pivot 2 changed");
    }

    @Test
    void testMediansIndexing() {
        assertIndexing(DualPivotingStrategy.MEDIANS, 2);
    }

    @Test
    void testSort5Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_5, 5);
    }

    @Test
    void testSort5BIndexing() {
        assertIndexing(DualPivotingStrategy.SORT_5B, 5);
    }

    @Test
    void testSort5CIndexing() {
        assertIndexing(DualPivotingStrategy.SORT_5C, 5);
    }

    @Test
    void testSort5JIndexing() {
        assertIndexing(DualPivotingStrategy.SORT_5J, 4);
    }

    @Test
    void testSort5of3Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_5_OF_3, 15);
    }

    @Test
    void testSort4of3Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_4_OF_3, 12);
    }

    @Test
    void testSort3of3Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_3_OF_3, 9);
    }

    @Test
    void testSort5of5Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_5_OF_5, 25);
    }

    @Test
    void testSort7Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_7, 7);
    }

    @Test
    void testSort8Indexing() {
        assertIndexing(DualPivotingStrategy.SORT_8, 8);
    }

    private static void assertIndexing(DualPivotingStrategy s, int safeLength) {
        final int[] pivot2 = {-1};
        final double[] a = new double[safeLength - 1];
        Assertions.assertThrows(ArrayIndexOutOfBoundsException.class, () -> s.pivotIndex(a, 0, a.length - 1, pivot2),
            () -> "Length: " + (safeLength - 1));
        for (int i = safeLength; i < 50; i++) {
            final int n = i;
            final double[] b = new double[i];
            Assertions.assertDoesNotThrow(() -> s.pivotIndex(b, 0, b.length - 1, pivot2), () -> "Length: " + n);
        }
    }

    static Stream<double[]> testMedians() {
        final Stream.Builder<double[]> builder = Stream.builder();
        // Require length 2.
        builder.add(new double[] {42.0, 46.0});
        builder.add(new double[] {42.0, 46.0, 49.0});
        builder.add(new double[] {-3.0, -46.0, -2.0});
        builder.add(new double[] {-3.0, -46.0, -2.0, 8.0});
        builder.add(new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        return builder.build();
    }

    static Stream<double[]> testSort5() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final double[] a = new double[5];
        // Permutations is 5! = 120
        final int shift = 42;
        for (int i = 0; i < 5; i++) {
            a[0] = i + shift;
            for (int j = 0; j < 5; j++) {
                if (j == i) {
                    continue;
                }
                a[1] = j + shift;
                for (int k = 0; k < 5; k++) {
                    if (k == j || k == i) {
                        continue;
                    }
                    a[2] = k + shift;
                    for (int l = 0; l < 5; l++) {
                        if (l == k || l == j || l == i) {
                            continue;
                        }
                        a[3] = l + shift;
                        for (int m = 0; m < 5; m++) {
                            if (m == l || m == k || m == j || m == i) {
                                continue;
                            }
                            a[4] = m + shift;
                            builder.add(a.clone());
                        }
                    }
                }
            }
        }
        return builder.build();
    }

    static Stream<double[]> testSort5IsSorted() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        for (int n = 8; n < 256; n *= 2) {
            for (int i = 0; i < 10; i++) {
                final int length = rng.nextInt(n, n * 2);
                builder.add(rng.doubles(length).toArray());
            }
        }
        return builder.build();
    }

    /**
     * This is not a test. It creates data and runs the pivoting strategy.
     * The true locations of the pivots are discovered in the data and this
     * printed to file.
     */
    @ParameterizedTest
    @MethodSource
    @Disabled("Used for testing")
    void testDistribution(DualPivotingStrategy ps, int n, int samples) {
        final String dir = System.getProperty("java.io.tmpdir");
        final Path path = Path.of(dir, String.format("%s_%d_%d.txt", ps, n, samples));
        final DescriptiveStatistics[] s = new DescriptiveStatistics[] {
            new DescriptiveStatistics(), new DescriptiveStatistics(), new DescriptiveStatistics()
        };
        try (BufferedWriter bw = Files.newBufferedWriter(path);
            Formatter f = new Formatter(bw)) {
            final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
            final double[] data = new double[n];
            final int[] pivot2 = {0};
            final int right = n - 1;
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < n; j++) {
                    data[j] = rng.nextInt();
                }
                //TestUtils.shuffle(rng, data);
                final int pivot1 = ps.pivotIndex(data, 0, right, pivot2);
                // Find pivot locations in the array
                int lo = 0;
                int hi = 0;
                final double p1 = data[pivot1];
                final double p2 = data[pivot2[0]];
                for (final double x : data) {
                    if (x < p1) {
                        lo++;
                    }
                    if (x > p2) {
                        hi++;
                    }
                }
                final double third1 = (double) lo / n;
                final double third3 = (double) hi / n;
                final double third2 = 1 - third1 - third3;
                f.format("%s %s %s%n", third1, third2, third3);
                s[0].addValue(third1);
                s[1].addValue(third2);
                s[2].addValue(third3);
            }
            TestUtils.printf("%s   n=%d   len=%d%n", ps, samples, n);
            TestUtils.printf("     * %8s %8s %8s %8s %8s %8s%n", "min", "max", "mean", "sd", "median", "skew");
            for (DescriptiveStatistics d : s) {
                TestUtils.printf("     * %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f%n",
                    d.getMin(), d.getMax(), d.getMean(),
                    d.getStandardDeviation(), d.getPercentile(0.5), d.getSkewness());
            }
        } catch (final IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static Stream<Arguments> testDistribution() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        //builder.add(Arguments.of(DualPivotingStrategy.MEDIANS, 10000, 10000));
        builder.add(Arguments.of(DualPivotingStrategy.SORT_5, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5B, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5C, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5_OF_3, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_4_OF_3, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_3_OF_3, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5_OF_5, 10000, 10000));
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_7, 10000, 10000));
        builder.add(Arguments.of(DualPivotingStrategy.SORT_8, 10000, 10000));

        // On small data the sort 5 method has skewed density.
        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5, 30, 1000000));
        //builder.add(Arguments.of(DualPivotingStrategy.MEDIANS, 30, 1000000));

        //builder.add(Arguments.of(DualPivotingStrategy.SORT_5, 1000, 100000));
        return builder.build();
    }

    @Test
    @Disabled("Used for testing")
    void testPrintSort5() {
        for (int i = 5; i < 100; i++) {
            // Copy logic for indices: 1/6, 1/3, 1/2, 2/3, 5/6
            final int len = i - 1;
            final int sixth = 1 + (len >>> 3) + (len >>> 5);
            final int p3 = len >>> 1;
            final int p2 = p3 - sixth;
            final int p1 = p2 - sixth;
            final int p4 = p3 + sixth;
            final int p5 = p4 + sixth;
            final double d = i;
            TestUtils.printf("%2d : %2d %2d %2d %2d %2d : %.3f %.3f %.3f %.3f %.3f%n",
                i, p1, p2, p3, p4, p5,
                (p1 + 1) / d, (p2 + 1) / d, (p3 + 1) / d, (p4 + 1) / d, (p5 + 1) / d);
        }
    }
}
