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
package org.apache.commons.statistics.inference;

import java.util.function.BiConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.statistics.ranking.RankingAlgorithm;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link MannWhitneyUTest}.
 */
class MannWhitneyUTestTest {

    private final MannWhitneyUTest testStatistic = MannWhitneyUTest.instance();

    @Test
    void testMannWhitneyUThrows() {
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
                () -> WilcoxonSignedRankTest.create(null), "ranking");

        final double[] sample1 = {1, 1};
        final double[] sample2 = {1, 1};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.mannWhitneyUTest(sample1, sample2, false, alpha), "mannWhitneyUTest");

        testMannWhitneyUInputThrows((x, y) -> testStatistic.mannWhitneyU(x, y));
        testMannWhitneyUInputThrows((x, y) -> testStatistic.mannWhitneyUTest(x, y, false));
        testMannWhitneyUInputThrows((x, y) -> testStatistic.mannWhitneyUTest(x, y, false, 0.05));
    }

    private static void testMannWhitneyUInputThrows(BiConsumer<double[], double[]> action) {
        // Samples must be present, i.e. length > 0
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {}, new double[] {1.0}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1.0}, new double[] {}), "values", "size");

        // x and y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, null));

        // x or y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, new double[] {1.0}));
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(new double[] {1.0}, null));
    }

    @Test
    void testWithRankingAlgorithm() {
        final IllegalStateException expected = new IllegalStateException("expected");
        final RankingAlgorithm ranking = new RankingAlgorithm() {
            @Override
            public double[] apply(double[] data) {
                throw expected;
            }
        };
        final MannWhitneyUTest test = MannWhitneyUTest.create(ranking);
        final double[] x = {1, 2, 3};
        final double[] y = {6, 4, 2};
        Assertions.assertSame(expected, Assertions.assertThrows(expected.getClass(), () -> test.mannWhitneyU(x, y)));
        Assertions.assertSame(expected, Assertions.assertThrows(expected.getClass(), () -> test.mannWhitneyUTest(x, y, false)));
    }

    @ParameterizedTest
    @MethodSource
    void testMannWhitneyU(double[] sample1, double[] sample2, double statistic, double p, boolean exact) {
        Assertions.assertEquals(statistic, testStatistic.mannWhitneyU(sample1, sample2), "statistic");
        final double actual = testStatistic.mannWhitneyUTest(sample1, sample2, exact);
        TestUtils.assertProbability(p, actual, 1e-14, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.mannWhitneyUTest(sample1, sample2, exact, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.mannWhitneyUTest(sample1, sample2, exact, p * 0.99), "reject");
        }
        // Check symmetry
        final double other = (long) sample1.length * sample2.length - statistic;
        Assertions.assertEquals(other, testStatistic.mannWhitneyU(sample2, sample1), "statistic (y, x)");
        Assertions.assertEquals(actual, testStatistic.mannWhitneyUTest(sample2, sample1, exact), "p-value (y, x)");
    }

    static Stream<Arguments> testMannWhitneyU() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Target values computed using R version 3.4.0

        // Exact cases: Requires no ties
        // x <- c(19, 22, 16, 29, 24)
        // y <- c(20, 11, 17, 12)
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = FALSE, exact = TRUE, correct = TRUE)
        builder.add(Arguments.of(
            new double[] {19, 22, 16, 29, 24},
            new double[] {20, 11, 17, 12},
            17, 0.1111111111111111, true));
        builder.add(Arguments.of(
                new double[] {2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18},
                new double[] {1, 3, 5, 7, 9, 11, 13, 19, 20},
                56, 0.65563229340319129, true));
        builder.add(Arguments.of(
            new double[] {2, 4, 6, 8, 10, 12, 14, 16, 18},
            new double[] {-3, -1, 1, 3, 5, 7, 9, 11},
            57, 0.046400658165364046, true));
        // Random data without ties
        // UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        // ZigguratSampler.NormalizedGaussian s = ZigguratSampler.NormalizedGaussian.of(rng);
        // for (int n = 50; n <= 150; n += 50) {
        //     double[] x = s.samples(n).map(i -> Precision.round(i, 3)).distinct().toArray();
        //     double[] y = s.samples().map(i -> Precision.round(i, 3))
        //                             .filter(i -> Arrays.stream(x).noneMatch(j -> i == j))
        //                             .limit(n+10).distinct().toArray();
        // CHECKSTYLE: stop regex
        //     System.out.println(Arrays.stream(x).mapToObj(Double::toString).collect(Collectors.joining(", ", "new double[] {", "},")));
        //     System.out.println(Arrays.stream(y).mapToObj(Double::toString).collect(Collectors.joining(", ", "new double[] {", "},")));
        // CHECKSTYLE: resume regex
        // }
        builder.add(Arguments.of(
            new double[] {1.208, -1.411, -0.507, -0.521, 0.325, 0.887, -0.543, -0.012, -2.185, 0.718, 0.659, -1.095, -0.41, 0.921, -0.442, 0.883, 2.817, 0.963, 0.452, -1.171, 1.32, -0.224, 1.88, -1.459, -0.955, 2.512, 1.147, -0.471, -1.124, 0.577, 0.362, 1.737, 0.407, 0.701, -0.302, -0.859, 0.648, 0.65, 1.869, -0.685, 0.317, -0.049, 0.155, 0.943, -1.516, -0.615, 0.663, 0.048, 1.386, -0.444},
            new double[] {1.042, -0.735, 3.151, 0.628, -1.442, 1.142, 0.834, 1.686, 0.37, 1.474, 0.975, 0.697, 1.552, 0.388, -0.408, 0.62, 0.032, 2.458, 1.723, 0.549, 1.055, 0.822, -0.549, -0.517, 2.322, 1.172, -1.63, -1.151, -1.065, -0.464, -1.188, 0.472, -2.228, -0.626, 0.521, -0.334, -0.687, -1.894, 1.217, 1.061, -0.393, 1.366, 2.217, -0.045, -0.552, 1.047, 0.138, 1.27, -0.838, 0.107, 0.555, 0.1, 0.276, -0.156, 1.11, -1.498, 0.26, -1.071},
            1380, 0.66972967418927165, true));
        builder.add(Arguments.of(
            new double[] {1.012, -1.187, 0.737, -0.465, -0.426, 0.373, -2.206, 0.102, 0.032, 1.171, 1.615, -0.167, 0.138, -0.043, -0.391, -0.318, -0.257, 0.053, 0.129, -1.385, 0.246, 0.189, 0.286, 0.26, 0.781, -1.124, -0.404, 1.364, -0.175, -0.567, 0.224, 0.075, 1.194, -0.549, 1.277, -0.337, 0.221, -0.29, -0.26, -0.904, 0.402, -0.645, 0.88, 0.497, 1.125, -0.803, -0.66, -0.082, -1.763, -0.631, -0.85, -1.661, -1.24, 2.018, -0.013, -0.272, 2.12, -0.913, 1.151, -0.759, 1.724, -0.021, -1.84, -0.417, 0.656, -0.814, -0.179, 1.282, 0.204, 1.122, 1.434, 1.293, 0.761, -0.668, -0.527, -0.712, -0.616, -2.102, -1.03, 1.138, 0.019, -1.038, -1.085, -0.579, 1.427, -1.184, 0.196, -0.145, -0.545, 0.876, -1.262, -1.833, -0.482, 0.209, -0.159, -0.163, -1.457, -0.339, -0.08, -0.459},
            new double[] {0.207, -0.381, 0.564, 1.116, 1.365, 0.417, -0.694, -1.301, 0.803, 0.238, 0.97, -1.597, 1.123, -1.296, -0.119, -0.176, -1.188, -1.303, 1.472, 0.212, 0.895, -1.919, -1.047, -0.419, 1.499, 0.033, 1.513, 0.762, 0.531, 1.071, -0.306, 0.896, 0.709, -0.363, -0.766, -0.61, 0.888, -0.435, 0.839, 1.192, -0.375, -0.861, -2.061, -0.834, -1.093, 2.483, -0.115, -2.47, 0.058, -0.187, -1.067, 0.2, 1.643, -2.492, -1.577, 0.086, 0.622, -0.658, -1.024, 2.55, 1.989, -0.767, 1.17, -0.866, 0.89, 0.979, 0.486, 1.089, 0.574, -1.013, 0.348, 0.851, 0.304, -0.711, 0.018, -0.5, 0.797, 0.017, 0.287, -0.935, 0.644, -0.857, -0.671, -0.797, -0.411, 1.048, -0.705, -1.383, -1.791, 0.142, -0.95, -0.263, -1.591, -1.36, 0.848, -0.716, -1.281, 0.659, 2.224, -0.737, -2.193, 0.79, -0.371, 0.46, -1.135, 0.733, -0.596, 0.303, 0.904},
            5348, 0.81661195397099406, true));
        // Larger samples are too slow in R

        // Inexact cases
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = FALSE, exact = FALSE, correct = TRUE)
        builder.add(Arguments.of(
            new double[] {19, 22, 16, 29, 24},
            new double[] {20, 11, 17, 12},
            17, 0.11134688653314045, false));
        builder.add(Arguments.of(
            new double[] {2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18},
            new double[] {1, 3, 5, 7, 9, 11, 13, 19, 20},
            56, 0.648503379652976, false));
        builder.add(Arguments.of(
            new double[] {2, 4, 6, 8, 10, 12, 14, 16, 18},
            new double[] {-3, -1, 1, 3, 5, 7, 9, 11},
            57, 0.048539622897320618, false));
        builder.add(Arguments.of(
            new double[] {1.208, -1.411, -0.507, -0.521, 0.325, 0.887, -0.543, -0.012, -2.185, 0.718, 0.659, -1.095, -0.41, 0.921, -0.442, 0.883, 2.817, 0.963, 0.452, -1.171, 1.32, -0.224, 1.88, -1.459, -0.955, 2.512, 1.147, -0.471, -1.124, 0.577, 0.362, 1.737, 0.407, 0.701, -0.302, -0.859, 0.648, 0.65, 1.869, -0.685, 0.317, -0.049, 0.155, 0.943, -1.516, -0.615, 0.663, 0.048, 1.386, -0.444},
            new double[] {1.042, -0.735, 3.151, 0.628, -1.442, 1.142, 0.834, 1.686, 0.37, 1.474, 0.975, 0.697, 1.552, 0.388, -0.408, 0.62, 0.032, 2.458, 1.723, 0.549, 1.055, 0.822, -0.549, -0.517, 2.322, 1.172, -1.63, -1.151, -1.065, -0.464, -1.188, 0.472, -2.228, -0.626, 0.521, -0.334, -0.687, -1.894, 1.217, 1.061, -0.393, 1.366, 2.217, -0.045, -0.552, 1.047, 0.138, 1.27, -0.838, 0.107, 0.555, 0.1, 0.276, -0.156, 1.11, -1.498, 0.26, -1.071},
            1380, 0.66849366084453488, false));
        builder.add(Arguments.of(
            new double[] {1.012, -1.187, 0.737, -0.465, -0.426, 0.373, -2.206, 0.102, 0.032, 1.171, 1.615, -0.167, 0.138, -0.043, -0.391, -0.318, -0.257, 0.053, 0.129, -1.385, 0.246, 0.189, 0.286, 0.26, 0.781, -1.124, -0.404, 1.364, -0.175, -0.567, 0.224, 0.075, 1.194, -0.549, 1.277, -0.337, 0.221, -0.29, -0.26, -0.904, 0.402, -0.645, 0.88, 0.497, 1.125, -0.803, -0.66, -0.082, -1.763, -0.631, -0.85, -1.661, -1.24, 2.018, -0.013, -0.272, 2.12, -0.913, 1.151, -0.759, 1.724, -0.021, -1.84, -0.417, 0.656, -0.814, -0.179, 1.282, 0.204, 1.122, 1.434, 1.293, 0.761, -0.668, -0.527, -0.712, -0.616, -2.102, -1.03, 1.138, 0.019, -1.038, -1.085, -0.579, 1.427, -1.184, 0.196, -0.145, -0.545, 0.876, -1.262, -1.833, -0.482, 0.209, -0.159, -0.163, -1.457, -0.339, -0.08, -0.459},
            new double[] {0.207, -0.381, 0.564, 1.116, 1.365, 0.417, -0.694, -1.301, 0.803, 0.238, 0.97, -1.597, 1.123, -1.296, -0.119, -0.176, -1.188, -1.303, 1.472, 0.212, 0.895, -1.919, -1.047, -0.419, 1.499, 0.033, 1.513, 0.762, 0.531, 1.071, -0.306, 0.896, 0.709, -0.363, -0.766, -0.61, 0.888, -0.435, 0.839, 1.192, -0.375, -0.861, -2.061, -0.834, -1.093, 2.483, -0.115, -2.47, 0.058, -0.187, -1.067, 0.2, 1.643, -2.492, -1.577, 0.086, 0.622, -0.658, -1.024, 2.55, 1.989, -0.767, 1.17, -0.866, 0.89, 0.979, 0.486, 1.089, 0.574, -1.013, 0.348, 0.851, 0.304, -0.711, 0.018, -0.5, 0.797, 0.017, 0.287, -0.935, 0.644, -0.857, -0.671, -0.797, -0.411, 1.048, -0.705, -1.383, -1.791, 0.142, -0.95, -0.263, -1.591, -1.36, 0.848, -0.716, -1.281, 0.659, 2.224, -0.737, -2.193, 0.79, -0.371, 0.46, -1.135, 0.733, -0.596, 0.303, 0.904},
            5348, 0.8162283275427048, false));
        builder.add(Arguments.of(
            new double[] {1.606, 0.442, -0.039, 0.837, -0.28, 0.826, 0.521, -1.523, 0.452, -0.252, 0.934, -0.607, -0.416, 0.211, -0.479, 0.298, -0.25, 0.249, 0.863, -0.964, 0.18, 1.183, -1.903, 0.536, 0.901, -0.759, -0.275, 0.432, -0.745, 1.235, -2.398, -0.946, 0.469, 0.235, -1.278, 0.024, -0.263, 0.382, -0.739, -0.369, 0.179, 0.595, -0.884, 0.499, 0.677, 1.014, -1.216, -0.49, 0.247, -0.192, -1.272, 0.824, -0.72, -0.876, -0.381, 1.232, -0.037, 1.96, -0.737, 1.485, 1.286, 0.256, -0.341, 0.419, -1.028, -0.34, 1.72, -0.802, 1.299, 0.087, 2.023, 0.584, 1.456, -0.873, 2.247, -0.496, 1.15, 1.569, -0.305, 0.941, 0.882, -0.505, 2.011, -2.787, -0.04, 0.652, 0.04, -0.935, -1.706, -0.772, -0.877, -0.64, 1.464, 0.054, 0.761, -1.241, 1.677, -0.024, 1.397, 0.322, 0.148, 0.698, -1.82, -1.785, 0.586, -0.021, -0.636, -0.257, -0.388, 1.163, 0.66, 1.552, -0.857, -0.987, -0.116, 0.244, -0.372, -0.256, -0.206, 1.504, 0.146, 1.347, -0.034, -0.044, -1.19, 0.21, -0.657, 2.021, 0.875, -1.304, -0.154, -0.574, 0.706, 0.724, 1.295, -0.307, -0.797, -0.627, 1.089, 0.38, -2.377, -1.209, -1.426, 0.263, 0.515, 0.013, 0.887},
            new double[] {-1.335, 0.377, -0.167, 0.137, 0.763, -0.98, -0.073, 2.204, -0.173, -0.886, 1.217, 0.855, 1.189, 1.121, 0.864, 0.528, -1.2, 0.716, -1.033, 0.142, 1.543, 1.101, -1.388, 0.326, -0.032, 0.083, 1.429, 0.274, 0.209, -1.115, 0.714, -0.516, -0.085, 1.004, -0.985, -0.897, -0.179, 0.556, -0.622, 0.843, -0.48, 2.202, -1.519, 1.478, -0.054, 0.954, -0.484, -0.778, 0.898, 0.389, -0.129, 0.358, -0.611, 0.487, 0.175, -0.816, -0.15, -0.397, 0.64, 0.375, 1.305, 2.19, -0.974, 0.131, 0.725, -1.746, -0.202, -0.194, 0.29, 0.756, -0.068, 1.261, -0.333, -0.486, 1.421, -0.653, -0.195, -0.88, 0.098, 0.701, 0.656, 1.288, -0.696, 0.694, -0.284, 2.131, 0.136, -0.115, 0.857, -0.243, -1.344, -1.555, -1.622, -0.148, 1.283, 1.075, -0.3, 1.568, -0.648, -1.755, 1.11, 1.178, 0.072, -0.342, 0.154, -0.466, 0.111, 0.189, 0.509, -0.438, 0.12, 1.687, 0.443, -1.87, -0.236, -1.223, 1.74, 0.866, 2.509, 1.773, 0.27, 0.893, 2.269, 0.15, 0.103, -0.841, -1.164, 1.061, 0.292, -0.337, 1.581, 2.529, 1.186, -0.017, -0.03, -0.087, -2.128, -0.635, 2.048, -0.049, -3.219, 0.58, 0.315, 0.047, -0.483, -0.639, 0.35, -2.385, 0.679, 0.665, 0.197, -0.851, -1.874, -0.523, -1.154},
            11005, 0.60989875841900276, false));
        // Exact computation requested but it is not currently supported. Compare to R inexact.
        // x = 0:515
        // y = 516:1 + 0.5
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = FALSE, exact = FALSE, correct = TRUE)
        // Currently the code does not throw an exception
        final double eps = Math.ulp(1.0);
        builder.accept(Arguments.of(
            IntStream.rangeClosed(0, 515).asDoubleStream().toArray(),
            IntStream.rangeClosed(0, 515).mapToDouble(i -> 516 - i + 0.5).toArray(),
            132355, 0.87181181107702632, false, 4 * eps
        ));
        builder.accept(Arguments.of(
            IntStream.rangeClosed(0, 515).asDoubleStream().toArray(),
            IntStream.rangeClosed(0, 515).mapToDouble(i -> 516 - i + 0.5).toArray(),
            132355, 0.87181181107702632, true, 4 * eps
        ));
        return builder.build();
    }

    @Test
    void testMannWhitneyUExactWithTies() {
        // Edge case where there are ties (x == y).
        // Currently the code does not throw an exception and reverts to the inexact p-value.
        final double[] x = {1, 2, 3, 4, 10};
        final double[] y = {2, 4, 6, 4, 5};
        final double p = testStatistic.mannWhitneyUTest(x, y, false);
        Assertions.assertNotEquals(0, p);
        Assertions.assertEquals(p, testStatistic.mannWhitneyUTest(x, y, true));
    }

    @Test
    void testBigDataSet() {
        final double[] d1 = new double[1500];
        final double[] d2 = new double[1500];
        for (int i = 0; i < 1500; i++) {
            d1[i] = 2 * i;
            d2[i] = 2 * i + 1;
        }
        final double result = testStatistic.mannWhitneyUTest(d1, d2, false);
        Assertions.assertTrue(result > 0.1);
        // This uses inexact computation. It does not currently throw an exception
        Assertions.assertEquals(result, testStatistic.mannWhitneyUTest(d1, d2, true));
    }

    @Test
    void testBigDataSetOverflow() {
        // MATH-1145: n*m > Integer.MAX_VALUE
        final double[] d1 = new double[110000];
        for (int i = 0; i < 110000; i++) {
            d1[i] = i;
        }
        final double[] d2 = d1.clone();
        final double u = testStatistic.mannWhitneyU(d1, d2);
        Assertions.assertEquals(6.05e+09, u);
        final double result = testStatistic.mannWhitneyUTest(d1, d2, false);
        Assertions.assertEquals(1.0, result);
    }

    // XXX
    // Create test method that outputs worst case speed of exact computation for various sizes
    // by directly calling CDF.
    @Test
    void testCDF() {
        // For m,n <= 514
        // For k <= m*n/2
        // Compute cdf(m, n, k) and output timing
    }
}
