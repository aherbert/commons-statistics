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
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Test cases for {@link WilcoxonSignedRankTest}.
 */
class WilcoxonSignedRankTestTest {

    private final WilcoxonSignedRankTest testStatistic = WilcoxonSignedRankTest.instance();

    // XXX - See https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/wilcox.test.R
    // https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Wilcoxon
    // https://github.com/SurajGupta/r-source/blob/master/src/nmath/wilcox.c
    // https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_mannwhitneyu.py#L249-L493

    @Test
    void testWilcoxonSignedRankTestThrows() {
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> WilcoxonSignedRankTest.create(null), "ranking");

        final double[] sample1 = {1, 1};
        final double[] sample2 = {1, 1};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.wilcoxonSignedRankTest(sample1, sample2, false, alpha), "wilcoxonSignedRankTest");

        assertWilcoxonSignedRankTestThrows((x, y) -> testStatistic.wilcoxonSignedRank(x, y));
        assertWilcoxonSignedRankTestThrows((x, y) -> testStatistic.wilcoxonSignedRankTest(x, y, false));
        assertWilcoxonSignedRankTestThrows((x, y) -> testStatistic.wilcoxonSignedRankTest(x, y, false, 0.05));
    }

    private static void assertWilcoxonSignedRankTestThrows(BiConsumer<double[], double[]> action) {
        // Samples must be present, i.e. length > 0
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {}, new double[] {1.0}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1.0}, new double[] {}), "values", "size");

        // Samples not same size, i.e. cannot be paired
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1.0, 2.0}, new double[] {3.0}), "values", "size", "mismatch");

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
        final WilcoxonSignedRankTest test = WilcoxonSignedRankTest.create(ranking);
        final double[] x = {1, 2, 3};
        final double[] y = {6, 4, 2};
        Assertions.assertSame(expected, Assertions.assertThrows(expected.getClass(), () -> test.wilcoxonSignedRank(x, y)));
        Assertions.assertSame(expected, Assertions.assertThrows(expected.getClass(), () -> test.wilcoxonSignedRankTest(x, y, true)));
    }

    /**
     * Test with a large sample. This tests the {@code exact} parameter does not
     * trigger an exception when the sample is too large. If this behaviour changes
     * this test can be updated to assert the exception.
     */
    @ParameterizedTest
    @ValueSource(ints = {2000, 3000})
    void testWilcoxonSignedRankLargeSample(int n) {
        final double[] x = IntStream.range(0, n).asDoubleStream().toArray();
        final double[] y = IntStream.range(0, n).mapToDouble(i -> n - i + 0.5).toArray();
        final double p1 = testStatistic.wilcoxonSignedRankTest(x, y, true);
        final double p2 = testStatistic.wilcoxonSignedRankTest(x, y, false);
        Assertions.assertEquals(p1, p2);
    }

    @ParameterizedTest
    @MethodSource
    void testWilcoxonSignedRank(double[] sample1, double[] sample2, double statistic, double p, boolean exact, double eps) {
        Assertions.assertEquals(statistic, testStatistic.wilcoxonSignedRank(sample1, sample2), "statistic");
        final double actual = testStatistic.wilcoxonSignedRankTest(sample1, sample2, exact);
        TestUtils.assertProbability(p, actual, eps, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.wilcoxonSignedRankTest(sample1, sample2, exact, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.wilcoxonSignedRankTest(sample1, sample2, exact, p * 0.99), "reject");
        }
        // Check symmetry
        final int n = sample1.length;
        final double other = (n * (n + 1.0)) * 0.5 - statistic;
        Assertions.assertEquals(other, testStatistic.wilcoxonSignedRank(sample2, sample1), "statistic (y, x)");
        Assertions.assertEquals(actual, testStatistic.wilcoxonSignedRankTest(sample2, sample1, exact), "p-value (y, x)");
    }

    static Stream<Arguments> testWilcoxonSignedRank() {
        // For relative error of p-value
        final double eps = Math.ulp(1.0);

        final Stream.Builder<Arguments> builder = Stream.builder();

        // Target values computed using R version 3.4.0, e.g.
        // options(digits=20)
        // x <- c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
        // y <- c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = TRUE, exact = TRUE, correct = TRUE)
        // V = 40, p-value = 0.0390625
        // Note: V here is directional (it corresponds to W+)

        // Exact cases
        builder.accept(Arguments.of(
            new double[] {1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30},
            new double[] {0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29},
            40, 0.0390625, true, 0
        ));
        // Edge case where statistic == 0
        builder.accept(Arguments.of(
            new double[] {1, 2, 3},
            new double[] {2, 4, 6},
            0, 0.25, true, 0
        ));
        // R removes any z = x - y where z == 0, so we generate data with no zero deltas.
        // For the exact computation z must be unique (no ties). The following attempts to create
        // a unique y (although samples within y may be identical). R will output when there are
        // ties and the exact computation is invalid. In this case generate again (and optionally
        // increase precision).
        // e.g. for 3 decimal places:
        // UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        // ZigguratSampler.NormalizedGaussian s = ZigguratSampler.NormalizedGaussian.of(rng);
        // for (int n = 50; n <= 200; n += 50) {
        //     double[] x = s.samples(n).map(i -> Precision.round(i, 3)).toArray();
        //     double[] y = s.samples().map(i -> Precision.round(i, 3))
        //                             .filter(i -> Arrays.stream(x).noneMatch(j -> i == j))
        //                             .limit(n).toArray();
        // CHECKSTYLE: stop regex
        //     System.out.println(Arrays.stream(x).mapToObj(Double::toString).collect(Collectors.joining(", ", "new double[] {", "},")));
        //     System.out.println(Arrays.stream(y).mapToObj(Double::toString).collect(Collectors.joining(", ", "new double[] {", "},")));
        // CHECKSTYLE: resume regex
        // }
        builder.accept(Arguments.of(
            new double[] {-2.35, 1.46, 0.88, 0.24, 0.52, -1.10, -0.36, -0.93, 0.60, 0.76},
            new double[] {1.31, -0.88, 0.24, -0.99, 0.79, 0.23, 1.09, -0.01, -0.02, 0.07},
            24, 0.76953125, true, 0
        ));
        builder.accept(Arguments.of(
            new double[] {-0.25, -0.83, -0.43, 1.55, -0.03, -0.03, -0.41, -1.97, 0.59, 0.90, 2.52, 1.29, -1.20, -0.66, 1.02},
            new double[] {0.51, 0.36, 0.91, 1.22, -2.10, -1.16, 0.29, 0.07, 0.91, -0.73, 1.49, -2.08, 1.01, 1.40, 1.82},
            53, 0.71972656250000011, true, eps
        ));
        builder.accept(Arguments.of(
            new double[] {-0.45, 0.75, 0.81, 0.24, -0.68, 1.05, -0.52, 1.12, -1.22, -1.59, 0.32, 1.18, 0.63, -1.21, 0.29, -0.51, -0.59, 1.10, -1.18, -1.08},
            new double[] {1.50, 0.82, -0.85, 0.17, 1.87, -0.56, 0.57, 0.04, -0.02, 0.38, 0.53, -0.97, -0.29, 0.48, 0.07, 0.01, 1.09, -0.37, 1.11, 2.49},
            69, 0.18934822082519531, true, eps
        ));
        builder.accept(Arguments.of(
            new double[] {-0.20, -0.81, 2.99, -1.14, 1.57, 0.95, -0.42, 0.87, -0.56, -0.43, 0.24, 0.42, 1.80, -0.29, -1.21, -1.34, -0.51, 0.82, 0.92, 0.77, -2.66, -0.77, 1.05, -1.88, 0.32},
            new double[] {-1.08, -0.93, 0.08, 0.17, -0.73, 0.00, -1.42, -0.20, -0.92, -1.06, 0.69, -1.42, -1.54, 1.14, -0.12, 2.45, 0.05, 0.21, 0.60, 2.57, 2.01, -0.61, -0.58, 0.25, 0.67},
            174, 0.77115941047668446, true, eps
        ));
        builder.accept(Arguments.of(
            new double[] {-0.874, 0.272, 0.732, 1.040, -1.244, -2.621, -0.750, 0.009, 0.212, 0.407, 2.200, 0.897, -1.378, -1.141, 0.403, -1.222, 1.135, -0.909, -0.211, 0.612, -0.349, -2.022, 0.515, -0.833, 0.724, 0.318, 0.665, 0.597, 0.008, -0.087},
            new double[] {0.997, -0.263, 1.310, -0.093, 0.180, -0.119, 0.013, 1.036, 0.391, -2.282, 1.865, -1.448, -0.181, 0.732, 0.551, 2.061, -1.883, -0.775, 1.004, -0.862, 1.943, -1.163, 0.128, -1.624, 1.197, 0.555, -1.168, 0.052, 0.874, 0.121},
            188, 0.37074060738086706, true, eps
        ));
        // Largest exact computation
        // x = 0:1022
        // y = 1023:1 + 0.5
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = TRUE, exact = TRUE)
        // Note: R is summing count / 2^n as count * exp(-n * ln(2)) so a difference accumulates:
        // Math.scalb(1.0, -1023)        == 1.1125369292536007E-308
        // Math.exp(-1023 * Math.log(2)) == 1.112536929253566E-308
        builder.accept(Arguments.of(
            IntStream.range(0, 1023).asDoubleStream().toArray(),
            IntStream.range(0, 1023).mapToDouble(i -> 1023 - i + 0.5).toArray(),
            261121, 0.93539811751499313, true, 5e-14
        ));

        // Inexact cases
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = TRUE, exact = FALSE, correct = TRUE)
        builder.accept(Arguments.of(
            new double[] {1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30},
            new double[] {0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29},
            40, 0.044010984012951455, false, 2 * eps
        ));
        // Edge case where statistic == 0
        builder.accept(Arguments.of(
            new double[] {1, 2, 3},
            new double[] {2, 4, 6},
            0, 0.18144920772142031, false, 4 * eps
        ));
        // Target values computed using R version 3.4.0, including the continuity correction e.g.
        // R removes any z = x - y where z == 0, so we generate data with no zero deltas (see above)
        builder.accept(Arguments.of(
            new double[] {0.716, 0.113, 0.507, 0.506, -0.11, -1.063, -1.543, -0.735, -0.38, 0.751, 0.778, -1.923, 0.053, 0.342, -0.121, -0.045, -0.086, 0.153, 0.802, -0.402, 1.443, 1.355, -0.518, -0.884, 0.027, 0.129, -1.219, -1.461, -1.426, -0.975, 1.332, -0.472, -1.028, -1.029, -0.962, 0.402, 1.0, 0.147, -0.536, 0.947, 0.721, -0.154, 0.061, -2.281, -1.993, -0.283, 0.02, -0.41, -1.586, 0.129},
            new double[] {0.143, -0.741, -2.008, -0.96, 0.474, 1.527, 1.514, 0.551, -2.142, -0.127, -1.152, 1.531, 0.557, 1.256, 0.629, 1.265, 0.488, -1.981, -0.489, -1.21, 1.52, 0.362, -0.235, -0.207, 1.497, 0.211, 2.054, 0.243, -1.18, -0.817, -0.42, 0.166, 1.575, -1.642, -0.389, -0.482, 2.789, -1.452, 0.795, 0.472, -0.221, -1.589, 1.26, -0.168, 0.647, -0.582, -0.999, -0.217, 0.058, 0.77},
            530.5, 0.30391211935371887, false, 4 * eps
        ));
        builder.accept(Arguments.of(
            new double[] {0.198, 0.299, -0.327, 0.68, 1.914, 2.069, -0.767, 0.833, -1.337, -0.238, 0.037, 1.785, 1.488, 1.164, 0.404, 0.033, 1.667, -0.838, 0.092, -0.696, -0.41, 0.341, -0.398, -1.381, 0.535, -0.528, 1.496, -2.384, -1.739, 0.335, 0.249, -0.779, -0.821, 0.777, 1.455, 1.025, -0.217, 0.21, 0.009, -0.158, -0.664, -0.079, 0.119, 0.342, 0.523, -0.742, -0.757, 0.538, 0.591, -0.241, -0.904, -0.407, 1.43, 0.436, -0.223, -0.663, 1.094, -0.03, -2.054, 1.183, 0.178, 0.171, 0.231, -0.376, -1.431, -1.744, 0.527, 0.523, -0.775, -1.792, 1.096, 1.055, -0.15, -0.247, -0.067, -0.661, -0.207, 0.162, 0.355, -1.945, 0.866, -0.868, 0.774, -1.823, -0.708, -0.877, -0.683, 0.774, -2.174, -1.312, -0.678, -0.236, -0.7, -0.751, -0.638, -1.324, -0.345, 0.164, -0.999, 2.433},
            new double[] {0.095, -0.385, -0.659, -0.051, 0.227, 0.323, 0.227, -0.151, -1.293, 1.195, 0.799, -0.121, 2.589, -0.121, 0.665, -1.417, -2.043, -1.459, -0.097, -0.753, -0.81, -0.33, -0.069, 1.453, 0.197, 0.411, -0.581, 0.35, 0.143, -1.07, 0.725, -0.469, -1.673, -1.097, -0.415, 1.708, 0.52, 1.001, -1.052, -1.319, 1.723, -0.561, 2.333, -0.344, -1.013, 1.062, -1.4, 0.439, -0.92, 1.037, 0.824, -0.912, 1.639, -1.098, 0.838, -0.797, 1.733, 1.464, 0.098, -0.287, 0.481, 0.259, -1.697, 0.371, -0.239, -1.536, -0.448, 0.065, -0.763, -1.497, 0.082, 0.213, 0.353, -0.161, 0.571, 1.091, 1.506, -0.141, 0.23, 0.381, 0.861, -0.416, -0.824, -0.681, -1.105, -0.713, 1.727, -0.902, 1.825, 0.101, 0.917, 0.2, 1.245, 0.046, 0.242, 0.318, 1.293, -0.747, 0.316, -0.51},
            2294, 0.42804876442031159, false, 4 * eps
        ));
        builder.accept(Arguments.of(
            new double[] {0.444, 0.331, -1.225, -1.391, 0.896, 0.885, 1.239, -0.985, -0.84, -0.344, -1.522, -0.845, 1.244, -1.609, -0.565, -0.371, 0.575, -0.775, -0.351, 0.427, -1.294, -0.33, 1.07, 1.307, -1.777, -0.011, 1.452, -0.29, 0.509, -1.604, 2.316, 0.978, 0.075, 0.439, 1.941, 0.122, -1.051, 0.116, -0.466, -1.561, -0.978, 0.54, -1.097, 1.106, 0.835, -1.15, 1.015, 0.182, 0.426, 0.496, -0.484, -1.222, 0.21, -1.213, 1.382, 0.375, -0.282, -0.879, 1.297, -1.467, -0.646, -0.322, -0.996, 1.488, 2.242, 0.479, 0.386, 0.254, -0.394, 2.275, -2.256, 1.644, -0.508, -0.042, -0.108, -0.105, 1.369, 2.124, -0.59, -0.281, -0.165, -1.779, -0.04, 0.534, -0.703, -0.2, 1.031, 1.102, 0.163, -2.057, -0.125, -0.378, 0.839, 0.627, -0.93, -2.635, -0.495, 0.467, 1.288, 1.051, -1.192, -1.354, 0.452, -0.351, 1.508, -0.033, 1.185, -0.546, -0.222, -1.274, -0.886, 1.901, 1.041, 0.042, -0.338, 1.811, -0.8, 0.077, 1.516, 0.183, 1.012, -1.273, 0.224, 1.065, 0.289, -0.336, -2.554, 0.813, 1.038, 0.589, -2.023, -0.422, -0.619, 0.075, 0.829, -0.808, -0.129, 0.725, 0.534, -0.368, -1.1, 1.121, 0.187, 1.095, 1.607, 0.323, -0.724, -0.682, 2.623, 0.999},
            new double[] {0.431, -0.665, -0.504, 1.36, 0.556, -0.096, 0.275, 0.298, 1.291, -1.212, 0.129, 1.455, 0.184, 0.178, -1.645, -0.791, -0.64, 1.068, -1.007, 0.377, 0.371, 0.901, -1.036, 0.88, -0.7, -0.457, -0.822, -0.348, -0.345, -1.338, 0.995, -1.132, 0.023, 0.796, 0.44, -0.586, 0.114, 0.041, 0.304, 0.147, -0.906, 0.208, -2.017, -0.352, -1.511, 0.42, 0.901, -0.036, -1.817, 0.617, -0.109, -0.308, 3.17, 0.333, 1.757, -1.525, -0.076, 0.977, -0.244, -0.512, -0.126, -0.925, -1.291, -0.164, -0.11, -0.747, 0.36, -0.156, -0.019, 0.026, -0.035, -0.637, -0.856, -0.278, -2.123, -0.335, 0.674, -0.325, -0.306, 0.792, -0.19, -1.12, -0.494, -2.005, -0.672, 1.784, 1.667, -1.427, -0.193, 0.815, 0.584, -0.122, 0.634, -1.035, -0.761, 0.09, -0.633, 0.371, -0.807, 0.164, -0.255, 0.161, -0.343, 0.07, 0.398, -1.432, -0.317, -0.588, -1.844, -0.08, 0.081, -0.675, 1.609, -0.82, 0.402, -0.63, -0.736, -0.74, 0.781, -0.778, 0.617, -0.449, 0.917, 0.236, -0.158, -0.785, -1.667, -1.173, -0.628, -0.608, 0.176, 0.476, -0.897, -0.977, 0.142, -0.185, -0.288, 0.935, -0.086, -0.287, -0.483, 0.574, 0.443, 0.791, 0.674, 0.086, -0.734, 0.631, 0.903, -0.733},
            6671, 0.058590782908404818, false, 4 * eps
        ));
        builder.accept(Arguments.of(
            new double[] {-1.913, -0.100, 0.361, -0.726, -0.637, 0.845, -2.157, -1.606, -0.747, -0.684, -0.127, 0.313, 1.140, 0.545, -2.100, 0.177, -0.347, 1.098, -0.020, -0.920, 1.152, -1.057, -0.254, 0.565, -2.260, -0.433, -0.685, -0.614, -2.786, 2.641, 0.880, 0.620, 0.538, 0.260, -0.026, 0.146, 0.516, 0.000, 2.162, -1.050, 1.951, 0.744, 0.349, -0.071, -0.289, -0.916, -0.990, -0.043, -0.173, 0.579, -1.373, 1.781, 1.202, 0.098, 0.082, -0.731, 1.667, -0.874, 0.388, -0.185, 0.117, -0.770, 1.456, -2.188, 0.400, -0.598, 1.216, -0.910, 2.016, 0.206, 2.601, -0.206, -1.390, -0.742, -1.416, -1.261, -0.044, 0.108, -0.575, 0.038, -0.152, -0.432, 1.064, -0.324, 0.665, -0.310, 0.468, -0.186, 1.856, -0.354, 0.226, -1.795, -1.430, -1.213, 0.958, 0.035, -0.126, 0.867, 0.911, -0.208, -1.056, -0.158, -0.662, -1.578, -0.347, 0.338, 1.373, -0.664, -0.129, -0.706, 0.078, 0.034, -0.239, -0.559, 0.710, -0.827, 0.805, -0.416, 0.582, 0.320, 0.891, -0.718, -0.317, 0.373, 0.160, -2.045, 2.371, 0.421, -1.305, -2.235, -0.009, 0.520, 0.758, -0.929, 0.799, -1.217, -0.641, 0.025, 0.999, -1.656, 1.621, 0.297, -0.291, 2.127, 0.663, -0.582, 0.904, -1.033, 1.305, 0.023, 0.190, 0.754, 1.008, -1.927, 0.079, 0.284, 0.865, -0.240, 0.853, 0.237, -1.275, -1.619, -1.552, 0.892, 1.112, 0.829, 0.858, -0.223, 1.246, 1.297, 0.906, 0.606, -0.012, -1.186, 0.971, -0.613, 0.298, 0.351, 0.314, 2.514, 0.287, -1.313, -0.500, 0.955, 0.500, -1.193, 2.070, 0.514, 0.298, -1.035, -0.177, 0.384, -0.472, 1.806, -0.544, -0.882, -0.048, 0.113, -0.622, 1.267},
            new double[] {0.512, 0.830, 0.725, -1.156, -0.704, 1.700, -2.194, -1.254, -2.072, -0.031, 0.381, -0.448, 0.903, 0.972, 0.364, -0.072, -0.319, 0.449, -0.643, 1.391, -0.139, 1.203, -0.519, 1.239, 1.147, -0.205, 0.242, -0.420, 0.458, -1.876, -0.096, -1.781, 0.656, -0.989, 0.841, -0.870, 1.167, 1.076, -1.296, 0.202, 1.519, 0.898, -0.626, 1.120, -1.466, 0.883, -0.251, 0.348, 0.352, 1.500, -1.097, -0.459, 1.750, 0.272, 0.624, 0.286, 1.328, -0.879, 0.772, 0.158, -0.937, 1.465, 1.399, 0.925, -1.382, 0.430, 1.386, 0.457, 2.430, -0.687, 0.398, -0.487, 0.590, 1.982, 0.549, -1.685, 1.776, -0.697, 0.027, -0.495, 2.677, -0.369, 0.465, 0.369, 1.384, 2.614, -0.668, 1.002, -0.079, -0.908, 1.973, 0.424, 0.657, 1.511, 1.171, -0.848, 0.507, -0.425, 0.381, 0.143, -0.586, 0.310, -1.125, 1.904, -0.301, 1.091, -0.868, -0.346, -0.580, -1.310, 1.063, 0.766, 0.567, 2.055, -1.457, -0.501, -1.098, 0.625, -0.615, -0.930, -1.341, 1.012, 0.441, -0.661, -0.792, 0.419, 2.773, -0.892, -0.829, 0.662, -1.051, 1.000, 0.333, 2.639, -0.550, -1.491, 1.542, -0.377, 0.801, -0.549, -1.992, -0.231, -0.469, -1.584, -2.626, -0.450, -0.160, -1.361, -0.037, -1.585, -1.064, -0.330, -0.311, -0.995, -0.300, 0.737, -0.549, 1.656, -0.806, -0.375, 0.301, 0.064, -1.200, -0.153, -0.906, -1.569, -0.757, -0.141, 0.083, -0.388, 0.983, 1.286, 0.533, -0.449, -1.357, -0.407, 0.123, -0.335, -0.546, 1.106, 0.512, -0.843, -0.445, 0.058, -0.562, 0.158, 1.464, 1.075, -1.222, -0.589, 0.181, -2.208, -0.142, -1.844, -0.353, -2.242, -1.383, 0.185, 1.696, 1.528},
            9947, 0.90047012937135407, false, 4 * eps
        ));
        // Largest exact computation as inexact
        // x = 0:1022
        // y = 1023:1 + 0.5
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = TRUE, exact = FALSE)
        builder.accept(Arguments.of(
            IntStream.range(0, 1023).asDoubleStream().toArray(),
            IntStream.range(0, 1023).mapToDouble(i -> 1023 - i + 0.5).toArray(),
            261121, 0.9353698195565141, false, 4 * eps
        ));
        // Exact computation requested but it is not currently supported. Compare to R inexact.
        // x = 0:1023
        // y = 1024:1 + 0.5
        // wilcox.test(x, y, alternative = "two.sided", mu = 0, paired = TRUE, exact = FALSE)
        // Currently the code does not throw an exception
        builder.accept(Arguments.of(
            IntStream.range(0, 1024).asDoubleStream().toArray(),
            IntStream.range(0, 1024).mapToDouble(i -> 1024 - i + 0.5).toArray(),
            261632, 0.93538020545645328, true, 4 * eps
        ));
        builder.accept(Arguments.of(
            IntStream.range(0, 1024).asDoubleStream().toArray(),
            IntStream.range(0, 1024).mapToDouble(i -> 1024 - i + 0.5).toArray(),
            261632, 0.93538020545645328, false, 4 * eps
        ));
        return builder.build();
    }

    @Test
    void testWilcoxonSignedRankExactWithZeros() {
        // Edge case where there are zeros (x == y).
        // Currently the code does not throw an exception and reverts to the inexact p-value.
        final double[] x = {1, 2, 3, 4, 10};
        final double[] y = {2, 4, 6, 4, 5};
        final double p = testStatistic.wilcoxonSignedRankTest(x, y, false);
        Assertions.assertNotEquals(0, p);
        Assertions.assertEquals(p, testStatistic.wilcoxonSignedRankTest(x, y, true));
    }
}
