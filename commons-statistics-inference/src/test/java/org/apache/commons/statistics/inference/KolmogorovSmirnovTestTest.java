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

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.distribution.NormalDistribution;
import org.apache.commons.statistics.distribution.UniformContinuousDistribution;
import org.apache.commons.statistics.ranking.NaturalRanking;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvFileSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link KolmogorovSmirnovTest}.
 */
class KolmogorovSmirnovTestTest {
    // Random N(0, 1) values generated using R rnorm (n=102)
    private static final double[] GAUSSIAN = {
        0.26055895, -0.63665233, 1.51221323, 0.61246988, -0.03013003, -1.73025682, -0.51435805, 0.70494168, 0.18242945,
        0.94734336, -0.04286604, -0.37931719, -1.07026403, -2.05861425, 0.11201862, 0.71400136, -0.52122185,
        -0.02478725, -1.86811649, -1.79907688, 0.15046279, 1.32390193, 1.55889719, 1.83149171, -0.03948003,
        -0.98579207, -0.76790540, 0.89080682, 0.19532153, 0.40692841, 0.15047336, -0.58546562, -0.39865469, 0.77604271,
        -0.65188221, -1.80368554, 0.65273365, -0.75283102, -1.91022150, -0.07640869, -1.08681188, -0.89270600,
        2.09017508, 0.43907981, 0.10744033, -0.70961218, 1.15707300, 0.44560525, -2.04593349, 0.53816843, -0.08366640,
        0.24652218, 1.80549401, -0.99220707, -1.14589408, -0.27170290, -0.49696855, 0.00968353, -1.87113545,
        -1.91116529, 0.97151891, -0.73576115, -0.59437029, 0.72148436, 0.01747695, -0.62601157, -1.00971538,
        -1.42691397, 1.03250131, -0.30672627, -0.15353992, -1.19976069, -0.68364218, 0.37525652, -0.46592881,
        -0.52116168, -0.17162202, 1.04679215, 0.25165971, -0.04125231, -0.23756244, -0.93389975, 0.75551407,
        0.08347445, -0.27482228, -0.4717632, -0.1867746, -0.1166976, 0.5763333, 0.1307952, 0.7630584, -0.3616248,
        2.1383790, -0.7946630, 0.0231885, 0.7919195, 1.6057144, -0.3802508, 0.1229078, 1.5252901, -0.8543149, 0.3025040
    };

    // Random N(0, 1.6) values generated using R rnorm (n=100)
    private static final double[] GAUSSIAN2 = {
        2.88041498038308, -0.632349445671017, 0.402121295225571, 0.692626364613243, 1.30693446815426,
        -0.714176317131286, -0.233169206599583, 1.09113298322107, -1.53149079994305, 1.23259966205809,
        1.01389927412503, 0.0143898711497477, -0.512813545447559, 2.79364360835469, 0.662008875538092,
        1.04861546834788, -0.321280099931466, 0.250296656278743, 1.75820367603736, -2.31433523590905,
        -0.462694696086403, 0.187725700950191, -2.24410950019152, 2.83473751105445, 0.252460174391016,
        1.39051945380281, -1.56270144203134, 0.998522814471644, -1.50147469080896, 0.145307533554146,
        0.469089457043406, -0.0914780723809334, -0.123446939266548, -0.610513388160565, -3.71548343891957,
        -0.329577317349478, -0.312973794075871, 2.02051909758923, 2.85214308266271, 0.0193222002327237,
        -0.0322422268266562, 0.514736012106768, 0.231484953375887, -2.22468798953629, 1.42197716075595,
        2.69988043856357, 0.0443757119128293, 0.721536984407798, -0.0445688839903234, -0.294372724550705,
        0.234041580912698, -0.868973119365727, 1.3524893453845, -0.931054600134503, -0.263514296006792,
        0.540949457402918, -0.882544288773685, -0.34148675747989, 1.56664494810034, 2.19850536566584,
        -0.667972122928022, -0.70889669526203, -0.00251758193079668, 2.39527162977682, -2.7559594317269,
        -0.547393502656671, -2.62144031572617, 2.81504147017922, -1.02036850201042, -1.00713927602786,
        -0.520197775122254, 1.00625480138649, 2.46756916531313, 1.64364743727799, 0.704545210648595,
        -0.425885789416992, -1.78387854908546, -0.286783886710481, 0.404183648369076, -0.369324280845769,
        -0.0391185138840443, 2.41257787857293, 2.49744281317859, -0.826964496939021, -0.792555379958975,
        1.81097685787403, -0.475014580016638, 1.23387615291805, 0.646615294802053, 1.88496377454523, 1.20390698380814,
        -0.27812153371728, 2.50149494533101, 0.406964323253817, -1.72253451309982, 1.98432494184332, 2.2223658560333,
        0.393086362404685, -0.504073151377089, -0.0484610869883821
    };

    // Random uniform(0, 1) generated using R runif (n=102)
    private static final double[] UNIFORM = {
        0.7930305, 0.6424382, 0.8747699, 0.7156518, 0.1845909, 0.2022326, 0.4877206, 0.8928752, 0.2293062, 0.4222006,
        0.1610459, 0.2830535, 0.9946345, 0.7329499, 0.26411126, 0.87958133, 0.29827437, 0.39185988, 0.38351185,
        0.36359611, 0.48646472, 0.05577866, 0.56152250, 0.52672013, 0.13171783, 0.95864085, 0.03060207, 0.33514887,
        0.72508148, 0.38901437, 0.9978665, 0.5981300, 0.1065388, 0.7036991, 0.1071584, 0.4423963, 0.1107071, 0.6437221,
        0.58523872, 0.05044634, 0.65999539, 0.37367260, 0.73270024, 0.47473755, 0.74661163, 0.50765549, 0.05377347,
        0.40998009, 0.55235182, 0.21361998, 0.63117971, 0.18109222, 0.89153510, 0.23203248, 0.6177106, 0.6856418,
        0.2158557, 0.9870501, 0.2036914, 0.2100311, 0.9065020, 0.7459159, 0.56631790, 0.06753629, 0.39684629,
        0.52504615, 0.14199103, 0.78551120, 0.90503321, 0.80452362, 0.9960115, 0.8172592, 0.5831134, 0.8794187,
        0.2021501, 0.2923505, 0.9561824, 0.8792248, 0.85201008, 0.02945562, 0.26200374, 0.11382818, 0.17238856,
        0.36449473, 0.69688273, 0.96216330, 0.4859432, 0.4503438, 0.1917656, 0.8357845, 0.9957812, 0.4633570,
        0.8654599, 0.4597996, 0.68190289, 0.58887855, 0.09359396, 0.98081979, 0.73659533, 0.89344777, 0.18903099,
        0.97660425
    };

    @Test
    void testOneSampleThrows() {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final DoubleUnaryOperator cdf = UniformContinuousDistribution.of(0, 5)::cumulativeProbability;
        final double[] data = {1, 3};
        TestUtils.assertSignificanceLevel(
            alpha -> test.kolmogorovSmirnovTest(cdf, data, alpha), "kolmogorovSmirnovTest");

        testOneSampleThrows(x -> test.kolmogorovSmirnovStatistic(cdf, x));
        testOneSampleThrows(x -> test.kolmogorovSmirnovTest(cdf, x));
        testOneSampleThrows(x -> test.kolmogorovSmirnovTest(cdf, x, 0.05));
    }

    private static void testOneSampleThrows(Consumer<double[]> action) {
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1}), "values", "size");

        // NaN
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(new double[] {2, Double.NaN}), "nan");

        // x is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null));
    }

    @ParameterizedTest
    @MethodSource
    void testOneSample(DoubleUnaryOperator cdf, double[] x, double statistic, double p, double eps) {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        TestUtils.assertRelativelyEquals(statistic, test.kolmogorovSmirnovStatistic(cdf, x), eps, "statistic");
        TestUtils.assertProbability(p, test.kolmogorovSmirnovTest(cdf, x), eps, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(test.kolmogorovSmirnovTest(cdf, x, p * 1.01), "reject");
            Assertions.assertFalse(test.kolmogorovSmirnovTest(cdf, x, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testOneSample() {
        final DoubleUnaryOperator unitNormal = NormalDistribution.of(0, 1)::cumulativeProbability;
        final DoubleUnaryOperator unif = UniformContinuousDistribution.of(0, 1)::cumulativeProbability;
        final DoubleUnaryOperator unif05 = UniformContinuousDistribution.of(-0.5, 0.5)::cumulativeProbability;
        final Stream.Builder<Arguments> builder = Stream.builder();
        // scipy.stats 1.9.3
        // Normal distribution, unit normal dataset
        // stats.kstest(GAUSSIAN, stats.norm.cdf)
        builder.add(Arguments.of(unitNormal, GAUSSIAN, 0.0932947561266756, 0.3172069207622401, 1e-15));
        // Normal distribution, unit normal small dataset
        // stats.kstest(GAUSSIAN[0:50], stats.norm.cdf)
        builder.add(Arguments.of(unitNormal, Arrays.copyOf(GAUSSIAN, 50), 0.0982077996946327, 0.6837364637283481, 1e-15));
        // Normal distribution, uniform dataset
        // stats.kstest(UNIFORM, stats.norm.cdf)
        builder.add(Arguments.of(unitNormal, UNIFORM, 0.5117493931609258, 2.6003915104391943e-25, 1e-15));
        // Uniform distribution, uniform dataset
        // stats.kstest(UNIFORM, stats.uniform.cdf)
        builder.add(Arguments.of(unif, UNIFORM, 0.06153833137254894, 0.8117325066019292, 1e-15));
        // Uniform distribution, uniform small dataset
        // stats.kstest(UNIFORM[0:20], stats.uniform.cdf)
        builder.add(Arguments.of(unif, Arrays.copyOf(UNIFORM, 20), 0.1610459, 0.6205703200955432, 1e-15));
        // Offset uniform distribution, uniform dataset
        // stats.kstest(UNIFORM, stats.uniform(-0.5).cdf)
        builder.add(Arguments.of(unif05, UNIFORM, 0.5400666982352942, 2.24649011643408e-28, 1e-15));
        // Offset uniform distribution, uniform small dataset
        // stats.kstest(UNIFORM[0:20], stats.uniform(-0.5).cdf)
        builder.add(Arguments.of(unif05, Arrays.copyOf(UNIFORM, 20), 0.6610459, 4.117594713484523e-09, 1e-15));
        // Offset uniform distribution, unit normal dataset
        // stats.kstest(GAUSSIAN, stats.uniform(-0.5).cdf)
        builder.add(Arguments.of(unif05, GAUSSIAN, 0.3401058049019608, 4.940576402189508e-11, 1e-15));
        return builder.build();
    }

    @Test
    void testTwoSampleThrows() {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double[] sample1 = {1, 3};
        final double[] sample2 = {2, 4};
        TestUtils.assertSignificanceLevel(
            alpha -> test.kolmogorovSmirnovTest(sample1, sample2, false, alpha), "kolmogorovSmirnovTest");

        testTwoSampleThrows((x, y) -> test.kolmogorovSmirnovStatistic(x, y));
        testTwoSampleThrows((x, y) -> test.kolmogorovSmirnovTest(x, y, false));
        testTwoSampleThrows((x, y) -> test.kolmogorovSmirnovTest(x, y, false, 0.05));
        testTwoSampleThrows((x, y) -> test.kolmogorovSmirnovTest(x, y, true));
        testTwoSampleThrows((x, y) -> test.kolmogorovSmirnovTest(x, y, true, 0.05));
    }

    private static void testTwoSampleThrows(BiConsumer<double[], double[]> action) {
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 3}, new double[] {2}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1}, new double[] {2, 4}), "values", "size");

        // NaN - small sample
        // Message current does not contain the sample name
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(new double[] {1, 3}, new double[] {2, Double.NaN}), "nan");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, Double.NaN}, new double[] {2, 4}), "nan");
        // NaN - large sample
        final double[] x = new double[10000];
        final double[] y = new double[10000];
        x[0] = Double.NaN;
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(x, y), "sample 1", "nan");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(y, x), "sample 2", "nan");

            // x and y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, null));

        // x or y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, new double[] {2, 4}));
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(new double[] {1, 2}, null));
    }

    @ParameterizedTest
    @MethodSource
    void testTwoSample(double[] x, double[] y, boolean strict, double statistic, double p, double eps) {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        TestUtils.assertRelativelyEquals(statistic, test.kolmogorovSmirnovStatistic(x, y), eps, "statistic");
        TestUtils.assertProbability(p, test.kolmogorovSmirnovTest(x, y, strict), eps, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(test.kolmogorovSmirnovTest(x, y, strict, p * 1.01), "reject");
            Assertions.assertFalse(test.kolmogorovSmirnovTest(x, y, strict, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testTwoSample() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // scipy.stats 1.9.3 (uses non-strict inequality in null hypothesis)
        // stats.kstest(x, y)
        builder.add(Arguments.of(
            new double[] {6, 7, 9, 13, 19, 21, 22, 23, 24},
            new double[] {10, 11, 12, 16, 20, 27, 28, 32, 44, 54},
            false,
            0.5, 0.10557708545324647, 1e-15));
        builder.add(Arguments.of(
            new double[] {6, 7, 9, 13, 19, 21, 22, 23, 24, 29, 30, 34, 36, 41, 45, 47, 51, 63, 33, 91},
            new double[] {10, 11, 12, 16, 20, 27, 28, 32, 44, 54, 56, 57, 64, 69, 71, 80, 81, 88, 90},
            false,
            0.4263157894736842, 0.0462986609429517, 1e-15));
        builder.add(Arguments.of(
            new double[] {-10, -5, 17, 21, 22, 23, 24, 30, 44, 50, 56, 57, 59, 67, 73, 75, 77, 78,
                79, 80, 81, 83, 84, 85, 88, 90, 92, 93, 94, 95, 98, 100, 101, 103, 105, 110},
            new double[] {-2, -1, 0, 10, 14, 15, 16, 20, 25, 26, 27, 31, 32, 33, 34, 45, 47, 48,
                51, 52, 53, 54, 60, 61, 62, 63, 74, 82, 106, 107, 109, 11, 112, 113, 114},
            false,
            0.4103174603174603, 0.0030074360223340544, 1e-15));
        // Identical input
        builder.add(Arguments.of(
                new double[] {1, 2, 3},
                new double[] {1, 2, 3},
                false,
                0, 1, 0));
        builder.add(Arguments.of(
                new double[] {1, 2, 3},
                new double[] {1, 2, 3},
                true,
                0, 1, 0));
        return builder.build();
    }

    /**
     * Checks exact p-value computations using critical values from Table 9 in V.K Rohatgi, An
     * Introduction to Probability and Mathematical Statistics, Wiley, 1976, ISBN 0-471-73135-8.
     * Verifies the inequality exactP(criticalValue, n, m, true) < alpha < exactP(criticalValue, n,
     * m, false).
     *
     * <p>Note that the validity of this check depends on the fact that alpha lies strictly between two
     * attained values of the distribution and that criticalValue is one of the attained values. The
     * critical value table (reference below) uses attained values. This test therefore also
     * verifies that criticalValue is attained.
     *
     * @param n first sample size
     * @param m second sample size
     * @param criticalValue critical value (D * n * m)
     * @param alpha significance level
     */
    @ParameterizedTest
    @MethodSource
    void testTwoSampleExactP(int n, int m, long criticalValue, double alpha) {
        Assertions.assertTrue(KolmogorovSmirnovTest.twoSampleExactP(criticalValue, n, m, true) < alpha);
        Assertions.assertTrue(KolmogorovSmirnovTest.twoSampleExactP(criticalValue, n, m, false) > alpha);
    }

    static Stream<Arguments> testTwoSampleExactP() {
        return Stream.of(
            Arguments.of(4, 6, 20, 0.01),  // d = 20 / 24 =  5 /  6
            Arguments.of(4, 7, 17, 0.2),   // d = 17 / 28
            Arguments.of(6, 7, 29, 0.05),  // d = 29 / 42
            Arguments.of(4, 10, 28, 0.05), // d = 28 / 40 =  7 / 10
            Arguments.of(5, 15, 55, 0.02), // d = 55 / 75 = 11 / 15
            Arguments.of(9, 10, 62, 0.01), // d = 62 / 90 = 31 / 45
            Arguments.of(7, 10, 43, 0.05)  // d = 43 / 70
        );
    }

    @ParameterizedTest
    @CsvFileSource(resources = {"ks.twosample.small.txt"}, delimiter = ' ')
    void testTwoSampleExactPSmall(int m, int n, long dmn, double p) {
        final double p2 = KolmogorovSmirnovTest.twoSampleExactP(dmn, n, m, false);
        TestUtils.assertProbability(p, p2, 0, "twoSampleExactP");
    }

    @ParameterizedTest
    @CsvFileSource(resources = {"ks.twosample.medium.txt"}, delimiter = ' ')
    void testTwoSampleExactPMedium(int m, int n, long dmn, double p) {
        final double p2 = KolmogorovSmirnovTest.twoSampleExactP(dmn, n, m, false);
        TestUtils.assertProbability(p, p2, 0, "twoSampleExactP");
    }

    @ParameterizedTest
    @CsvFileSource(resources = {"ks.twosample.large.txt"}, delimiter = ' ')
    void testTwoSampleExactPLarge(int m, int n, long dmn, double p) {
        final double p2 = KolmogorovSmirnovTest.twoSampleExactP(dmn, n, m, false);
        TestUtils.assertProbability(p, p2, Math.ulp(1.0), "twoSampleExactP");
    }

    @ParameterizedTest
    @MethodSource(value = {"testTwoSampleApproximateP"})
    void testTwoSampleApproximateP(int n, int m, double criticalValue, double alpha, double epsilon, double ignored) {
        TestUtils.assertProbability(alpha, KolmogorovSmirnovTest.twoSampleApproximateP(criticalValue, n, m), epsilon, "approximateP");
    }

    @ParameterizedTest
    @MethodSource(value = {"testTwoSampleApproximateP"})
    void testTwoSampleApproximatePKSsum(int n, int m, double criticalValue, double alpha, double ignored, double epsilon) {
        final double nn = ((double) m * n) / ((double) m + n);
        final double x2 = nn * criticalValue * criticalValue;
        TestUtils.assertProbability(alpha, KolmogorovSmirnovDistributionExt.ksSum(x2), epsilon, "approximateP");
    }

    static Stream<Arguments> testTwoSampleApproximateP() {
        final Stream.Builder<Arguments> builder = Stream.builder();

        // This method is used to test the approximate p-value
        // using the KolmogorovSmirnovDistribution.Two.sf
        // and KolmogorovSmirnovDistributionExt.ksSum.
        // The KS sum is close to the asymptotic limit. The Two.sf
        // is closer to the exact p-value computed using
        // KolmogorovSmirnovTest.exactP for a range of integral d-statistic
        // where p is in the range 0.001 to 0.1.

        // From Wikipedia KS article.
        // This data is for very large n where the asymptotic series reduces to a sum of a
        // single term: 2 exp(-2 z^2); z = d * sqrt(n*m / (n+m))
        final double tol = 1.5e-2;
        final double tolKSsum = 1.5e-4;
        final double[] alpha = {
            0.10, 0.05, 0.025, 0.01, 0.005, 0.001
        };
        final double[] c = {
            // sqrt(-log(alpha/2) / 2)
            1.2238734153404083, 1.3581015157406195, 1.4802071873007983,
            1.6276236307187293, 1.7308183826022854, 1.9494746035204051
        };
        final int[] k = {
            // Very large n
            10000, 50000, 100000
        };
        int n;
        int m;
        for (int i = 0; i < k.length; i++) {
            n = k[i];
            for (int j = 0; j < i; j++) {
                m = k[j];
                for (int l = 0; l < alpha.length; l++) {
                    final double dCrit = c[l] * Math.sqrt((n + m) / ((double) n * m));
                    builder.add(Arguments.of(n, m, dCrit, alpha[l], tol, tolKSsum));
                }
            }
        }

        // Edge case with d=0
        builder.add(Arguments.of(10, 10, 0.0, 1, 0, 0));
        builder.add(Arguments.of(100, 100, 0.0, 1, 0, 0));

        // scipy uses the two-sided one sample for an approximate p:
        // from scipy.stats import distributions
        // import numpy as np
        // m, n = sorted([float(n1), float(n2)], reverse=True)
        // en = m * n / (m + n)
        // stats.kstwo.sf(d, np.round(en))
        // This is more accurate as p << 1e-16.
        builder.add(Arguments.of(10000, 10000, 0.02, 0.03613941395325637, 8e-15, 0.02));
        builder.add(Arguments.of(10000, 10000, 0.01, 0.6954557253913853, 1e-15, 6e-3));
        builder.add(Arguments.of(10000, 10000, 0.008, 0.9035907876780276, 1e-15, 3e-3));
        builder.add(Arguments.of(10000, 10000, 0.007, 0.9656426076715481, 1e-15, 2e-3));
        builder.add(Arguments.of(10000, 10000, 0.006, 0.9933190457003288, 1e-15, 5e-4));
        builder.add(Arguments.of(10000, 10000, 0.005, 0.9995858441477996, 1e-15, 5e-5));
        builder.add(Arguments.of(10000, 11000, 0.008, 0.8880643485565741, 1e-15, 4e-3));
        builder.add(Arguments.of(10000, 11000, 0.007, 0.9579679709076735, 1e-15, 2e-3));
        builder.add(Arguments.of(10000, 11000, 0.006, 0.9911437365335491, 1e-15, 6e-4));

        return builder.build();
    }

    /**
     * Test the two-sample approximations.
     * The p-value of the approximations are compared to the exact p-value over a range
     * of p typical for critical alpha values. The root mean square error is computed
     * and the used to check the current approximation is the best choice.
     */
    @ParameterizedTest
    @CsvSource({
        "16, 16, 0.001, 0.1, 500",
        "16, 32, 0.001, 0.1, 500",
        "2048, 2048, 0.001, 0.1, 500",
    })
    void testTwoSampleApproximations(int n, int m, double pMin, double pMax, int samples) {
        // Note: the integral statistic is lower for a larger p-value
        final long upper = findTwoSampleD(pMin, n, m, 0, Long.MAX_VALUE, false);
        final long lower = findTwoSampleD(pMax, n, m, 0, upper, true);
        int c = 0;
        double m1 = 0;
        double m2 = 0;
        final long step = (long) Math.ceil((double) (upper - lower) / samples);
        for (long dnm = lower; dnm <= upper; dnm += step) {
            final double p = KolmogorovSmirnovTest.twoSampleExactP(dnm, n, m, false);
            // Ignore p-values outside the range of interest
            if (p < pMin || p > pMax) {
                continue;
            }
            final double x = dnm / ((double) n * m);
            final double en = ((double) m * n) / ((double) m + n);
            final double p1 = KolmogorovSmirnovDistribution.Two.sf(x, (int) Math.round(en));
            final double e1 = Math.abs(p - p1) / Math.max(p, p1);
            final double p2 = KolmogorovSmirnovDistributionExt.ksSum(x * x * en);
            final double e2 = Math.abs(p - p2) / Math.max(p, p2);
            m1 += e1 * e1;
            m2 += e2 * e2;
            c++;
        }
        final int count = c;
        Assertions.assertTrue(c > 30, () -> "Not enough samples: " + count);
        final double rmsd1 = Math.sqrt(m1 / c);
        final double rmsd2 = Math.sqrt(m2 / c);
        Assertions.assertTrue(rmsd1 <= rmsd2, () -> String.format("RMSD Two.sf %s > KS sum %s", rmsd1, rmsd2));
    }

    /**
     * Search for the integral D statistics for the given p value.
     *
     * @param p Target probability.
     * @param n First sample size.
     * @param m Second sample size.
     * @param a Lower bound on the search range.
     * @param b Upper bound ofn the search range.
     * @param above True to return the d value for the closest p-value above the
     * target
     * @return Integral D statistic d*n*m
     */
    private static long findTwoSampleD(double p, int n, int m, long a, long b, boolean above) {
        long l = Math.max(0, a);
        long h = Math.min(b, (long) n * m);
        while (l + 1 < h) {
            final long mid = (l + h) >>> 1;
            final double x = KolmogorovSmirnovTest.twoSampleExactP(mid, n, m, false);
            // Higher values of d give a lower p
            if (x < p) {
                h = mid;
            } else {
                l = mid;
            }
        }
        return above ? l : h;
    }

    /** Verifies large sample approximate p values. */
    @Test
    void testTwoSampleLarge() {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        // Reference values from R, version 2.15.3
        // Data from scipy 1.9.3
        // stats.kstest(GAUSSIAN, GAUSSIAN2, method='exact')
        // statistic=0.2023529411764706, pvalue=0.026098802349515924
        // stats.kstest(GAUSSIAN, GAUSSIAN2, method='asymp')
        // 'asymp' uses kstwo.sf not a KS sum limiting form.
        // statistic=0.2023529411764705, pvalue=0.028514899820936468
        TestUtils.assertRelativelyEquals(0.202352941176471,
            test.kolmogorovSmirnovStatistic(GAUSSIAN, GAUSSIAN2), 1e-14, "statistic");
        // XXX: Update test when the p-value can be user-selected
        TestUtils.assertProbability(0.026098802349515924,
            test.kolmogorovSmirnovTest(GAUSSIAN, GAUSSIAN2, false), 1e-14, "p-value");
    }

    /**
     * MATH-1181
     * Verify that large sample method is selected for sample product > Integer.MAX_VALUE
     * (integer overflow in sample product)
     */
    @Test
    @Timeout(value = 5000, unit = TimeUnit.MILLISECONDS)
    void testTwoSampleProductSizeOverflow() {
        final int n = 46341;
        Assertions.assertTrue(n * n < 0);
        final double[] x = new double[n];
        final double[] y = new double[n];
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        Assertions.assertFalse(Double.isNaN(test.kolmogorovSmirnovTest(x, y, false)));
    }

    /**
     * Checks that ties in the data are resolved deterministically (i.e. repeatable).
     */
    @ParameterizedTest
    @MethodSource
    void testTwoSampleWithTies(double[] x, double[] y) {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double p = test.kolmogorovSmirnovTest(x, y, false);
        // Should be deterministic
        Assertions.assertEquals(p, test.kolmogorovSmirnovTest(x, y, false), "Not repeatable");
        Assertions.assertEquals(p, test.kolmogorovSmirnovTest(y, x, false), "Sensitive to array input order");
    }

    static Stream<Arguments> testTwoSampleWithTies() {
        return Stream.of(
            // Identical
            Arguments.of(new double[] {1, 2, 3}, new double[] {1, 2, 3}),
            // Tied for their entire length, but different lengths
            Arguments.of(new double[] {1, 1, 1, 1, 1}, new double[] {1, 1}),
            Arguments.of(new double[] {1, 1, 1}, new double[] {1, 1, 1, 1}),
            // Some ties
            Arguments.of(new double[] {0, 1, 2, 3, 4, 2}, new double[] {5, 6, 7, 8, 1, 2}),
            Arguments.of(new double[] {0, 1, 1, 4, 0}, new double[] {0, 5, 0.5, 0.55, 7}),
            Arguments.of(new double[] {1, 1, 0, 1, 0}, new double[] {0, 0, 0}),
            // Ties to the end
            Arguments.of(new double[] {-1, 1, 1, 1}, new double[] {-2, 1, 1, 1}),
            // Ties from the start
            Arguments.of(new double[] {1, 1, 1, 2}, new double[] {1, 1, 1, 3}),
            // Ties at the start/end respectively
            Arguments.of(new double[] {1, 1, 1, 2}, new double[] {0, 1, 1, 1}),
            // This cannot be resolved by random jitter
            Arguments.of(
                DoubleStream.of(0, 1, 2, 3, 4, 2).map(i -> Double.MIN_VALUE * i).toArray(),
                DoubleStream.of(5, 6, 7, 8, 1, 2).map(i -> Double.MIN_VALUE * i).toArray())
        );
    }

    @Test
    void testTwoSampleWithManyTies() {
        // MATH-1197
        // Computation of the incorrect d statistic in the event of ties
        final double[] x = {
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 2.202653,
            3.181199, 3.181199, 3.181199, 3.181199, 3.181199, 3.181199,
            3.723539, 3.723539, 3.723539, 3.723539, 4.383482, 4.383482,
            4.383482, 4.383482, 5.320671, 5.320671, 5.320671, 5.717284,
            6.964001, 7.352165, 8.710510, 8.710510, 8.710510, 8.710510,
            8.710510, 8.710510, 9.539004, 9.539004, 10.720619, 17.726077,
            17.726077, 17.726077, 17.726077, 22.053875, 23.799144, 27.355308,
            30.584960, 30.584960, 30.584960, 30.584960, 30.751808
        };

        final double[] y = {
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 2.202653, 2.202653, 2.202653,
            2.202653, 2.202653, 2.202653, 2.202653, 2.202653, 3.061758,
            3.723539, 5.628420, 5.628420, 5.628420, 5.628420, 5.628420,
            6.916982, 6.916982, 6.916982, 10.178538, 10.178538, 10.178538,
            10.178538, 10.178538
        };

        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();

        final double d = test.kolmogorovSmirnovStatistic(x, y);
        // Computed using R version 3.4.0: ks.test(x, y)
        // D = 0.064039408866995120584, p-value = 0.97927772901338495
        // Warning: 'p-value will be approximate in the presence of ties'
        TestUtils.assertRelativelyEquals(0.064039408866995120584, d, 1e-15, "statistic");

        // The p-value is not close to p(D=0.0640) due to random tie resolution.
        // exactP = 0.9659836534406034 (with dnm = (long) (d * n * m))
        // estimateP = 0.558364 (with 1000000 iterations)
        // Here we repeat call the test using different random seeds
        // for tie resolution and take an average p-value.
        // The relative error is stable around 0.065. Examples with a randomly seeded RNG:
        // Trials  Relative error
        // 1000    ~0.0622
        // 10000   ~0.0651
        // 100000  ~0.0628
        // For robustness use a fixed seed with a small number of trials
        final double p = RandomSource.SPLIT_MIX_64.create(12345)
            .longs(100)
            .mapToDouble(seed -> KolmogorovSmirnovTest.kolmogorovSmirnovTest(x, y, false, seed))
            .average().getAsDouble();
        TestUtils.assertProbability(0.558364, p, 0.065, "p-value");
    }

    @Test
    void testTwoSampleWithManyTiesAndVerySmallDelta() {
        // Cf. MATH-1405

        final double[] x = {
            0.0, 0.0,
            1.0, 1.0,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.000000000000001
        };

        final double[] y = {
            0.0, 0.0,
            10.0, 10.0,
            11.0, 11.0, 11.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            20.000000000000001
        };

        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        // XXX - This test is sensitive to using strict inequality. Tie resolution
        // will be determined based on the RNG. Find out the possible p-values for
        // the possible statistics.
        Assertions.assertEquals(1.12173015e-5, test.kolmogorovSmirnovTest(x, y, true), 1e-6);
    }

    @Test
    void testTwoSampleWithManyTiesAndExtremeValues() {
        // Cf. MATH-1405
        // fixTies can set minDelta too small for random jitter to have significant effect.
        // This is no longer applicable because jitter is not used. The random ordering is
        // applied directly to the run of tied data.
        final double[] largeX = {
            Double.MAX_VALUE, Double.MAX_VALUE,
            1e40, 1e40,
            2e40, 2e40,
            1e30,
            2e30,
            3e30,
            4e30,
            5e10,
            6e10,
            7e10,
            8e10
        };
        final double[] smallY = {
            Double.MIN_VALUE,
            2 * Double.MIN_VALUE,
            1e-40, 1e-40,
            2e-40, 2e-40,
            1e-30,
            2e-30,
            3e-30,
            4e-30,
            5e-10,
            6e-10,
            7e-10,
            8e-10
        };
        // The magnitudes do not matter, only the relative ordering of values.
        // Convert to ranks and get a p-value for that.
        final double[] combined = DoubleStream.concat(Arrays.stream(largeX), Arrays.stream(smallY)).toArray();
        new NaturalRanking().apply(combined);
        // Extract the ranks for x and y
        final double[] rx = Arrays.copyOf(combined, largeX.length);
        final double[] ry = Arrays.copyOfRange(combined, largeX.length, combined.length);
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double p = test.kolmogorovSmirnovTest(rx, ry, false);
        Assertions.assertEquals(p, test.kolmogorovSmirnovTest(largeX, smallY, false));
    }

    @Test
    void testTwoSamplesWithInfinitiesAndTies() {
        final double[] x = {
            1, 1,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY
        };

        final double[] y = {
            1, 1,
            3, 3,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY
        };
        // Infinity is treated as any other comparable number.
        // Convert to large values and get a p-value for that.
        final double[] xm = mapInfinites(x);
        final double[] ym = mapInfinites(y);
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double p = test.kolmogorovSmirnovTest(xm, ym, false);
        Assertions.assertEquals(p, test.kolmogorovSmirnovTest(x, y, false));
    }

    @Test
    void testTwoSamplesWithOnlyInfinities() {
        final double[] x = {
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY
        };
        final double[] y = {
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY
        };
        // Infinity is treated as any other comparable number.
        // Convert to large values and get a p-value for that.
        final double[] xm = mapInfinites(x);
        final double[] ym = mapInfinites(y);
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double p = test.kolmogorovSmirnovTest(xm, ym, false);
        Assertions.assertEquals(p, test.kolmogorovSmirnovTest(x, y, false));
    }

    /**
     * Map infinite values to {@code +/- MAX_VALUE}.
     *
     * @param x Values.
     * @return the mapped values
     */
    private static double[] mapInfinites(double[] x) {
        return Arrays.stream(x).map(z ->
            Double.isInfinite(z) ? Math.copySign(Double.MAX_VALUE, z) : z).toArray();
    }

    @Test
    void testTwoSampleWithTiesAndNaN() {
        // Cf. MATH-1405
        final double[] x = {1, Double.NaN, 3, 4};
        final double[] y = {1, 2, 3, 4};
        assertThrowsIllegalArgumentException(x, y);
        assertThrowsIllegalArgumentException(y, x);
    }

    private static void assertThrowsIllegalArgumentException(double[] x, double[] y) {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        Assertions.assertThrows(IllegalArgumentException.class, () -> test.kolmogorovSmirnovStatistic(x, y), "statistic");
        Assertions.assertThrows(IllegalArgumentException.class, () -> test.kolmogorovSmirnovTest(x, y, false), "test");
        Assertions.assertThrows(IllegalArgumentException.class, () -> test.kolmogorovSmirnovTest(x, y, true), "test strict");
    }

    @Test
    void testTwoSamplesAllEqual() {
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        for (int i = 2; i < 30; i++) {
            // testing values with ties
            final double[] values = new double[i];
            Arrays.fill(values, i);
            // testing values without ties (within samples; there are ties between samples)
            final double[] ascendingValues = new double[i];
            for (int j = 0; j < ascendingValues.length; j++) {
                ascendingValues[j] = j;
            }

            Assertions.assertEquals(0.0, test.kolmogorovSmirnovStatistic(values, values), "statistic");
            Assertions.assertEquals(0.0, test.kolmogorovSmirnovStatistic(ascendingValues, ascendingValues), "statistic");

            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleExactP(0, values.length, values.length, false), "exact p");
            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleExactP(0, ascendingValues.length, ascendingValues.length, false), "exact p");
            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleExactP(0, values.length, values.length, true), "exact p (strict)");
            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleExactP(0, ascendingValues.length, ascendingValues.length, true), "exact p (strict)");

            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleApproximateP(0, values.length, values.length));
            Assertions.assertEquals(1.0, KolmogorovSmirnovTest.twoSampleApproximateP(0, ascendingValues.length, ascendingValues.length));
        }
    }

    /**
     * JIRA: MATH-1245
     *
     * Verify that D-values are not viewed as distinct when they are mathematically equal
     * when computing p-statistics for small sample tests. Reference values are from R 3.2.0.
     */
    @Test
    void testDRounding() {
        final double tol = 1e-12;
        final double[] x = {0, 2, 3, 4, 5, 6, 7, 8, 9, 12};
        final double[] y = {1, 10, 11, 13, 14, 15, 16, 17, 18};
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        Assertions.assertEquals(0.0027495724090154106, test.kolmogorovSmirnovTest(x, y, false), tol);

        final double[] x1 = {2, 4, 6, 8, 9, 10, 11, 12, 13};
        final double[] y1 = {0, 1, 3, 5, 7};
        Assertions.assertEquals(0.085914085914085896, test.kolmogorovSmirnovTest(x1, y1, false), tol);

        final double[] x2 = {4, 6, 7, 8, 9, 10, 11};
        final double[] y2 = {0, 1, 2, 3, 5};
        Assertions.assertEquals(0.015151515151515027, test.kolmogorovSmirnovTest(x2, y2, false), tol);
    }

    @Test
    void testTwoSampleEstimatePThrows() {
        final UniformRandomProvider rng = RandomSource.SPLIT_MIX_64.create();
        final BiConsumer<double[], double[]> action =
            (x, y) -> KolmogorovSmirnovTest.estimateP(x, y, 100, false, rng);

        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 3}, new double[] {2}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1}, new double[] {2, 4}), "values", "size");

        // NaN samples
        // Message current does not contain the sample name
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(new double[] {1, 3}, new double[] {2, Double.NaN}), "nan");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, Double.NaN}, new double[] {2, 4}), "nan");

        // x and y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, null));

        // x or y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, new double[] {2, 4}));
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(new double[] {1, 2}, null));

        // Not strictly positive iterations
        final double[] x = {1, 2, 3};
        for (final int i : new int[] {-1, 0, Integer.MIN_VALUE}) {
            TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> KolmogorovSmirnovTest.estimateP(x, x, i, false, rng), Integer.toString(i));
        }
    }

    /**
     * Test an example with ties in the data.  Reference data is R 3.2.0,
     * ks.boot implemented in Matching (Version 4.8-3.4, Build Date: 2013/10/28)
     */
    @Test
    void testTwoSampleEstimatePSmallSamplesWithTies() {
        final double[] x = {0, 2, 4, 6, 8, 8, 10, 15, 22, 30, 33, 36, 38};
        final double[] y = {9, 17, 20, 33, 40, 51, 60, 60, 72, 90, 101};
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create(2000);
        Assertions.assertEquals(0.0059, KolmogorovSmirnovTest.estimateP(x, y, 10000, false, rng), 1e-3);
    }

    /**
     * Reference data is R 3.2.0, ks.boot implemented in
     * Matching (Version 4.8-3.4, Build Date: 2013/10/28)
     */
    @Test
    void testTwoSampleEstimatePLargeSamples() {
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create(1000);
        Assertions.assertEquals(0.0237, KolmogorovSmirnovTest.estimateP(GAUSSIAN, GAUSSIAN2, 10000, true, rng), 1e-2);
    }

    /**
     * Test an example where D-values are close (subject to rounding).
     * Reference data is R 3.2.0, ks.boot implemented in
     * Matching (Version 4.8-3.4, Build Date: 2013/10/28)
     */
    @Test
    void testTwoSampleEstimatePRounding() {
        final double[] x = {2, 4, 6, 8, 9, 10, 11, 12, 13};
        final double[] y = {0, 1, 3, 5, 7};
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create(1000);
        Assertions.assertEquals(0.06303, KolmogorovSmirnovTest.estimateP(x, y, 10000, false, rng), 1e-2);
    }

    @Test
    void testMath1475() {
        // MATH-1475
        final double[] x = new double[] {
            0.12350159883499146, -0.2601194679737091, -1.322849988937378, 0.379696249961853,
            0.3987586498260498, -0.06924121081829071, -0.13951236009597778, 0.3213207423686981,
            0.7949811816215515, -0.15811105072498322, 0.19912190735340118, -0.46363770961761475,
            -0.20019817352294922, 0.3062838613986969, -0.3872813880443573, 0.10733723640441895,
            0.10910066962242126, 0.625770092010498, 0.2824835777282715, 0.3107619881629944,
            0.1432388722896576, -0.08056988567113876, -0.5816712379455566, -0.09488576650619507,
            -0.2154506891965866, 0.2509046196937561, -0.06600788980722427, -0.01133995596319437,
            -0.22642627358436584, -0.12150175869464874, -0.21109570562839508, -0.17732949554920197,
            -0.2769380807876587, -0.3607368767261505, -0.07842907309532166, -0.2518743574619293,
            0.035517483949661255, -0.6556509137153625, -0.360045850276947, -0.09371964633464813,
            -0.7284095883369446, -0.22719840705394745, -1.5540679693222046, -0.008972732350230217,
            -0.09106933325529099, -0.6465389132499695, 0.036245591938495636, 0.657580554485321,
            0.32453101873397827, 0.6105462908744812, 0.25256943702697754, -0.194427490234375,
            0.6238796710968018, 0.5203511118888855, -0.2708645761013031, 0.07761227339506149,
            0.5315862894058228, 0.44320303201675415, 0.6283767819404602, 0.2618369162082672,
            0.47253096103668213, 0.3889777660369873, 0.6856100559234619, 0.3007083833217621,
            0.4963226914405823, 0.08229698985815048, 0.6170856952667236, 0.7501978874206543,
            0.5744063258171082, 0.5233180522918701, 0.32654184103012085, 0.3014495372772217,
            0.4082445800304413, -0.1075737327337265, -0.018864337354898453, 0.34642550349235535,
            0.6414541602134705, 0.16678297519683838, 0.46028634905815125, 0.4151197075843811,
            0.14407725632190704, 0.41751566529273987, -0.054958608001470566, 0.4995657801628113,
            0.4485369324684143, 0.5600396990776062, 0.4098612368106842, 0.2748555839061737,
            0.2562614381313324, 0.4324824810028076 };
        final double[] y = new double[] {
            2.6881366763426717, 2.685469965655465, 2.261888917462379, -2.1933598759641226,
            -2.4279488152810145, -3.159389495849609, -2.3150004548153444, 2.468029206047388,
            2.9442494682288953, 2.653360013462529, -2.1189940659194835, -2.121635289903703,
            -2.103092459792032, -2.737034221468073, -2.203389332350286, 2.1985949039005512,
            -2.5021604073154737, 2.2732754920764533, -2.3867025598454346, 2.135919387338413,
            2.338120776050672, 2.2579794509726874, 2.083329059799027, -2.209733724709957,
            2.297192240399189, -2.201703830825843, -3.460208691996806, 2.428839296615834,
            -3.2944259224581574, 2.0654875493620883, -2.743948930837782, -2.2240674680805212,
            -3.646366778182357, -2.12513198437294, 2.979166188824589, -2.6275491570089033,
            -2.3818176136461338, 2.882096356968376, -2.2147229261558334, -3.159389495849609,
            2.312428759406432, 2.3313864098846477, -2.72802504046371, -2.4216068225364245,
            3.0119599306499123, 2.5753099009496783, -2.9200121783556843, -2.519352725437922,
            -4.133932580227538, -2.30496316762808, 2.5381353678521363, 2.4818233632136697,
            2.5277451177925685, -2.166465445816232, -2.1193897819471563, -2.109654332722425,
            3.260211545834851, -3.9527673876059013, -2.199885089466947, 2.152573429747697,
            -3.1593894958496094, 2.5479522823226795, 3.342810742466116, -2.8197184957304007,
            -2.3407900299253765, -2.3303967152728537, 2.1760131201015565, 2.143930552944634,
            2.33336231754409, 2.9126278362420575, -2.121169134387265, -2.2980208408109095,
            -2.285400411434817, -2.0742764640932903, 2.304178664095016, -2.2893825538911634,
            -3.7714771984158806, -2.7153698816026886, 2.8995011276220226, -2.158787087333056,
            -2.1045987952052547, 2.8478762016468147, -2.694578565956955, -2.696014432856399,
            -2.3190122657403496, -2.48225194403028, 3.3393947563371764, 2.7775468034263517,
            -3.396526561479875, -2.699967947404961};
        final KolmogorovSmirnovTest kst = new KolmogorovSmirnovTest();
        kst.kolmogorovSmirnovTest(x, y, false);
    }

    @Test
    void testMath1535() {
        // MATH-1535
        // Internal error in case of ties with many similar surrounding values
        // that cannot be resolved with random jitter.
        // This is no longer an issue as random jitter is not used to resolve ties.
        final double[] x = new double[] {
            0.8767630865438496, 0.9998809418147052, 0.9999999715463531, 0.9999985849345421,
            0.973584315883326, 0.9999999875782982, 0.999999999999994, 0.9999999999908233,
            1.0, 0.9999999890925574, 0.9999998345734327, 0.9999999350772448,
            0.999999999999426, 0.9999147040688201, 0.9999999999999922, 1.0,
            1.0, 0.9919050954798272, 0.8649014770687263, 0.9990869497973084,
            0.9993222540990464, 0.999999999998189, 0.9999999999999365, 0.9790934801762917,
            0.9999578695006303, 0.9999999999999998, 0.999999999996166, 0.9999999999995546,
            0.9999999999908036, 0.99999999999744, 0.9999998802655555, 0.9079334221214075,
            0.9794398308007372, 0.9999044231134367, 0.9999999999999813, 0.9999957841707683,
            0.9277678892094009, 0.999948269893843, 0.9999999886132888, 0.9999998909699096,
            0.9999099536620326, 0.9999999962217623, 0.9138936987350447, 0.9999999999779976,
            0.999999999998822, 0.999979247207911, 0.9926904388316407, 1.0,
            0.9999999999998814, 1.0, 0.9892505696426215, 0.9999996514123723,
            0.9999999999999429, 0.9999999995399116, 0.999999999948221, 0.7358264887843119,
            0.9999999994098534, 1.0, 0.9999986456748472, 1.0,
            0.9999999999921501, 0.9999999999999996, 0.9999999999999944, 0.9473070068606853,
            0.9993714060209042, 0.9999999409098718, 0.9999999592791519, 0.9999999999999805};
        final double[] y = new double[x.length];
        Arrays.fill(y, 1);
        final KolmogorovSmirnovTest kst = new KolmogorovSmirnovTest();
        final double p = kst.kolmogorovSmirnovTest(x, y, false);
        Assertions.assertTrue(0 <= p && p <= 1, () -> "Invalid p-value: " + p);
    }

    @Test
    void testMath1246() {
        final double[] x = {4, 5, 6, 7};
        final double[] y = {1, 2, 3, 4};
        final boolean strict = true;
        final KolmogorovSmirnovTest test = new KolmogorovSmirnovTest();
        final double d1 = test.kolmogorovSmirnovStatistic(x, y);
        final double d2 = test.kolmogorovSmirnovStatistic(y, x);
        final double p1 = KolmogorovSmirnovTest.twoSampleExactP((long) (d1 * x.length * y.length), x.length, y.length, strict);
        Assertions.assertEquals(d1, d2);

        // This will use the exactP method but resolve ties with jitter
        final double p2 = test.kolmogorovSmirnovTest(x, y, strict);
        final double p3 = test.kolmogorovSmirnovTest(y, x, strict);
        Assertions.assertEquals(p2, p3, "tie resolution should be stable to argument order");

        // Possible variations
        y[3] = 4.1;
        final double p4 = test.kolmogorovSmirnovTest(x, y, strict);
        y[3] = 3.9;
        final double p5 = test.kolmogorovSmirnovTest(x, y, strict);
        Assertions.assertNotEquals(p4, p5);

        Assertions.assertTrue(p1 == p4 || p1 == p5, "statistic p-value");
        Assertions.assertTrue(p2 == p4 || p2 == p5, "test p-value");
    }
}
