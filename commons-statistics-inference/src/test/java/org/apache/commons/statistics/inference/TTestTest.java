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
import java.util.stream.Stream;
import org.apache.commons.statistics.inference.TTest.Options;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for the {@link TTest}.
 */
class TTestTest {

    private interface DDDLConsumer {
        void accept(double a, double b, double c, long d);
    }

    private interface DDLDDLConsumer {
        void accept(double a, double b, long c, double d, double e, long f);
    }

    @Test
    void testOptions() {
        final Options o = Options.defaults();
        Assertions.assertEquals(AlternativeHypothesis.TWO_SIDED, o.getAlternative());
        Assertions.assertEquals(false, o.isEqualVariances());
        final Options.Builder b = Options.builder();
        Assertions.assertThrows(NullPointerException.class, () -> b.setAlternative(null));
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            for (final boolean equal : new boolean[] {true, false}) {
                final Options o1 = o.toBuilder().setAlternative(h).setEqualVariances(equal).build();
                Assertions.assertEquals(h, o1.getAlternative());
                Assertions.assertEquals(equal, o1.isEqualVariances());
                final Options o2 = o1.toBuilder().build();
                Assertions.assertEquals(h, o2.getAlternative());
                Assertions.assertEquals(equal, o2.isEqualVariances());
            }
        }
    }

    @Test
    void testOneSampleDatasetThrows() {
        assertOneSampleDatasetThrows(TTest::statistic);
        assertOneSampleDatasetThrows(TTest::test);
    }

    private static void assertOneSampleDatasetThrows(DDDLConsumer action) {
        final double mu = 0;
        final double m = 0;
        final double v = 1;
        final long n = 5;
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(mu, m, v, 1), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(mu, m, -1, n), "negative");
    }

    @ParameterizedTest
    @MethodSource
    void testOneSampleDataset(double mu, double m, double v, long n, double statistic, double[] p) {
        final double s = TTest.statistic(mu, m, v, n);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final Options.Builder b = Options.builder();
        int i = 0;
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            b.setAlternative(h);
            // Test the default if possible
            final TTest.Result r =
                    h == Options.defaults().getAlternative() ?
                    TTest.test(mu, m, v, n) :
                    TTest.test(mu, m, v, n, b.build());
            Assertions.assertEquals(s, r.getStatistic(), "statistic");
            Assertions.assertEquals(n - 1, r.getDegreesOfFreedom(), "Degrees of freedom");
            TestUtils.assertProbability(p[i++], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testOneSampleDataset() {
        // Extract the mean, variance, size from the observations
        return testOneSample().map(a -> {
            final Object[] args = a.get();
            final double[] sample = (double[]) args[1];
            final Object[] args2 = new Object[args.length + 2];
            final double m = StatisticUtils.mean(sample);
            final double v = StatisticUtils.variance(sample, m);
            args2[0] = args[0];
            args2[1] = m;
            args2[2] = v;
            args2[3] = sample.length;
            System.arraycopy(args, 2, args2, 4, args.length - 2);
            return Arguments.of(args2);
        });
    }

    @Test
    void testOneSampleThrows() {
        assertOneSampleThrows(TTest::statistic);
        assertOneSampleThrows(TTest::test);
    }

    private static void assertOneSampleThrows(BiConsumer<Double, double[]> action) {
        final double mu = 0;
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(mu, new double[1]), "values", "size");
    }

    @ParameterizedTest
    @MethodSource
    void testOneSample(double mu, double[] sample, double statistic, double[] p) {
        final double s = TTest.statistic(mu, sample);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final Options.Builder b = Options.builder();
        int i = 0;
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            b.setAlternative(h);
            // Test the default if possible
            final TTest.Result r =
                    h == Options.defaults().getAlternative() ?
                    TTest.test(mu, sample) :
                    TTest.test(mu, sample, b.build());
            Assertions.assertEquals(s, r.getStatistic(), "statistic");
            Assertions.assertEquals(sample.length - 1, r.getDegreesOfFreedom(), "Degrees of freedom");
            TestUtils.assertProbability(p[i++], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testOneSample() {
        // p-values are in the AlternativeHypothesis enum order: two-sided, greater, less
        final Stream.Builder<Arguments> builder = Stream.builder();
        // R 3.4.0 t.test
        builder.add(Arguments.of(100,
            new double[] {93, 103, 95, 101, 91, 105, 96, 94, 101,  88, 98, 94, 101, 92, 95},
            -2.8197644534585268872,
            new double[] {0.013639058587288887, 0.99318047070635562, 0.0068195292936444434}));
        builder.add(Arguments.of(1,
            new double[] {4, 6, 9, -1, 6, 3, 2, 8},
            3.1142442579521292245,
            new double[] {0.016979974684470666, 0.008489987342235333, 0.99151001265776473}));
        builder.add(Arguments.of(0,
            new double[] {2, 0, 6, 6, 3, 3, 2, 3, -6, 6, 6, 6, 3, 0, 1, 1, 0, 2, 3, 3},
            3.864855355409694937,
            new double[] {0.0010432740392736656, 0.00052163701963683278, 0.99947836298036319}));
        return builder.build();
    }

    @Test
    void testPairedSampleThrows() {
        assertPairedSampleThrows(TTest::pairedStatistic);
        assertPairedSampleThrows(TTest::pairedTest);
    }

    private static void assertPairedSampleThrows(BiConsumer<double[], double[]> action) {
        final double[] sample = new double[5];
        final double[] tooSmall = {1.0};
        final double[] unequalSize = new double[4];
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(sample, tooSmall), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(tooSmall, sample), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(sample, unequalSize), "values", "size", "mismatch");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(unequalSize, sample), "values", "size", "mismatch");
    }

    @ParameterizedTest
    @MethodSource
    void testPairedSample(double[] sample1, double[] sample2, double statistic, double[] p) {
        final double s = TTest.pairedStatistic(sample1, sample2);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final Options.Builder b = Options.builder();
        int i = 0;
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            b.setAlternative(h);
            // Test the default if possible
            final TTest.Result r =
                    h == Options.defaults().getAlternative() ?
                    TTest.pairedTest(sample1, sample2) :
                    TTest.pairedTest(sample1, sample2, b.build());
            Assertions.assertEquals(s, r.getStatistic(), "statistic");
            Assertions.assertEquals(sample1.length - 1, r.getDegreesOfFreedom(), "Degrees of freedom");
            TestUtils.assertProbability(p[i++], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testPairedSample() {
        // p-values are in the AlternativeHypothesis enum order: two-sided, greater, less
        final Stream.Builder<Arguments> builder = Stream.builder();
        final double[] s1 = {1, 3, 5, 7};
        final double[] s2 = {0, 6, 11, 2};
        final double[] s3 = {5, 7, 8, 10};
        // R 3.4.0 t.test(s1, s2, paired=TRUE, alternative=['t', 'g', 'l'])
        builder.add(Arguments.of(s1, s2,
            -0.31333978072025608919,
            new double[] {0.77454429581922446, 0.61272785209038783, 0.38727214790961223}));
        builder.add(Arguments.of(s1, s3,
            -12.124355652982142573,
            new double[] {0.0012077024702717076, 0.99939614876486416, 0.0006038512351358538}));
        builder.add(Arguments.of(s3, s2,
            1.1489125293076056789,
            new double[] {0.33388698610855666, 0.16694349305427833, 0.83305650694572164}));
        return builder.build();
    }

    @Test
    void testTwoSampleDatasetThrows() {
        assertTwoSampleDatasetThrows((a, b, c, d, e, f) -> TTest.statistic(a, b, c, d, e, f, false));
        assertTwoSampleDatasetThrows(TTest::test);
    }

    private static void assertTwoSampleDatasetThrows(DDLDDLConsumer action) {
        final double m = 0;
        final double v = 1;
        final long n = 5;
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(m, v, n, m, v, 1), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(m, v, 1, m, v, n), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(m, -1, n, m, v, n), "negative");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(m, v, n, m, -1, n), "negative");
    }

    @ParameterizedTest
    @MethodSource
    void testTwoSampleDataset(double m1, double v1, long n1, double m2, double v2, long n2,
            boolean equal, double statistic, double df, double[] p) {
        final double s = TTest.statistic(m1, v1, n1, m2, v2, n2, equal);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final Options.Builder b = Options.builder().setEqualVariances(equal);
        int i = 0;
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            b.setAlternative(h);
            // Test the default if possible
            final TTest.Result r =
                    equal == Options.defaults().isEqualVariances() &&
                    h == Options.defaults().getAlternative() ?
                    TTest.test(m1, v1, n1, m2, v2, n2) :
                    TTest.test(m1, v1, n1, m2, v2, n2, b.build());
            Assertions.assertEquals(s, r.getStatistic(), "statistic");
            TestUtils.assertRelativelyEquals(df, r.getDegreesOfFreedom(), 1e-15, "Degrees of freedom");
            TestUtils.assertProbability(p[i++], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testTwoSampleDataset() {
        // p-values are in the AlternativeHypothesis enum order: two-sided, greater, less
        // scipy.stats (version 1.9.3)
        // Note: scipy uses the standard deviation:
        // Examples are from the scipy documentation.
        // import numpy as np
        // from scipy.stats import ttest_ind_from_stats as ttest
        // ttest(15, np.sqrt(87.5), 13, 12, np.sqrt(39), 11,
        //       equal_var=True, alternative=<'two-sided', 'less', 'greater'>)
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(15, 87.5, 13, 12, 39, 11, true,
            0.9051358093310269, 22,
            new double[] {0.3751996797581487, 0.18759983987907436, 0.8124001601209256}));
        builder.add(Arguments.of(15, 87.5, 13, 12, 39, 11, false,
            0.9358461935556048, 20.984611233429924,
            new double[] {0.35999818693244245, 0.17999909346622123, 0.8200009065337788}));
        builder.add(Arguments.of(0.2, 0.161073, 150, 0.225, 0.175251, 200, true,
            -0.5627187905196761, 348,
            new double[] {0.5739887114209541, 0.713005644289523, 0.28699435571047704}));
        builder.add(Arguments.of(0.2, 0.161073, 150, 0.225, 0.175251, 200, false,
            -0.5661276301071694, 327.90436123021186,
            new double[] {0.5716942537704801, 0.71415287311476, 0.28584712688524005}));
        return builder.build();
    }

    @Test
    void testTwoSampleThrows() {
        assertTwoSampleThrows((a, b) -> TTest.statistic(a, b, false));
        assertTwoSampleThrows(TTest::test);
    }

    private static void assertTwoSampleThrows(BiConsumer<double[], double[]> action) {
        final double[] sample = new double[5];
        final double[] tooSmall = {1.0};
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(sample, tooSmall), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(tooSmall, sample), "values", "size");
    }

    @ParameterizedTest
    @MethodSource
    void testTwoSample(double[] s1, double[] s2, boolean equal, double statistic, double df, double[] p) {
        final double s = TTest.statistic(s1, s2, equal);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final Options.Builder b = Options.builder().setEqualVariances(equal);
        int i = 0;
        for (final AlternativeHypothesis h : AlternativeHypothesis.values()) {
            b.setAlternative(h);
            // Test the default if possible
            final TTest.Result r =
                    equal == Options.defaults().isEqualVariances() &&
                    h == Options.defaults().getAlternative() ?
                    TTest.test(s1, s2) :
                    TTest.test(s1, s2, b.build());
            Assertions.assertEquals(s, r.getStatistic(), "statistic");
            TestUtils.assertRelativelyEquals(df, r.getDegreesOfFreedom(), 1e-15, "Degrees of freedom");
            TestUtils.assertProbability(p[i++], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testTwoSample() {
        // p-values are in the AlternativeHypothesis enum order: two-sided, greater, less
        final Stream.Builder<Arguments> builder = Stream.builder();
        // R version 3.4.0
        // t.test(s1, s2, alternative='t', var.equal=TRUE)
        final double[] s1 = {7, -4, 18, 17, -3, -5, 1, 10, 11, -2};
        final double[] s2 = {-1, 12, -1, -3, 3, -5, 5, 2, -11, -1, -3};
        builder.add(Arguments.of(s1, s2, true,
            1.6341082415908594339, 19,
            new double[] {0.11869682666685942, 0.05934841333342971, 0.94065158666657034}));
        builder.add(Arguments.of(s1, s2, false,
            1.6037172876755148021, 15.590512968733776233,
            new double[] {0.12883936962193396, 0.064419684810966979, 0.93558031518903295}));
        final double[] s3 = {2, 4, 6, 8, 10, 97};
        final double[] s4 = {4, 6, 8, 10, 16};
        builder.add(Arguments.of(s3, s4, true,
            0.7309631008575527833, 9,
            new double[] {0.4833963785800246, 0.2416981892900123, 0.7583018107099877}));
        builder.add(Arguments.of(s3, s4, false,
            0.80568260610405273425, 5.1827667502020595691,
            new double[] {0.45578652613550491, 0.22789326306775246, 0.77210673693224752}));
        // small samples (t is the same for both variance options)
        final double[] s5 = {1, 3};
        final double[] s6 = {4, 5};
        builder.add(Arguments.of(s5, s6, true,
            -2.2360679774997898051, 2,
            new double[] {0.1548457452714834, 0.92257712736425823, 0.077422872635741699}));
        builder.add(Arguments.of(s5, s6, false,
            -2.2360679774997898051, 1.4705882352941177516,
            new double[] {0.19872738893452604, 0.90063630553273699, 0.099363694467263022}));
        return builder.build();
    }
}
