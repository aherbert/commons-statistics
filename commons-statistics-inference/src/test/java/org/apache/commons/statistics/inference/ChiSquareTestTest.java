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
import java.util.function.BiConsumer;
import java.util.stream.Stream;
import org.apache.commons.statistics.inference.ChiSquareTest.Options;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link ChiSquareTest}.
 */
class ChiSquareTestTest {

    @Test
    void testOptions() {
        final Options o = Options.defaults();
        Assertions.assertEquals(0, o.getDegreesOfFreedomAdjustment());
        final Options.Builder b = Options.builder();
        Assertions.assertThrows(IllegalArgumentException.class, () -> b.setDegreesOfFreedomAdjustment(-1));
        for (int i = 0; i < 3; i++) {
            final Options o1 = o.toBuilder().setDegreesOfFreedomAdjustment(i).build();
            Assertions.assertEquals(i, o1.getDegreesOfFreedomAdjustment());
            final Options o2 = o1.toBuilder().build();
            Assertions.assertEquals(i, o2.getDegreesOfFreedomAdjustment());
        }
    }

    @Test
    void testChiSquareTestUniformThrows() {
        GTestTest.assertTestUniformThrows(ChiSquareTest::statistic);
        GTestTest.assertTestUniformThrows(ChiSquareTest::test);
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ChiSquareTest.statistic(new long[] {1}), "values", "size");

        // Degrees of freedom error.
        // This occurs before a values size error for the 'test' method.
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> ChiSquareTest.test(new long[] {1}), "degrees", "of", "freedom");
        final Options options = Options.builder().setDegreesOfFreedomAdjustment(1).build();
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> ChiSquareTest.test(new long[] {1, 1}, options), "degrees", "of", "freedom");
    }

    @ParameterizedTest
    @MethodSource
    void testChiSquareTestUniform(long[] obs, double statistic, double[] p) {
        final double s = ChiSquareTest.statistic(obs);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        SignificanceResult r = ChiSquareTest.test(obs);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p[0], r.getPValue(), 1e-14, "p-value");
        // Changing degrees of freedom
        final Options.Builder b = Options.builder();
        for (int i = 1; i < p.length; i++) {
            r = ChiSquareTest.test(obs, b.setDegreesOfFreedomAdjustment(i).build());
            Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
            TestUtils.assertProbability(p[i], r.getPValue(), 2e-14, "p-value");
        }
    }

    static Stream<Arguments> testChiSquareTestUniform() {
        // Find all test cases where the expected is uniform,
        // then remove the expected from the arguments.
        return testChiSquareTest().filter(a -> {
            // Only those cases where the expected is uniform
            final double[] exp = (double[]) a.get()[0];
            for (int i = 1; i < exp.length; i++) {
                if (exp[0] != exp[i]) {
                    return false;
                }
            }
            return true;
        }).map(arg -> Arguments.of(Arrays.copyOfRange(arg.get(), 1, arg.get().length)));
    }

    @Test
    void testChiSquareTestThrows() {
        GTestTest.assertTestThrows(ChiSquareTest::statistic);
        GTestTest.assertTestThrows(ChiSquareTest::test);
        // Degrees of freedom error
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> ChiSquareTest.test(new double[] {1, 1}, new long[] {1}), "degrees", "of", "freedom");
        final Options options = Options.builder().setDegreesOfFreedomAdjustment(1).build();
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> ChiSquareTest.test(new double[] {1, 1}, new long[] {1, 1}, options), "degrees", "of", "freedom");
    }

    @ParameterizedTest
    @MethodSource
    void testChiSquareTest(double[] exp, long[] obs, double statistic, double[] p) {
        final double s = ChiSquareTest.statistic(exp, obs);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        SignificanceResult r = ChiSquareTest.test(exp, obs);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p[0], r.getPValue(), 1e-14, "p-value");
        // Changing degrees of freedom
        final Options.Builder b = Options.builder();
        for (int i = 1; i < p.length; i++) {
            r = ChiSquareTest.test(exp, obs, b.setDegreesOfFreedomAdjustment(i).build());
            Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
            TestUtils.assertProbability(p[i], r.getPValue(), 2e-14, "p-value");
        }
    }

    static Stream<Arguments> testChiSquareTest() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Target values computed using R version 3.4.0
        // Some assembly required ;-)
        // chi2 = sum((obs - exp)^2/exp)
        // p = pchisq(sum((obs - exp)^2/exp), length(obs) - 1, lower.tail=FALSE)
        builder.add(Arguments.of(
            new double[] {10, 10, 10},
            new long[] {10, 9, 11},
            0.2000000000000000111,
            new double[] {0.90483741803595951758}));
        // Requires scaling:
        // scale = sum(obs) / sum(exp)
        // sum((obs - exp*scale)^2/(exp*scale))
        // pchisq(sum((obs - exp*scale)^2/(exp*scale)), length(obs) - 1, lower.tail=FALSE)
        builder.add(Arguments.of(
            new double[] {485, 541, 82, 61, 37},
            new long[] {500, 623, 72, 70, 31},
            9.0233079364273880429,
            new double[] {0.060519526474536095018}));
        // Large test statistic
        builder.add(Arguments.of(
            new double[] {3389119.5, 649136.6, 285745.4, 25357364.76, 11291189.78, 543628.0,
                232921.0, 437665.75},
            new long[] {2372383, 584222, 257170, 17750155, 7903832, 489265, 209628, 393899},
            114875.90421929006698,
            new double[] {0, 0, 0}));
        // scipy.stats 1.9.3
        // Uniform
        // scipy.stats.chisquare([21, 18, 24, 17], ddof=[0, 1, 2])
        builder.add(Arguments.of(
            new double[] {0.25, 0.25, 0.25, 0.25},
            new long[] {21, 18, 24, 17},
            1.5,
            new double[] {0.6822703303362125, 0.4723665527410149, 0.22067136191984324}));
        builder.add(Arguments.of(
            new double[] {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
            new long[] {21, 18, 28, 17, 13, 27, 13, 30},
            15.275449101796406,
            new double[] {0.03262625605539875, 0.01821919797535816, 0.00924799820716602}));
        // Zero counts
        builder.add(Arguments.of(
            new double[] {0.25, 0.25, 0.25, 0.25},
            new long[] {11, 8, 9, 0},
            10.0,
            new double[] {0.01856613546304325, 0.006737946999085468, 0.001565402258002549}));
        builder.add(Arguments.of(
            new double[] {1, 1, 1},
            new long[] {7, 0, 11},
            10.3333333333333333,
            new double[] {0.005703548998007402, 0.001306490437268988}));
        return builder.build();
    }

    @Test
    void testChiSquareTestTableThrows() {
        GTestTest.assertTestTableThrows(ChiSquareTest::statistic);
        GTestTest.assertTestTableThrows(ChiSquareTest::test);
    }

    @ParameterizedTest
    @MethodSource
    void testChiSquareTestTable(long[][] counts, double statistic, double p) {
        final double s = ChiSquareTest.statistic(counts);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final SignificanceResult r = ChiSquareTest.test(counts);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p, r.getPValue(), 1e-14, "p-value");
    }

    static Stream<Arguments> testChiSquareTestTable() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Target values computed using R version 3.4.0
        // M <- as.table(rbind(c(40, 22, 43), c(91, 21, 28), c(60, 10, 22)))
        // chisq.test(M)
        builder.add(Arguments.of(
            new long[][] {{40, 22, 43}, {91, 21, 28}, {60, 10, 22}},
            22.709027688037505044, 0.00014475146013430618));
        builder.add(Arguments.of(
            new long[][] {{10, 15}, {30, 40}, {60, 90}},
            0.16896551724137939821, 0.91898749985230743));
        /** Contingency table containing zeros - PR # 32531 */
        builder.add(Arguments.of(
            new long[][] {{40, 0, 4}, {91, 1, 2}, {60, 2, 0}},
            9.6744466226332050951, 0.046283577060288794));
        // 2*m table: equal counts
        builder.add(Arguments.of(
            new long[][] {{10, 12, 12, 10}, {5, 15, 14, 10}},
            2.1538461538461537437, 0.54109635560419655));
        // 2*m table: unequal counts
        builder.add(Arguments.of(
            new long[][] {{10, 12, 12, 10, 15}, {15, 10, 10, 15, 5}},
            7.2321893822664256035, 0.12411513346702596));
        return builder.build();
    }

    @Test
    void testChiSquareTestTwoSampleThrows() {
        testChiSquareTestTwoSampleThrows(ChiSquareTest::statistic);
        testChiSquareTestTwoSampleThrows(ChiSquareTest::test);
    }

    private static void testChiSquareTestTwoSampleThrows(BiConsumer<long[], long[]> action) {
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1}, new long[] {1, 2}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, 1}, new long[] {1}), "values", "size");

        // Samples not same size, i.e. cannot be paired
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, 1}, new long[] {1, 2, 3}), "values", "size", "mismatch");

        // negative
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, -2}, new long[] {1, -1}), "negative", "-2");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, 1}, new long[] {1, -1}), "negative", "-1");

        // Sum of column/row zero
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, 0, 3}, new long[] {2, 0, 4}), "row", "1", "zero");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {0, 0, 0}, new long[] {2, 3, 4}), "column", "0", "zero");

        // x and y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, null));

        // x or y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, new long[] {1, 2}));
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(new long[] {1, 1}, null));
    }

    @ParameterizedTest
    @MethodSource
    void testChiSquareTestTwoSample(long[] observed1, long[] observed2, double statistic, double p) {
        final double s = ChiSquareTest.statistic(observed1, observed2);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        final SignificanceResult r = ChiSquareTest.test(observed1, observed2);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p, r.getPValue(), 1e-14, "p-value");
    }

    static Stream<Arguments> testChiSquareTestTwoSample() {
        // The two sample chi-square test is the same as a n*m contingency table where n=2.
        // Reuse the same data.
        return testChiSquareTestTable()
            .filter(arg -> ((long[][]) arg.get()[0]).length == 2)
            .map(arg -> {
                // Split the long[][] into two longs
                final Object[] a = arg.get();
                final long[][] counts = (long[][]) a[0];
                final Object[] b = new Object[a.length + 1];
                b[0] = counts[0];
                b[1] = counts[1];
                System.arraycopy(a, 1, b, 2, a.length - 1);
                return Arguments.of(b);
            });
    }
}
