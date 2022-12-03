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
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.apache.commons.statistics.inference.GTest.Options;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link GTest}.
 */
class GTestTest {

    // Note: No implementation in R or scipy.stats

    // Some data for the tests are from p64-69 in: McDonald, J.H. 2009. Handbook of
    // Biological Statistics (2nd ed.). Sparky House Publishing, Baltimore,
    // Maryland. This is available online and provides spreadsheets to compute G:
    // https://www.biostathandbook.com/gtestgof.html
    // https://www.biostathandbook.com/gtestind.html

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
    void testGTestUniformThrows() {
        assertTestUniformThrows(GTest::statistic);
        assertTestUniformThrows(GTest::test);
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> GTest.statistic(new long[] {1}), "values", "size");

        // Degrees of freedom error.
        // This occurs before a values size error for the 'test' method.
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> GTest.test(new long[] {1}), "degrees", "of", "freedom");
        final Options options = Options.builder().setDegreesOfFreedomAdjustment(1).build();
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> GTest.test(new long[] {1, 1}, options), "degrees", "of", "freedom");
    }

    /**
     * Assert the test for goodness-of-fit to a uniform distribution of the
     * {@code observed} throws. This method is shared by the {@link GTest} and
     * {@link ChiSquareTest}.
     *
     * @param action Test for goodness-of-fit.
     */
    static void assertTestUniformThrows(Consumer<long[]> action) {
        // negative observed
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[] {1, -1}), "negative", "-1");
        // no data
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[5]), "no", "data");
        // x is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null));
    }

    @ParameterizedTest
    @MethodSource
    void testGTestUniform(long[] obs, double statistic, double[] p) {
        final double s = GTest.statistic(obs);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        SignificanceResult r = GTest.test(obs);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p[0], r.getPValue(), 1e-14, "p-value");
        // Changing degrees of freedom
        final Options.Builder b = Options.builder();
        for (int i = 1; i < p.length; i++) {
            r = GTest.test(obs, b.setDegreesOfFreedomAdjustment(i).build());
            Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
            TestUtils.assertProbability(p[i], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testGTestUniform() {
        // Find all test cases where the expected is uniform,
        // then remove the expected from the arguments.
        return testGTest().filter(a -> {
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
    void testGTestThrows() {
        assertTestThrows(GTest::statistic);
        assertTestThrows(GTest::test);
        // Degrees of freedom error
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> GTest.test(new double[] {1, 1}, new long[] {1}), "degrees", "of", "freedom");
        final Options options = Options.builder().setDegreesOfFreedomAdjustment(1).build();
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> GTest.test(new double[] {1, 1}, new long[] {1, 1}, options), "degrees", "of", "freedom");
    }

    /**
     * Assert the test for goodness-of-fit on the {@code expected} and
     * {@code observed} throws. This method is shared by the {@link GTest} and
     * {@link ChiSquareTest}.
     *
     * @param action Test for goodness-of-fit.
     */
    static void assertTestThrows(BiConsumer<double[], long[]> action) {
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1}, new long[] {1, 2}), "values", "size");
        // May be a size mismatch or degrees of freedom error
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(new double[] {1, 1, 1}, new long[] {1, 2}), "values", "size");

        // Samples not same size, i.e. cannot be paired
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 1}, new long[] {1, 2, 3}), "values", "size", "mismatch");

        // not strictly positive expected
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {0, 1}, new long[] {1, 1}), "0.0");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {-0.5, 1}, new long[] {1, 1}), "-0.5");
        // negative observed
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 1}, new long[] {1, -1}), "negative", "-1");

        // no data
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 2, 3}, new long[3]), "no", "data");

        // x and y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, null));

        // x or y is null
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null, new long[] {1, 2}));
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(new double[] {1, 1}, null));
    }

    @ParameterizedTest
    @MethodSource
    void testGTest(double[] exp, long[] obs, double statistic, double[] p) {
        final double s = GTest.statistic(exp, obs);
        TestUtils.assertRelativelyEquals(statistic, s, 1e-14, "statistic");
        SignificanceResult r = GTest.test(exp, obs);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p[0], r.getPValue(), 1e-14, "p-value");
        // Changing degrees of freedom
        final Options.Builder b = Options.builder();
        for (int i = 1; i < p.length; i++) {
            r = GTest.test(exp, obs, b.setDegreesOfFreedomAdjustment(i).build());
            Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
            TestUtils.assertProbability(p[i], r.getPValue(), 1e-14, "p-value");
        }
    }

    static Stream<Arguments> testGTest() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(
            new double[] {3, 1},
            new long[] {423, 133},
            0.34872172558757,
            new double[] {0.554837623613194}));
        builder.add(Arguments.of(
            new double[] {0.54, 0.40, 0.05, 0.01},
            new long[] {70, 79, 3, 4},
            13.1447992204914,
            new double[] {0.00433370617191827, 0.00139843767734328, 0.000288318409713092}));
        // Uniform
        builder.add(Arguments.of(
            new double[] {0.25, 0.25, 0.25, 0.25},
            new long[] {21, 18, 24, 17},
            1.48199945461987,
            new double[] {0.686431034854903, 0.476637170057175, 0.223461950936231}));
        builder.add(Arguments.of(
            new double[] {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
            new long[] {21, 18, 28, 17, 13, 27, 13, 30},
            15.4039737622005,
            new double[] {0.0311559660417338, 0.0173370059187284, 0.00876873786129152}));
        // Zero counts
        builder.add(Arguments.of(
            new double[] {0.25, 0.25, 0.25, 0.25},
            new long[] {11, 8, 9, 0},
            16.6038347133959,
            new double[] {0.000852489082734623, 0.000248040788218953}));
        builder.add(Arguments.of(
            new double[] {1, 1, 1},
            new long[] {7, 0, 11},
            15.4930971961286,
            new double[] {0.000432231774708477, 0.0000828071000018937}));
        return builder.build();
    }

    @Test
    void testGTestTableThrows() {
        assertTestTableThrows(GTest::statistic);
        assertTestTableThrows(GTest::test);
    }

    /**
     * Assert the test of independence on the n-by-m contingency table throws.
     * This method is shared by the {@link GTest} and {@link ChiSquareTest}.
     *
     * @param action Test of independence.
     */
    static void assertTestTableThrows(Consumer<long[][]> action) {
        // insufficient data
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{40, 22, 43}}), "categories", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{40}, {40}, {30}, {10}}), "values", "size");

        // non-rectangular input
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{40, 22, 43}, {91, 21, 28}, {60, 10}}), "non", "rectangular");

        // negative counts
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][]  {{10, -2}, {30, 40}, {60, 90}}), "negative", "-2");

        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{1, -2}, {1, -1}}), "negative", "-2");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{1, 1}, {1, -1}}), "negative", "-1");

        // Sum of column/row zero
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{1, 0, 3}, {2, 0, 4}}), "column", "1", "zero");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new long[][] {{0, 0, 0}, {2, 3, 4}}), "row", "0", "zero");

        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null));
    }

    @ParameterizedTest
    @MethodSource
    void testGTestTable(long[][] counts, double statistic, double p, double statisticEps, double pEps) {
        final double s = GTest.statistic(counts);
        TestUtils.assertRelativelyEquals(statistic, s, statisticEps, "statistic");
        final SignificanceResult r = GTest.test(counts);
        Assertions.assertEquals(s, r.getStatistic(), "Difference statistic");
        TestUtils.assertProbability(p, r.getPValue(), pEps, "p-value");
    }

    static Stream<Arguments> testGTestTable() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(
            new long[][] {{268, 199, 42}, {807, 759, 184}},
            7.30081707585487, 0.0259805125855789, 8e-13, 3e-12));
        builder.add(Arguments.of(
            new long[][] {{127, 99, 264}, {116, 67, 161}},
            6.2272884037302, 0.0444387162511812, 4e-13, 9e-13));
        builder.add(Arguments.of(
            new long[][] {{190, 149}, {42, 49}},
            2.81867561650324, 0.0931732525091412, 2e-13, 3e-13));
        builder.add(Arguments.of(
                new long[][] {
                    {268, 199, 43, 42},
                    {299, 178, 77, 67},
                    {345, 211, 56, 54}
                },
                17.5892726476959, 0.00734493823990698, 6e-13, 4e-12));
        // Zero counts
        builder.add(Arguments.of(
                new long[][] {{268, 199, 0, 42}, {807, 0, 759, 184}},
                986.235141341116, 1.74232879611265000E-213, 8e-13, 4e-12));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testScaling(double[] expected, long[] observed) {
        final double g = GTest.statistic(expected, observed);
        // Scale observed
        for (final int scale : new int[] {2, 3, 4}) {
            final long[] o = Arrays.stream(observed).map(x -> x * scale).toArray();
            TestUtils.assertRelativelyEquals(scale * g, GTest.statistic(expected, o), 2e-15, () -> "scale o: " + scale);
        }
        for (final double scale : new double[] {0.25, 0.5, 2}) {
            final double[] e = Arrays.stream(expected).map(x -> x * scale).toArray();
            TestUtils.assertRelativelyEquals(g, GTest.statistic(e, observed), 1e-15, () -> "scale e: " + scale);
        }
    }

    static Stream<Arguments> testScaling() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Sum to a power of 2 (32)
        builder.add(Arguments.of(new double[] {8, 8, 8, 8}, new long[] {9, 7, 10, 6}));
        builder.add(Arguments.of(new double[] {6, 7, 8, 9}, new long[] {9, 7, 10, 6}));
        builder.add(Arguments.of(new double[] {0.1, 0.2, 0.3, 0.4, 0.5}, new long[] {1, 3, 2, 5, 4}));
        return builder.build();
    }
}
