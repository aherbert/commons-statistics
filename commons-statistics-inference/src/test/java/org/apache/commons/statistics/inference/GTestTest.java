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
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link GTest}.
 *
 * <p>Data for the tests are from p64-69 in: McDonald, J.H. 2009. Handbook of
 * Biological Statistics (2nd ed.). Sparky House Publishing, Baltimore,
 * Maryland.
 */
class GTestTest {

    private final GTest testStatistic = new GTest();

    // R:
    // install.packages("DescTools")  // Not for R 3.4.0
    // library(DescTools)
    // GTest.test(Matriz)
    //
    // install.packages("AMR")
    // library(AMR)
    // g.test(x, y, p = , rescale.p = TRUE)
    // Looking at the code Matrix support does not compute using p * log(p) for row/column sums
    //
    // install.packages("RVAideMemoire")  // Not for R 3.4.0
    // library(RVAideMemoire)
    // G.test(...)

    // Spreadsheets to compute G available from:
    // http://www.biostathandbook.com/gtestgof.html
    // http://www.biostathandbook.com/gtestind.html

    @Test
    void testGTestThrows() {
        final double[] expected = {1, 1};
        final long[] observed = {1, 1};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.gTest(expected, observed, alpha), "gTest");

        assertChiSquareTestThrows((x, y) -> testStatistic.g(x, y));
        assertChiSquareTestThrows((x, y) -> testStatistic.gTest(x, y));
        assertChiSquareTestThrows((x, y) -> testStatistic.gTest(x, y, 0.05));
    }

    private static void assertChiSquareTestThrows(BiConsumer<double[], long[]> action) {
        // Samples must be present, i.e. length > 1
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1}, new long[] {1, 2}), "values", "size");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(new double[] {1, 1}, new long[] {1}), "values", "size");

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
    void testGTest(double[] exp, long[] obs, double g, double p) {
        TestUtils.assertRelativelyEquals(g, testStatistic.g(exp, obs), 1e-14, "statistic");
        TestUtils.assertProbability(p, testStatistic.gTest(exp, obs), 1e-14, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.gTest(exp, obs, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.gTest(exp, obs, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testGTest() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(
            new double[] {3, 1},
            new long[] {423, 133},
            0.34872172558757, 0.554837623613194));
        builder.add(Arguments.of(
            new double[] {0.54, 0.40, 0.05, 0.01},
            new long[] {70, 79, 3, 4},
            13.1447992204914, 0.00433370617191827));
        return builder.build();
    }

    @Test
    void testGTestTableThrows() {
        final long[][] counts = {{1, 2}, {3, 4}};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.gTest(counts, alpha), "gTest");

        assertChiSquareTestTableThrows(x -> testStatistic.g(x));
        assertChiSquareTestTableThrows(x -> testStatistic.gTest(x));
        assertChiSquareTestTableThrows(x -> testStatistic.gTest(x, 0.05));
    }

    private static void assertChiSquareTestTableThrows(Consumer<long[][]> action) {
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
    void testGTestTable(long[][] counts, double g, double p, double gEps, double pEps) {
        TestUtils.assertRelativelyEquals(g, testStatistic.g(counts), gEps, "statistic");
        TestUtils.assertProbability(p, testStatistic.gTest(counts), pEps, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.gTest(counts, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.gTest(counts, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testGTestTable() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(
            new long[][] {{268, 199, 42}, {807, 759, 184}},
            7.30081707585487, 0.0259805125855789, 8e-13, 3e-12));
        builder.add(Arguments.of(
            new long[][] {{127, 99, 264}, {116, 67, 161}},
            6.2272884037302, 0.0444387162511812, 4e-13, 8e-13));
        builder.add(Arguments.of(
            new long[][] {{190, 149}, {42, 49}},
            2.81867561650324, 0.0931732525091412, 2e-13, 3e-13));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testScaling(double[] expected, long[] observed) {
        final double g = testStatistic.g(expected, observed);
        // Scale observed
        for (final int scale : new int[] {2, 3, 4}) {
            final long[] o = Arrays.stream(observed).map(x -> x * scale).toArray();
            TestUtils.assertRelativelyEquals(scale * g, testStatistic.g(expected, o), 2e-15, () -> "scale o: " + scale);
        }
        for (final double scale : new double[] {0.25, 0.5, 2}) {
            final double[] e = Arrays.stream(expected).map(x -> x * scale).toArray();
            TestUtils.assertRelativelyEquals(g, testStatistic.g(e, observed), 1e-15, () -> "scale e: " + scale);
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
