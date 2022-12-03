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
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link ChiSquareTest}.
 */
class ChiSquareTestTest {

    private final ChiSquareTest testStatistic = new ChiSquareTest();

    @Test
    void testChiSquareTestThrows() {
        final double[] expected = {1, 1};
        final long[] observed = {1, 1};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.chiSquareTest(expected, observed, alpha), "chiSquareTest");

        assertChiSquareTestThrows((x, y) -> testStatistic.chiSquare(x, y));
        assertChiSquareTestThrows((x, y) -> testStatistic.chiSquareTest(x, y));
        assertChiSquareTestThrows((x, y) -> testStatistic.chiSquareTest(x, y, 0.05));
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
    void testChiSquare(double[] exp, long[] obs, double chiSquare, double p) {
        TestUtils.assertRelativelyEquals(chiSquare, testStatistic.chiSquare(exp, obs), 1e-14, "statistic");
        TestUtils.assertProbability(p, testStatistic.chiSquareTest(exp, obs), 1e-14, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.chiSquareTest(exp, obs, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.chiSquareTest(exp, obs, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testChiSquare() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Target values computed using R version 3.4.0
        // Some assembly required ;-)
        // chi2 = sum((obs - exp)^2/exp)
        // p = pchisq(sum((obs - exp)^2/exp), length(obs) - 1, lower.tail=FALSE)
        builder.add(Arguments.of(
            new double[] {10, 10, 10},
            new long[] {10, 9, 11},
            0.2000000000000000111, 0.90483741803595951758));
        // Requires scaling:
        // scale = sum(obs) / sum(exp)
        // sum((obs - exp*scale)^2/(exp*scale))
        // pchisq(sum((obs - exp*scale)^2/(exp*scale)), length(obs) - 1, lower.tail=FALSE)
        builder.add(Arguments.of(
            new double[] {485, 541, 82, 61, 37},
            new long[] {500, 623, 72, 70, 31},
            9.0233079364273880429, 0.060519526474536095018));
        // Large test statistic
        builder.add(Arguments.of(
            new double[] {3389119.5, 649136.6, 285745.4, 25357364.76, 11291189.78, 543628.0,
                232921.0, 437665.75},
            new long[] {2372383, 584222, 257170, 17750155, 7903832, 489265, 209628, 393899},
            114875.90421929006698, 0));
        return builder.build();
    }

    @Test
    void testChiSquareTestTableThrows() {
        final long[][] counts = {{1, 2}, {3, 4}};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.chiSquareTest(counts, alpha), "chiSquareTest");

        assertChiSquareTestTableThrows(x -> testStatistic.chiSquare(x));
        assertChiSquareTestTableThrows(x -> testStatistic.chiSquareTest(x));
        assertChiSquareTestTableThrows(x -> testStatistic.chiSquareTest(x, 0.05));
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
    void testChiSquareTestTable(long[][] counts, double chiSquare, double p) {
        TestUtils.assertRelativelyEquals(chiSquare, testStatistic.chiSquare(counts), 1e-14, "statistic");
        TestUtils.assertProbability(p, testStatistic.chiSquareTest(counts), 1e-14, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.chiSquareTest(counts, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.chiSquareTest(counts, p * 0.99), "reject");
        }
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
    void testChiSquareTestDataSetsComparisonThrows() {
        final long[] observed1 = {1, 1};
        final long[] observed2 = {1, 1};
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.chiSquareTestDataSetsComparison(observed1, observed2, alpha), "chiSquareTestDataSetsComparison");

        testChiSquareTestDataSetsComparisonThrows((x, y) -> testStatistic.chiSquareDataSetsComparison(x, y));
        testChiSquareTestDataSetsComparisonThrows((x, y) -> testStatistic.chiSquareTestDataSetsComparison(x, y));
        testChiSquareTestDataSetsComparisonThrows((x, y) -> testStatistic.chiSquareTestDataSetsComparison(x, y, 0.05));
    }

    private static void testChiSquareTestDataSetsComparisonThrows(BiConsumer<long[], long[]> action) {
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
    void testChiSquareTestTwoSample(long[] observed1, long[] observed2, double chiSquare, double p) {
        TestUtils.assertRelativelyEquals(chiSquare, testStatistic.chiSquareDataSetsComparison(observed1, observed2), 1e-14, "statistic");
        TestUtils.assertProbability(p, testStatistic.chiSquareTestDataSetsComparison(observed1, observed2), 1e-14, "p-value");
        if (0.0001 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.chiSquareTestDataSetsComparison(observed1, observed2, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.chiSquareTestDataSetsComparison(observed1, observed2, p * 0.99), "reject");
        }
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
