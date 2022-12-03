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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.statistics.inference.OneWayAnova.AnovaSummaryStatistics;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test cases for {@link OneWayAnova}.
 */
class OneWayAnovaTest {

    private final OneWayAnova testStatistic = new OneWayAnova();

    @Test
    void testAnovaSummaryStatisticsThrows() {
        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> AnovaSummaryStatistics.of(null));
    }

    @ParameterizedTest
    @MethodSource
    void testAnovaSummaryStatistics(double[] values, double sum, double sumSq) {
        final AnovaSummaryStatistics s = AnovaSummaryStatistics.of(values);
        Assertions.assertEquals(values.length, s.getN(), "n");
        Assertions.assertEquals(sum, s.getSum(), "sum");
        Assertions.assertEquals(sumSq, s.getSumOfSquares(), "sumOfSquares");
    }

    static Stream<Arguments> testAnovaSummaryStatistics() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(new double[] {}, 0, 0));
        builder.add(Arguments.of(new double[] {4}, 4, 16));
        builder.add(Arguments.of(new double[] {1, 2, 3}, 6, 14));
        return builder.build();
    }

    @Test
    void testAnovaTestThrows() {
        final List<double[]> data = Arrays.asList(
                new double[] {1, 2, 3}, new double[] {4, 5, 6});
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.anovaTest(data, alpha), "anovaTest");

        assertAnovaTestThrows(testStatistic::anovaFValue);
        assertAnovaTestThrows(testStatistic::anovaPValue);
        assertAnovaTestThrows(x -> testStatistic.anovaTest(x, 0.05));
    }

    private static void assertAnovaTestThrows(Consumer<Collection<double[]>> action) {
        final List<double[]> emptyContents = new ArrayList<>();
        emptyContents.add(new double[] {1, 2, 3});
        emptyContents.add(new double[] {});
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(emptyContents), "values", "size");

        final List<double[]> tooFew = new ArrayList<>();
        tooFew.add(new double[] {1, 2, 3});
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(tooFew), "categories");

        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null));
    }


    @Test
    void testAnovaTest1Throws() {
        final List<AnovaSummaryStatistics> data = Arrays.asList(
                AnovaSummaryStatistics.of(new double[] {1, 2, 3}),
                AnovaSummaryStatistics.of(new double[] {4, 5, 6}));
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.anovaTest1(data, alpha), "anovaTest");

        assertAnovaTest1Throws(testStatistic::anovaFValue1);
        assertAnovaTest1Throws(testStatistic::anovaPValue1);
        assertAnovaTest1Throws(x -> testStatistic.anovaTest1(x, 0.05));
    }

    private static void assertAnovaTest1Throws(Consumer<Collection<AnovaSummaryStatistics>> action) {
        final List<AnovaSummaryStatistics> emptyContents = new ArrayList<>();
        emptyContents.add(AnovaSummaryStatistics.of(new double[] {1, 2, 3}));
        emptyContents.add(AnovaSummaryStatistics.of());
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> action.accept(emptyContents), "values", "size");

        final List<AnovaSummaryStatistics> tooFew = new ArrayList<>();
        tooFew.add(AnovaSummaryStatistics.of(new double[] {1, 2, 3}));
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> action.accept(tooFew), "categories");

        TestUtils.assertThrowsWithMessage(NullPointerException.class,
            () -> action.accept(null));
    }

    @ParameterizedTest
    @MethodSource
    void testAnova(Collection<double[]> data, double f, double p, double statEps, double pEps) {
        final double statistic = testStatistic.anovaFValue(data);
        final double pValue = testStatistic.anovaPValue(data);
        TestUtils.assertRelativelyEquals(f, statistic, statEps, "statistic");
        TestUtils.assertProbability(p, pValue, pEps, "p-value");
        if (1e-6 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.anovaTest(data, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.anovaTest(data, p * 0.99), "reject");
        }

        final Collection<AnovaSummaryStatistics> summaryData =
            data.stream().map(AnovaSummaryStatistics::of).collect(Collectors.toList());
        Assertions.assertEquals(statistic, testStatistic.anovaFValue1(summaryData));
        Assertions.assertEquals(pValue, testStatistic.anovaPValue1(summaryData));
        if (1e-6 < p && p < 0.45) {
            Assertions.assertTrue(testStatistic.anovaTest1(summaryData, p * 1.01), "reject");
            Assertions.assertFalse(testStatistic.anovaTest1(summaryData, p * 0.99), "reject");
        }
    }

    static Stream<Arguments> testAnova() {
        final double[] classA = {93.0, 103.0, 95.0, 101.0, 91.0, 105.0, 96.0, 94.0, 101.0};
        final double[] classB = {99.0, 92.0, 102.0, 100.0, 102.0, 89.0};
        final double[] classC = {110.0, 115.0, 111.0, 117.0, 128.0, 117.0};
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Target values computed using R version 3.4.0
        // A = c(93.0, 103.0, 95.0, 101.0, 91.0, 105.0, 96.0, 94.0, 101.0)
        // B = c(99.0, 92.0, 102.0, 100.0, 102.0, 89.0)
        // C = c(110.0, 115.0, 111.0, 117.0, 128.0, 117.0)
        // cA = rep("a", length(A))
        // cB = rep("b", length(B))
        // cC = rep("c", length(C))
        // abc <- data.frame(cat=c(cA, cB, cC), value=c(A, B, C))
        // res.aov <- aov(value ~ cat, data = abc)
        // d = summary(res.aov)
        // d[[1]][["F value"]]
        // d[[1]][["Pr(>F)"]]
        builder.add(Arguments.of(
            Arrays.asList(classA, classB, classC),
            24.67361709460625363, 6.9594458853833384454e-06, 1e-14, 1e-13));
        // ab <- data.frame(cat=c(cA, cB), value=c(A, B))
        // d <- summary(aov(value ~ cat, data = ab))
        builder.add(Arguments.of(
            Arrays.asList(classA, classB),
            0.01505791505791488627, 0.90421296046434118665, 5e-12, 3e-13));
        // ac <- data.frame(cat=c(cA, cC), value=c(A, C))
        // d <- summary(aov(value ~ cat, data = ac))
        builder.add(Arguments.of(
            Arrays.asList(classA, classC),
            40.632558139534836528, 2.439935448222361878e-05, 3e-14, 2e-13));
        // bc <- data.frame(cat=c(cB, cC), value=c(B, C))
        // d <- summary(aov(value ~ cat, data = bc))
        builder.add(Arguments.of(
            Arrays.asList(classB, classC),
            30.195167286245290228, 0.00026362644003193318621, 5e-15, 1e-14));
        return builder.build();
    }
}
