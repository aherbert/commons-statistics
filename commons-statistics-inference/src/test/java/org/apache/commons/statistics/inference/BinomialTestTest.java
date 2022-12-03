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
import java.util.stream.IntStream;
import org.apache.commons.statistics.distribution.BinomialDistribution;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Test cases for {@link BinomialTest}.
 */
class BinomialTestTest {

    private final BinomialTest testStatistic = new BinomialTest();

    @ParameterizedTest
    @CsvSource({
        "10, 5, -1",
        "10, 5, 2",
        "10, -1, 0.5",
        "10, 11, 0.5",
        "-1, 5, 0.5",
        "1, 2, 0.5",
    })
    void testBinomialTestThrows(int n, int k, double p) {
        Assertions.assertThrows(IllegalArgumentException.class, () ->
            testStatistic.binomialTest(n, k, p, AlternativeHypothesis.TWO_SIDED));
        final double alpha = 0.05;
        Assertions.assertThrows(IllegalArgumentException.class, () ->
            testStatistic.binomialTest(n, k, p, AlternativeHypothesis.TWO_SIDED, alpha));
    }

    @Test
    void testBinomialTestThrowsNullAlternativeHypothesis() {
        Assertions.assertThrows(NullPointerException.class, () ->
            testStatistic.binomialTest(10, 5, 0.5, null));
        Assertions.assertThrows(NullPointerException.class, () ->
            testStatistic.binomialTest(10, 5, 0.5, null, 0.05));
    }

    @Test
    void testBinomialTestThrowsInvalidAlpha() {
        TestUtils.assertSignificanceLevel(
            alpha -> testStatistic.binomialTest(10, 5, 0.5, AlternativeHypothesis.TWO_SIDED, alpha),
            () -> "two sided");
    }

    /**
     * Test the binomial test for each alternative hypothesis and all possible k given
     * the number of trials (n).
     *
     * <p>Compute the p-values using a direct summation of the probability values.
     * See https://en.wikipedia.org/wiki/Binomial_test.
     *
     * <p>The summation is not as accurate as using the CDF / SF so the epsilon
     * is changed for larger number of trials (n). When n is very large summing the
     * individual p-values has too much error and is covered by
     * {@link #testBinomialTestLargeN(int, double)}.
     */
    @ParameterizedTest
    @CsvSource({
        "0, 0.25, 1e-15, 0",
        "1, 0.25, 1e-15, 0",
        "2, 0.25, 1e-15, 0",
        "0, 0.5, 1e-15, 0",
        "1, 0.5, 1e-15, 0",
        "2, 0.5, 1e-15, 0",
        "0, 0.75, 1e-15, 0",
        "1, 0.75, 1e-15, 0",
        "2, 0.75, 1e-15, 0",
        "10, 0.25, 2e-15, 0",
        "10, 0.49, 2e-15, 0",
        "10, 0.5, 2e-15, 0",
        "10, 0.51, 2e-15, 0",
        "10, 0.75, 2e-15, 0",
        "11, 0.25, 3e-15, 0",
        "11, 0.49, 2e-15, 0",
        "11, 0.5, 2e-15, 0",
        "11, 0.51, 2e-15, 0",
        "11, 0.75, 3e-15, 0",
        "5, 0.1, 2e-15, 0",
        "5, 0.7, 1e-15, 0",
        "20, 0.7, 3e-15, 0",
    })
    void testBinomialTest(int n, double p, double eps) {
        final BinomialDistribution dist = BinomialDistribution.of(n, p);
        final double[] pk = IntStream.rangeClosed(0, n).mapToDouble(dist::probability).toArray();

        // Note: TestUtils.assertProbability expects exact equality when p is 0 or 1.
        // Set the maximum for the sum to below 1 to avoid this.
        final double maxP = Math.nextDown(1.0);

        IntStream.rangeClosed(0, n).forEach(k -> {
            double expected;

            // One-sided
            expected = Math.min(maxP, IntStream.rangeClosed(0, k).mapToDouble(i -> pk[i]).sum());
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.LESS_THAN), eps,
                () -> "less than: k=" + k);

            expected = Math.min(maxP, IntStream.rangeClosed(k, n).mapToDouble(i -> pk[i]).sum());
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.GREATER_THAN), eps,
                () -> "greater than: k=" + k);

            // Two-sided
            // Find all i where Pr(X = i) <= Pr(X = k) and sum them.
            expected = Math.min(maxP, Arrays.stream(pk).filter(x -> x <= pk[k]).sum());
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.TWO_SIDED), eps,
                () -> "two-sided: k=" + k);
        });
    }

    /**
     * Test the binomial test for each alternative hypothesis and all possible k given
     * the number of trials (n).
     *
     * <p>Compute the p-values using a summation of the probability values.
     * See https://en.wikipedia.org/wiki/Binomial_test.
     *
     * <p>The summation is performed using the CDF / SF after look-up of the appropriate
     * value for k. The actual value must be an exact match to the expected. This test
     * verifies the binary search to locate indices in BinomialTest is correct.
     */
    @ParameterizedTest
    @CsvSource({
        "1234, 0.3",
        "1234, 0.55",
        "1234, 0.87",
        "12345, 0.3",
        "12345, 0.55",
        "12345, 0.87",
        // Case where the upper and lower mode have different values
        "10000, 0.49999",
        "10000, 0.50001",
    })
    void testBinomialTestLargeN(int n, double p) {
        final BinomialDistribution dist = BinomialDistribution.of(n, p);
        // Use the log probability here which has a larger range than probability and
        // can detect small p-values that are non-zero but less than Double.MIN_NORMAL.
        // It is also faster as it avoids a call to Math.exp.
        final double[] pk = IntStream.rangeClosed(0, n).mapToDouble(dist::logProbability).toArray();

        // Require an exact match.
        // This ensures the BinomialTest uses the CDF and SF.
        final double eps = 0;

        IntStream.rangeClosed(0, n).forEach(k -> {
            double expected;

            // One-sided.
            expected = dist.cumulativeProbability(k);
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.LESS_THAN), eps,
                () -> "less than: k=" + k);

            expected = dist.survivalProbability(k - 1);
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.GREATER_THAN), eps,
                () -> "greater than: k=" + k);

            // Two-sided
            // Find all i where Pr(X = i) <= Pr(X = k) and sum them.
            int i = -1;
            while (i < n && pk[i + 1] <= pk[k]) {
                i++;
            }
            int j = n + 1;
            while (j > 0 && pk[j - 1] <= pk[k]) {
                j--;
            }
            expected = j <= i ? 1 : dist.cumulativeProbability(i) + dist.survivalProbability(j - 1);
            TestUtils.assertProbability(expected,
                testStatistic.binomialTest(n, k, p, AlternativeHypothesis.TWO_SIDED), eps,
                () -> "two-sided: k=" + k);
        });
    }

    @Test
    void testBinomialTestPValues() {
        final int successes = 51;
        final int trials = 235;
        final double probability = 1.0 / 6.0;

        Assertions.assertEquals(0.04375, testStatistic.binomialTest(
            trials, successes, probability, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.02654, testStatistic.binomialTest(
            trials, successes, probability, AlternativeHypothesis.GREATER_THAN), 1e-4);
        Assertions.assertEquals(0.982, testStatistic.binomialTest(
            trials, successes, probability, AlternativeHypothesis.LESS_THAN), 1e-4);

        // for special boundary conditions
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 3, 1, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 3, 0.9, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 3, 0.8, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.559, testStatistic.binomialTest(
            3, 3, 0.7, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.28, testStatistic.binomialTest(
            3, 3, 0.6, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.25, testStatistic.binomialTest(
            3, 3, 0.5, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.064, testStatistic.binomialTest(
            3, 3, 0.4, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.027, testStatistic.binomialTest(
            3, 3, 0.3, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.008, testStatistic.binomialTest(
            3, 3, 0.2, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.001, testStatistic.binomialTest(
            3, 3, 0.1, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0, testStatistic.binomialTest(
            3, 3, 0.0, AlternativeHypothesis.TWO_SIDED), 1e-4);

        Assertions.assertEquals(0, testStatistic.binomialTest(
            3, 0, 1, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.001, testStatistic.binomialTest(
            3, 0, 0.9, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.008, testStatistic.binomialTest(
            3, 0, 0.8, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.027, testStatistic.binomialTest(
            3, 0, 0.7, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.064, testStatistic.binomialTest(
            3, 0, 0.6, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.25, testStatistic.binomialTest(
            3, 0, 0.5, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.28, testStatistic.binomialTest(
            3, 0, 0.4, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(0.559, testStatistic.binomialTest(
            3, 0, 0.3, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 0, 0.2, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 0, 0.1, AlternativeHypothesis.TWO_SIDED), 1e-4);
        Assertions.assertEquals(1, testStatistic.binomialTest(
            3, 0, 0.0, AlternativeHypothesis.TWO_SIDED), 1e-4);
    }

    @ParameterizedTest
    @CsvSource({
        "235, 51, 0.166666666666666, TWO_SIDED",
        "235, 51, 0.166666666666666, GREATER_THAN",
        "235, 29, 0.166666666666666, LESS_THAN",
    })
    void testBinomialTestReject(int n, int k, double p, AlternativeHypothesis h) {
        final double alpha = testStatistic.binomialTest(n, k, p, h);
        Assertions.assertFalse(testStatistic.binomialTest(n, k, p, h, alpha), "Should not reject at p == alpha");
        Assertions.assertFalse(testStatistic.binomialTest(n, k, p, h, Math.nextDown(alpha)),  "Should not reject at p > alpha");
        Assertions.assertTrue(testStatistic.binomialTest(n, k, p, h, Math.nextUp(alpha)),  "Should reject at p < alpha");
    }

    @ParameterizedTest
    @CsvSource({
        // numberOfSuccesses = numberOfTrials * probability (median)
        "10, 5, 0.5",
        "11, 5, 0.5",
        "11, 6, 0.5",
        "20, 5, 0.25",
        "21, 5, 0.25",
        "21, 6, 0.25",
        "20, 15, 0.75",
        "21, 15, 0.75",
        "21, 16, 0.75",
    })
    void testMath1644(int n, int k, double p) {
        final BinomialTest bt = new BinomialTest();
        final double pval = bt.binomialTest(n, k, p, AlternativeHypothesis.TWO_SIDED);
        Assertions.assertTrue(pval <= 1, () -> "pval=" + pval);
    }
}
