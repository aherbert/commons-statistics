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

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Test cases for the {@link TTest}.
 */
class TTestTest {

    private final TTest testStatistic = new TTest();
    private final double[] tooShortObs = {1.0};
    private final double[] emptyObs = {};

    @Test
    void testOneSampleTThrows() {
        final double mu = 0;
        final double m = 0;
        final double v = 1;
        final long n = 5;
        final double[] sample = new double[5];
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.t(mu, (double[]) null), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(mu, emptyObs), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(mu, tooShortObs), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(mu, m, v, 0), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(mu, m, v, 1), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(mu, m, -1, n), "sample negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest(mu, (double[]) null), "null sample");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, emptyObs), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, tooShortObs), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, v, 0), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, v, 1), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, -1, n), "sample negative varience");
        final double alpha = 0.05;
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest(mu, (double[]) null, alpha), "null sample");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, emptyObs, alpha), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, tooShortObs, alpha), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, v, 0, alpha), "sample empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, v, 1, alpha), "sample too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(mu, m, -1, n, alpha), "sample negative varience");
        TestUtils.assertSignificanceLevel(a -> testStatistic.tTest(mu, sample, a), () -> "With double[]");
        TestUtils.assertSignificanceLevel(a -> testStatistic.tTest(mu, m, v, n, a), () -> "With (m, v, n)");
    }

    @Test
    void testOneSampleT() {
        final double[] observed =
            {93.0, 103.0, 95.0, 101.0, 91.0, 105.0, 96.0, 94.0, 101.0,  88.0, 98.0, 94.0, 101.0, 92.0, 95.0 };
        final double mu = 100.0;
        final SummaryStatistics sampleStats = new SummaryStatistics();
        for (int i = 0; i < observed.length; i++) {
            sampleStats.addValue(observed[i]);
        }
        final double m = sampleStats.getMean();
        final double v = sampleStats.getVariance();
        final long n = sampleStats.getN();

        // Target comparison values computed using R version 1.8.1 (Linux version)
        Assertions.assertEquals(-2.81976445346,
                testStatistic.t(mu, m, v, n), 10e-10, "t statistic");
        Assertions.assertEquals(-2.81976445346,
                testStatistic.t(mu, m, v, n), 10e-10, "t statistic");
        Assertions.assertEquals(0.0136390585873,
                testStatistic.tTest(mu, observed), 10e-10, "p value");
        Assertions.assertEquals(0.0136390585873,
                testStatistic.tTest(mu, m, v, n), 10e-10, "p value");
    }

    @Test
    void testOneSampleTTest() {
        final double[] oneSidedP = {2, 0, 6, 6, 3, 3, 2, 3, -6, 6, 6, 6, 3, 0, 1, 1, 0, 2, 3, 3};
        final SummaryStatistics oneSidedPStats = new SummaryStatistics();
        for (int i = 0; i < oneSidedP.length; i++) {
            oneSidedPStats.addValue(oneSidedP[i]);
        }
        final double m = oneSidedPStats.getMean();
        final double v = oneSidedPStats.getVariance();
        final long n = oneSidedPStats.getN();
        // Target comparison values computed using R version 1.8.1 (Linux version)
        Assertions.assertEquals(3.86485535541,
                testStatistic.t(0, oneSidedP), 10e-10, "one sample t stat");
        Assertions.assertEquals(3.86485535541,
                testStatistic.t(0, m, v, n), 1e-10, "one sample t stat");
        Assertions.assertEquals(0.000521637019637,
                testStatistic.tTest(0, oneSidedP) / 2, 10e-10, "one sample p value");
        Assertions.assertEquals(0.000521637019637,
                testStatistic.tTest(0, m, v, n) / 2, 10e-5, "one sample p value");
        Assertions.assertTrue(testStatistic.tTest(0, oneSidedP, 0.01), "one sample t-test reject");
        Assertions.assertTrue(testStatistic.tTest(0, m, v, n, 0.01), "one sample t-test reject");
        Assertions.assertFalse(testStatistic.tTest(0, oneSidedP, 0.0001), "one sample t-test accept");
        Assertions.assertFalse(testStatistic.tTest(0, m, v, n, 0.0001), "one sample t-test accept");
    }

    @Test
    void testTwoSampleTHeterscedasticThrows() {
        final double m = 0;
        final double v = 1;
        final long n = 5;
        final double[] sample = new double[5];
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.t((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, v, v, 0, n), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, v, v, 1, n), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, -1, v, n, n), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.t(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(sample, tooShortObs), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, v, v, n, 0), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, v, v, n, 1), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.t(m, m, v, -1, n, n), "sample2 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, 0, n), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, 1, n), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, -1, v, n, n), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(sample, tooShortObs), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, n, 0), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, n, 1), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, -1, n, n), "sample2 negative variance");
        final double alpha = 0.05;
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest((double[]) null, sample, alpha), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(emptyObs, sample, alpha), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(tooShortObs, sample, alpha), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, 0, n, alpha), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, 1, n, alpha), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, -1, v, n, n, alpha), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.tTest(sample, (double[]) null, alpha), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(sample, emptyObs, alpha), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(sample, tooShortObs, alpha), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, n, 0, alpha), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, v, n, 1, alpha), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.tTest(m, m, v, -1, n, n, alpha), "sample2 negative variance");
        TestUtils.assertSignificanceLevel(a -> testStatistic.tTest(sample, sample, a), () -> "With double[], double[]");
        TestUtils.assertSignificanceLevel(a -> testStatistic.tTest(m, m, v, v, n, n, a), () -> "With (m, m, v, v, n, n)");
    }

    @Test
    void testTwoSampleTHeterscedastic() {
        final double[] sample1 = {7, -4, 18, 17, -3, -5, 1, 10, 11, -2};
        final double[] sample2 = {-1, 12, -1, -3, 3, -5, 5, 2, -11, -1, -3};
        final SummaryStatistics sampleStats1 = new SummaryStatistics();
        for (int i = 0; i < sample1.length; i++) {
            sampleStats1.addValue(sample1[i]);
        }
        final SummaryStatistics sampleStats2 = new SummaryStatistics();
        for (int i = 0; i < sample2.length; i++) {
            sampleStats2.addValue(sample2[i]);
        }
        final double m1 = sampleStats1.getMean();
        final double v1 = sampleStats1.getVariance();
        final long n1 = sampleStats1.getN();
        final double m2 = sampleStats2.getMean();
        final double v2 = sampleStats2.getVariance();
        final long n2 = sampleStats2.getN();

        // Target comparison values computed using R version 1.8.1 (Linux version)
        Assertions.assertEquals(1.60371728768,
                testStatistic.t(sample1, sample2), 1e-10, "two sample heteroscedastic t stat");
        Assertions.assertEquals(1.60371728768,
                testStatistic.t(m1, m2, v1, v2, n1, n2), 1e-10, "two sample heteroscedastic t stat");
        Assertions.assertEquals(0.128839369622,
                testStatistic.tTest(sample1, sample2), 1e-10, "two sample heteroscedastic p value");
        Assertions.assertEquals(0.128839369622,
                testStatistic.tTest(m1, m2, v1, v2, n1, n2), 1e-10, "two sample heteroscedastic p value");
        Assertions.assertTrue(
                testStatistic.tTest(sample1, sample2, 0.2), "two sample heteroscedastic t-test reject");
        Assertions.assertTrue(
                testStatistic.tTest(m1, m2, v1, v2, n1, n2, 0.2), "two sample heteroscedastic t-test reject");
        Assertions.assertFalse(testStatistic.tTest(sample1, sample2, 0.1), "two sample heteroscedastic t-test accept");
        Assertions.assertFalse(testStatistic.tTest(1, m2, v1, v2, n1, n2, 0.1), "two sample heteroscedastic t-test accept");
    }

    @Test
    void testTwoSampleTHomoscedasticThrows() {
        final double m = 0;
        final double v = 1;
        final long n = 5;
        final double[] sample = new double[5];
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticT((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, v, v, 0, n), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, v, v, 1, n), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, -1, v, n, n), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticT(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(sample, tooShortObs), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, v, v, n, 0), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, v, v, n, 1), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticT(m, m, v, -1, n, n), "sample2 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticTTest((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, 0, n), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, 1, n), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, -1, v, n, n), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticTTest(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(sample, tooShortObs), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, n, 0), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, n, 1), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, -1, n, n), "sample2 negative variance");
        final double alpha = 0.05;
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticTTest((double[]) null, sample, alpha), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(emptyObs, sample, alpha), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(tooShortObs, sample, alpha), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, 0, n, alpha), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, 1, n, alpha), "sample1 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, -1, v, n, n, alpha), "sample1 negative variance");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.homoscedasticTTest(sample, (double[]) null, alpha), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(sample, emptyObs, alpha), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(sample, tooShortObs, alpha), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, n, 0, alpha), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, v, n, 1, alpha), "sample2 too small");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.homoscedasticTTest(m, m, v, -1, n, n, alpha), "sample2 negative variance");
        TestUtils.assertSignificanceLevel(a -> testStatistic.homoscedasticTTest(sample, sample, a), () -> "With double[], double[]");
        TestUtils.assertSignificanceLevel(a -> testStatistic.homoscedasticTTest(m, m, v, v, n, n, a), () -> "With (m, m, v, v, n, n)");
    }

    @Test
    void testTwoSampleTHomoscedastic() {
        final double[] sample1 = {2, 4, 6, 8, 10, 97};
        final double[] sample2 = {4, 6, 8, 10, 16};
        final SummaryStatistics sampleStats1 = new SummaryStatistics();
        for (int i = 0; i < sample1.length; i++) {
            sampleStats1.addValue(sample1[i]);
        }
        final SummaryStatistics sampleStats2 = new SummaryStatistics();
        for (int i = 0; i < sample2.length; i++) {
            sampleStats2.addValue(sample2[i]);
        }
        final double m1 = sampleStats1.getMean();
        final double v1 = sampleStats1.getVariance();
        final long n1 = sampleStats1.getN();
        final double m2 = sampleStats2.getMean();
        final double v2 = sampleStats2.getVariance();
        final long n2 = sampleStats2.getN();

        // Target comparison values computed using R version 1.8.1 (Linux version)
        Assertions.assertEquals(0.73096310086, testStatistic.homoscedasticT(sample1, sample2), 10e-11,
            "two sample homoscedastic t stat");
        Assertions.assertEquals(0.4833963785, testStatistic.homoscedasticTTest(m1, m2, v1, v2, n1, n2), 1e-10,
            "two sample homoscedastic p value");
        Assertions.assertTrue(testStatistic.homoscedasticTTest(sample1, sample2, 0.49),
            "two sample homoscedastic t-test reject");
        Assertions.assertFalse(testStatistic.homoscedasticTTest(sample1, sample2, 0.48),
            "two sample homoscedastic t-test accept");
        Assertions.assertTrue(testStatistic.homoscedasticTTest(m1, m2, v1, v2, n1, n2, 0.49),
            "two sample homoscedastic t-test reject");
        Assertions.assertFalse(testStatistic.homoscedasticTTest(m1, m2, v1, v2, n1, n2, 0.48),
            "two sample homoscedastic t-test accept");
    }

    @Test
    void testSmallSamples() {
        final double[] sample1 = {1, 3};
        final double[] sample2 = {4, 5};

        // Target values computed using R, version 1.8.1 (linux version)
        Assertions.assertEquals(-2.2360679775, testStatistic.t(sample1, sample2), 1e-10);
        Assertions.assertEquals(0.198727388935, testStatistic.tTest(sample1, sample2), 1e-10);
    }

    @Test
    void testPairedThrows() {
        final double[] sample = new double[5];
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedT((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedT(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedT(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedT(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedT(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedT(sample, tooShortObs), "sample2 too small");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedTTest((double[]) null, sample), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(emptyObs, sample), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(tooShortObs, sample), "sample1 too small");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedTTest(sample, (double[]) null), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(sample, emptyObs), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(sample, tooShortObs), "sample2 too small");
        final double alpha = 0.05;
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedTTest((double[]) null, sample, alpha), "null sample1");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(emptyObs, sample, alpha), "sample1 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(tooShortObs, sample, alpha), "sample1 too small");
        Assertions.assertThrows(NullPointerException.class, () -> testStatistic.pairedTTest(sample, (double[]) null, alpha), "null sample2");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(sample, emptyObs, alpha), "sample2 empty");
        Assertions.assertThrows(IllegalArgumentException.class, () -> testStatistic.pairedTTest(sample, tooShortObs, alpha), "sample2 too small");
        TestUtils.assertSignificanceLevel(a -> testStatistic.pairedTTest(sample, sample, a), () -> "With double[], double[]");
    }

    @Test
    void testPaired() {
        final double[] sample1 = {1, 3, 5, 7};
        final double[] sample2 = {0, 6, 11, 2};
        final double[] sample3 = {5, 7, 8, 10};

        // Target values computed using R, version 1.8.1 (linux version)
        Assertions.assertEquals(-0.3133, testStatistic.pairedT(sample1, sample2), 1e-4);
        Assertions.assertEquals(0.774544295819, testStatistic.pairedTTest(sample1, sample2), 1e-10);
        Assertions.assertEquals(0.001208, testStatistic.pairedTTest(sample1, sample3), 1e-6);
        Assertions.assertFalse(testStatistic.pairedTTest(sample1, sample3, .001));
        Assertions.assertTrue(testStatistic.pairedTTest(sample1, sample3, .002));
    }
}
