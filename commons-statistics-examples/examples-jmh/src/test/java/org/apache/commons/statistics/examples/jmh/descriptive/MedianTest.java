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

package org.apache.commons.statistics.examples.jmh.descriptive;

import java.util.Arrays;
import java.util.stream.Stream;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Median}.
 */
class MedianTest {
    @Test
    void testNullPropertyThrows() {
        final Median m = Median.withDefaults();
        Assertions.assertThrows(NullPointerException.class, () -> m.with((NaNPolicy) null));
        Assertions.assertThrows(NullPointerException.class, () -> m.withKthSelector(null));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianSP(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateSP(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianSPN(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateSPN(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianSBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateSBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianDP(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateDP(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianDP5(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateDP5(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianSBM2(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateSBM2(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianKSBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateKSBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianK1SBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults()
            .withPartition(new Partition().setKeyStrategy(KeyStrategy.PIVOT_CACHE))
            .evaluateK1SBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianPairedSBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluatePairedSBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianISBM(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateISBM(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianIDP(double[] values, double expected) {
        Assertions.assertEquals(expected, Median.withDefaults().evaluateIDP(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedianWithCentralPivotStrategy(double[] values, double expected) {
        final Median m = Median.withDefaults().withKthSelector(new KthSelector(PivotingStrategy.CENTRAL));
        Assertions.assertEquals(expected, m.evaluateSP(values));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedian"})
    void testMedian(double[] values, double expected) {
        final Median m = Median.withDefaults();
        Assertions.assertEquals(expected, m.evaluate(values));
    }

    static Stream<Arguments> testMedian() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final Percentile p = new Percentile(50).withNaNStrategy(NaNStrategy.FIXED);
        // Note: Cannot use CM when NaN is adjacent to the middle of an odd length
        // as it always interpolates pairs and uses: low + 0.0 * (NaN - low)
        for (final double[] x : new double[][] {
            {1},
            {1, 2},
            {2, 1},
            {1, Double.NaN},
            {Double.NaN, Double.NaN},
            {1, Double.NaN, Double.NaN},
            {1, 2, Double.NaN, Double.NaN},
            {Double.NaN, Double.NaN, 1, 2, 3, 4},
            {Double.MAX_VALUE, Double.MAX_VALUE},
            {-Double.MAX_VALUE, -Double.MAX_VALUE / 2},
        }) {
            builder.add(Arguments.of(x, p.evaluate(x)));
        }
        // Cases where CM Percentile returns NaN
        builder.add(Arguments.of(new double[]{1, 2, Double.NaN}, 2));
        builder.add(Arguments.of(new double[]{Double.NaN, 1, 2, 3, Double.NaN}, 3));

        // Test against the percentile can fail at 1 ULP so used a fixed seed
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(26378461823L);
        // Sizes above and below the threshold for partitioning
        double[] x;
        for (final int size : new int[] {5, 6, 50, 51}) {
            final double[] values = rng.doubles(size, -4.5, 1.5).toArray();
            final double expected = p.evaluate(values);
            for (int i = 0; i < 20; i++) {
                x = TestUtils.shuffle(rng, values.clone());
                builder.add(Arguments.of(x, expected));
            }
            for (final double y : new double[] {-0.0, 0.0, 1, Double.MAX_VALUE,
                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN}) {
                x = new double[size];
                Arrays.fill(x, y);
                builder.add(Arguments.of(x, y));
            }
            // Odd: just over half -0.0
            // Even: half -0.0
            x = new double[size];
            Arrays.fill(x, 0, (size + 1) / 2, -0.0);
            TestUtils.shuffle(rng, x);
            builder.add(Arguments.of(x.clone(), (size & 0x1) == 1 ? -0.0 : 0.0));
            Arrays.fill(x, 123.45);
            builder.add(Arguments.of(x, x[0]));
        }
        // Special cases
        builder.add(Arguments.of(new double[] {}, Double.NaN));
        builder.add(Arguments.of(new double[] {-Double.MAX_VALUE, Double.MAX_VALUE}, 0));
        builder.add(Arguments.of(new double[] {-0.0, -0.0}, -0.0));
        builder.add(Arguments.of(new double[] {-0.0, 0.0, -0.0}, -0.0));
        return builder.build();
    }

    @Test
    void testMedianWithOverwrite() {
        final double[] values = {3, 4, 2, 1, 0};
        final double[] original = values.clone();
        Assertions.assertEquals(2, Median.withDefaults().withOverwrite(true).evaluateSP(values));
        Assertions.assertFalse(Arrays.equals(original, values));
    }

    @Test
    void test() {
        // TODO - remove this. It is here to check the CM implementation
        final org.apache.commons.math3.stat.descriptive.rank.Median m =
            new org.apache.commons.math3.stat.descriptive.rank.Median();
        double[] x = new double[50];
        Arrays.fill(x, 123.45);
        Assertions.assertEquals(x[0], m.evaluate(x));
    }
}
