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
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Quantile.EstimationMethod;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link Quantile}.
 */
class QuantileTest {
    /** Estimation types to test. */
    private static final EstimationMethod[] TYPES = EstimationMethod.values();

    interface QuantileFunction {
        double evaluate(Quantile m, double[] values, double p);
    }

    interface QuantileFunction2 {
        double[] evaluate(Quantile m, double[] values, double[] p);
    }

    @Test
    void testNullPropertyThrows() {
        final Quantile m = Quantile.withDefaults();
        Assertions.assertThrows(NullPointerException.class, () -> m.with((NaNPolicy) null));
        Assertions.assertThrows(NullPointerException.class, () -> m.with((EstimationMethod) null));
        Assertions.assertThrows(NullPointerException.class, () -> m.withKthSelector(null));
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSPH(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateSPH, Quantile::evaluateSPH);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSPE(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            null, Quantile::evaluateSPE);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSP(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateSP, Quantile::evaluateSP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileWithCentralPivotStrategy(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults().withKthSelector(new KthSelector(PivotingStrategy.CENTRAL)),
            values, p, expected, delta,
            Quantile::evaluateSP, Quantile::evaluateSP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateBM, Quantile::evaluateBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateSBM, Quantile::evaluateSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileDP(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateDP, Quantile::evaluateDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileDP5(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateDP5, Quantile::evaluateDP5);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSBM2(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateSBM2, Quantile::evaluateSBM2);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileKSBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults().withPartition(new Partition().setKeyStrategy(KeyStrategy.PIVOT_CACHE)),
            values, p, expected, delta,
            Quantile::evaluateKSBM, Quantile::evaluateKSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileK1SBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults().withPartition(new Partition().setKeyStrategy(KeyStrategy.PIVOT_CACHE)),
            values, p, expected, delta,
            Quantile::evaluateK1SBM, Quantile::evaluateK1SBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantilePairedSBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluatePairedSBM, Quantile::evaluatePairedSBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileISBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateISBM, Quantile::evaluateISBM);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileIDP(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateIDP, Quantile::evaluateIDP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantile(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluate, Quantile::evaluate);
    }

    private static void assertQuantile(Quantile m, double[] values, double[] p,
        double[][] expected, double delta,
        QuantileFunction f1, QuantileFunction2 f2) {
        Assertions.assertEquals(expected.length, TYPES.length);
        for (int i = 0; i < TYPES.length; i++) {
            final EstimationMethod type = TYPES[i];
            m = m.with(type);
            // Single quantiles
            for (int j = 0; j < p.length; j++) {
                if (f1 != null) {
                    assertEqualsOrExactlyEqual(expected[i][j], f1.evaluate(m, values, p[j]), delta,
                        () -> type.toString());
                }
                assertEqualsOrExactlyEqual(expected[i][j], f2.evaluate(m, values, new double[] {p[j]})[0], delta,
                    () -> type.toString());
            }
            // Bulk quantiles
            if (delta < 0) {
                Assertions.assertArrayEquals(expected[i], f2.evaluate(m, values, p),
                    () -> type.toString());
            } else {
                Assertions.assertArrayEquals(expected[i], f2.evaluate(m, values, p), delta,
                    () -> type.toString());
            }
        }
    }

    /**
     * Assert that {@code expected} and {@code actual} are equal within the given{@code delta}.
     * If the {@code delta} is negative it is ignored and values must be exactly equal.
     *
     * @param expected Expected
     * @param actual Actual
     * @param delta Delta
     * @param messageSupplier Failure message.
     */
    private static void assertEqualsOrExactlyEqual(double expected, double actual, double delta,
        Supplier<String> messageSupplier) {
        if (delta < 0) {
            Assertions.assertEquals(expected, actual, messageSupplier);
        } else {
            Assertions.assertEquals(expected, actual, delta, messageSupplier);
        }
    }

    static Stream<Arguments> testQuantile() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Special cases
        final double nan = Double.NaN;
        addQuantiles(builder, new double[] {}, new double[] {0.75}, 1e-5,
            new double[] {nan, nan, nan, nan, nan, nan, nan, nan, nan});
        addQuantiles(builder, new double[] {42}, new double[] {0.75}, 1e-5,
            new double[] {42, 42, 42, 42, 42, 42, 42, 42, 42});
        // Cases from Commons Math PercentileTest
        addQuantiles(builder, new double[] {1, 2, 3}, new double[] {0.75}, 1e-5,
            new double[] {3, 3, 2, 2.25, 2.75, 3, 2.5, 2.83333, 2.81250});
        addQuantiles(builder, new double[] {0, 1}, new double[] {0.25}, 1e-5,
            new double[] {0, 0, 0, 0, 0, 0, 0.25, 0, 0});
        final double[] d = new double[] {1, 3, 2, 4};
        addQuantiles(builder, d, new double[] {0.3, 0.25, 0.75, 0.5}, 1e-5,
            new double[] {2, 2, 1, 1.2, 1.7, 1.5, 1.9, 1.63333, 1.65},
            new double[] {1, 1.5, 1, 1, 1.5, 1.25, 1.75, 1.41667, 1.43750},
            new double[] {3, 3.5, 3, 3, 3.5, 3.75, 3.25, 3.58333, 3.56250},
            new double[] {2, 2.5, 2, 2, 2.5, 2.5, 2.5, 2.5, 2.5});
        // NIST example
        addQuantiles(builder,
            new double[] {95.1772, 95.1567, 95.1937, 95.1959, 95.1442, 95.0610, 95.1591, 95.1195, 95.1772, 95.0925,
                95.1990, 95.1682},
            new double[] {0.9}, 1e-4,
            new double[] {95.19590, 95.19590, 95.19590, 95.19546, 95.19683, 95.19807, 95.19568, 95.19724, 95.19714});
        addQuantiles(builder,
            new double[] {12.5, 12.0, 11.8, 14.2, 14.9, 14.5, 21.0, 8.2, 10.3, 11.3, 14.1, 9.9, 12.2, 12.0, 12.1, 11.0,
                19.8, 11.0, 10.0, 8.8, 9.0, 12.3},
            new double[] {0.05}, 1e-4,
            new double[] {8.8000, 8.8000, 8.2000, 8.2600, 8.5600, 8.2900, 8.8100, 8.4700, 8.4925});
        // Special values tests
        addQuantiles(builder,
            new double[] {nan},
            new double[] {0.5}, 1e-4,
            new double[] {nan, nan, nan, nan, nan, nan, nan, nan, nan});
        addQuantiles(builder,
            new double[] {nan, nan},
            new double[] {0.5}, 1e-4,
            new double[] {nan, nan, nan, nan, nan, nan, nan, nan, nan});
        addQuantiles(builder,
            new double[] {1, nan},
            new double[] {0.5}, 1e-4,
            new double[] {1, nan, 1, 1, nan, nan, nan, nan, nan});
        addQuantiles(builder,
            new double[] {1, 2, nan},
            new double[] {0.5}, 1e-4,
            new double[] {2, 2, 2, 1.5, 2, 2, 2, 2, 2});
        addQuantiles(builder,
            new double[] {1, 2, nan, nan},
            new double[] {0.5}, 1e-4,
            new double[] {2, nan, 2, 2, nan, nan, nan, nan, nan});
        // Note: Any method using interpolation between negative zeros will return
        // positive zero because we use the scheme: x + (y - x) * alpha
        // (-0.0 - -0.0) == 0.0
        // Thus signed zeros do not need to be maintained for interpolation.
        // They are required for discrete schemes, or if the user wishes to
        // partition in place.
        // An optimisation for single quantiles would be to ignore signed zero
        // support when: all quantiles require interpolation; the data is not
        // modified in-place.
        addQuantiles(builder,
            new double[] {-0.0, -0.0, -0.0},
            new double[] {0.45}, -1,
            // Here HF1-3 are discrete; All continuous distributions interpolate to +0.0
            new double[] {-0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        addQuantiles(builder,
            new double[] {-0.0, -0.0},
            new double[] {0.0}, -1,
            // No interpolation
            new double[] {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0});
        return builder.build();
    }

    /**
     * Adds the quantiles.
     *
     * @param builder Builder.
     * @param x Data.
     * @param p Quantiles to compute.
     * @param expected Expected result for each p for every estimation type.
     */
    private static void addQuantiles(Stream.Builder<Arguments> builder,
        double[] x, double[] p, double delta, double[]... expected) {
        Assertions.assertEquals(p.length, expected.length);
        for (final double[] e : expected) {
            Assertions.assertEquals(e.length, TYPES.length);
        }
        // Transpose
        final double[][] t = new double[TYPES.length][p.length];
        for (int i = 0; i < t.length; i++) {
            for (int j = 0; j < p.length; j++) {
                t[i][j] = expected[j][i];
            }
        }
        builder.add(Arguments.of(x, p, t, delta));
    }

    @Test
    void testQuantileWithOverwrite() {
        final double[] values = {3, 4, 2, 1, 0};
        final double[] original = values.clone();
        Assertions.assertEquals(2, Quantile.withDefaults().withOverwrite(true).evaluateSP(values, 0.5));
        Assertions.assertFalse(Arrays.equals(original, values));
    }

    @Test
    void testQuantileWithOverwrite2() {
        final double[] values = {3, 4, 2, 1, 0};
        final double[] original = values.clone();
        Assertions.assertEquals(2, Quantile.withDefaults().withOverwrite(true).evaluateSP(values, new double[] {0.5})[0]);
        Assertions.assertFalse(Arrays.equals(original, values));
    }

    @Test
    void testBadQuantileThrows() {
        final double[] values = {3, 4, 2, 1, 0};
        final Quantile m = Quantile.withDefaults();
        for (final double p : new double[] {-0.5, 1.2, Double.NaN}) {
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSPH(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSPH(values, new double[] {p}));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSP(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSP(values, new double[] {p}));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateBM(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateBM(values, new double[] {p}));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSBM(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSBM(values, new double[] {p}));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateDP(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateDP(values, new double[] {p}));
        }
    }

    @Test
    void testNoQuantilesThrows() {
        final double[] values = {3, 4, 2, 1, 0};
        final Quantile m = Quantile.withDefaults();
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSPH(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSPH(values, new double[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSP(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSP(values, new double[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateBM(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateBM(values, new double[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSBM(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateSBM(values, new double[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateDP(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateDP(values, new double[0]));
    }
}
