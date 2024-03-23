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
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
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

    interface QuantileRangeFunction {
        double[] evaluate(Quantile m, double[] values, int c);
    }

    @Test
    void testNullPropertyThrows() {
        final Quantile m = Quantile.withDefaults();
        Assertions.assertThrows(NullPointerException.class, () -> m.with((NaNPolicy) null));
        Assertions.assertThrows(NullPointerException.class, () -> m.with((EstimationMethod) null));
        Assertions.assertThrows(NullPointerException.class, () -> m.withKthSelector(null));
    }

    @Test
    void testProbabilitiesThrows() {
        for (final int n : new int[] {-1, -42, Integer.MIN_VALUE}) {
            Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(n));
            Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(n, 0.5, 0.75));
        }
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, -0.5, 0.75));
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, 0.5, 1.75));
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, 0.75, 0.75));
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, 0.75, 0.5));
        final double nan = Double.NaN;
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, nan, 0.5));
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, 0.5, nan));
        Assertions.assertThrows(IllegalArgumentException.class, () -> Quantile.probabilities(1, nan, nan));
    }

    @ParameterizedTest
    @MethodSource(value = {"testProbabilities"})
    void testProbabilities(int n, double p1, double p2, double[] expected) {
        Assertions.assertArrayEquals(expected, Quantile.probabilities(n, p1, p2), 1e-10);
    }

    static Stream<Arguments> testProbabilities() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(1, 0.0, 1.0, new double[] {0.5}));
        builder.add(Arguments.of(2, 0.0, 1.0, new double[] {1.0 / 3, 2.0 / 3}));
        builder.add(Arguments.of(5, 0.0, 1.0, new double[] {1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6}));
        builder.add(Arguments.of(1, 0.25, 0.75, new double[] {0.5}));
        builder.add(Arguments.of(2, 0.25, 0.75, new double[] {0.25 + 1.0 / 6, 0.25 + 2.0 / 6}));
        builder.add(Arguments.of(1, 0.0, 0.5, new double[] {0.25}));
        builder.add(Arguments.of(2, 0.0, 0.5, new double[] {1.0 / 6, 2.0 / 6}));
        return builder.build();
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
    void testQuantileISP(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateISP, Quantile::evaluateISP);
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileIBM(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            Quantile::evaluateIBM, Quantile::evaluateIBM);
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

    @ParameterizedTest
    @MethodSource(value = {"testQuantile"})
    void testQuantileSorted(double[] values, double[] p, double[][] expected, double delta) {
        assertQuantile(Quantile.withDefaults(), values, p, expected, delta,
            (m, x, q) -> {
                // No clone here as later calls with the same array will also sort it
                Arrays.sort(x);
                return m.evaluate(x.length, i -> x[i], q);
            },
            (m, x, q) -> {
                // No clone here as later calls with the same array will also sort it
                Arrays.sort(x);
                return m.evaluate(x.length, i -> x[i], q);
            });
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
        // (0.0 - -0.0) == 0.0
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
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(values, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(values, new double[] {p}));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(10, i -> 1, p));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(10, i -> 1, new double[] {p}));
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
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(values));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(values, new double[0]));
        Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(10, i -> 1, new double[0]));
    }

    @Test
    void testInvalidSizeThrows() {
        final Quantile m = Quantile.withDefaults();
        for (final int n : new int[] {-1, -42, Integer.MIN_VALUE}) {
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(n, i -> 1, 0.5));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluate(n, i -> 1, 0.5, 0.75));
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateRange(n, i -> 1, 100));
        }
    }

    @Test
    void testInvalidNumberOfQuantilesThrows() {
        final Quantile m = Quantile.withDefaults();
        for (final int c : new int[] {-1, 0, -42, Integer.MIN_VALUE}) {
            Assertions.assertThrows(IllegalArgumentException.class, () -> m.evaluateRange(10, i -> 1, c));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantileRange"})
    void testQuantileRange(double[] values, int n) {
        assertQuantileRange(Quantile.withDefaults(), values, n,
            (m, x, c) -> m.evaluateRange(x, c));
    }

    @ParameterizedTest
    @MethodSource(value = {"testQuantileRange"})
    void testQuantileRangeSorted(double[] values, int n) {
        assertQuantileRange(Quantile.withDefaults(), values, n,
            (m, x, c) -> {
                // No clone here as later calls with the same array will also sort it
                Arrays.sort(x);
                return m.evaluateRange(x.length, i -> x[i], c);
            });
    }

    private static void assertQuantileRange(Quantile m, double[] values, int c, QuantileRangeFunction f) {
        // Use Quantile components to compute the expected value from a sorted copy
        final double[] copy = values.clone();
        Arrays.sort(copy);
        // Uniform quantiles
        final double[] p = IntStream.rangeClosed(1, c).mapToDouble(i -> i / (c + 1.0)).toArray();
        final double[] expected = new double[c];

        // Evaluate
        for (final EstimationMethod type : TYPES) {
            m = m.with(type);

            // Create expected result
            for (int k = 0; k < c; k++) {
                final double pos = type.index(p[k], copy.length);
                final int i = (int) pos;
                if (pos > i) {
                    expected[k] = DoubleMath.interpolate(copy[i], copy[i + 1], pos - i);
                } else {
                    expected[k] = copy[i];
                }
            }

            Assertions.assertArrayEquals(expected, f.evaluate(m, values, c),
                () -> type.toString());
        }
    }

    static Stream<Arguments> testQuantileRange() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final double nan = Double.NaN;
        builder.add(Arguments.of(new double[] {nan}, 1));
        builder.add(Arguments.of(new double[] {nan, nan}, 10));
        builder.add(Arguments.of(new double[] {1, nan, nan}, 10));
        builder.add(Arguments.of(new double[] {1, 2, nan, nan}, 10));
        builder.add(Arguments.of(new double[] {1, 3, 2, 4}, 2));
        builder.add(Arguments.of(new double[] {1, 3, 2, 4}, 5));
        // NIST data
        final double[] x = {95.1772, 95.1567, 95.1937, 95.1959, 95.1442, 95.0610, 95.1591, 95.1195, 95.1772, 95.0925,
            95.1990, 95.1682};
        builder.add(Arguments.of(x, 5));
        builder.add(Arguments.of(x, 10));
        // Some data
        final double[] y = {12.5, 12.0, 11.8, 14.2, 14.9, 14.5, 21.0, 8.2, 10.3, 11.3, 14.1, 9.9, 12.2, 12.0, 12.1,
            11.0, 19.8, 11.0, 10.0, 8.8, 9.0, 12.3};
        builder.add(Arguments.of(y, 5));
        builder.add(Arguments.of(y, 10));
        // Random data of different lengths
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        for (int p = 5; p <= 10; p++) {
            final int size = 1 << p;
            builder.add(Arguments.of(rng.doubles(size).toArray(), p - 3));
            builder.add(Arguments.of(rng.doubles(size + 1).toArray(), p - 3));
        }
        return builder.build();
    }
}
