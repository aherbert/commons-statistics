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

package org.apache.commons.statistics.distribution;

/**
 * Implementation of the trapezoidal distribution.
 *
 * <p>The probability density function of \( X \) is:
 *
 * <p>\[ f(x; a, b, c, d) = \begin{cases}
 *       \frac{2}{d+c-a-b}\frac{x-a}{b-a} &amp; \text{for } a\le x \lt b \\
 *       \frac{2}{d+c-a-b}                &amp; \text{for } b\le x \lt c \\
 *       \frac{2}{d+c-a-b}\frac{d-x}{d-c} &amp; \text{for } c\le x \le d
 *       \end{cases} \]
 *
 * <p>for \( -\infty \lt a \le b \le c \le d \lt \infty \) and
 * \( x \in [a, d] \).
 *
 * @see <a href="https://en.wikipedia.org/wiki/Trapezoidal_distribution">Trapezoidal distribution (Wikipedia)</a>
 */
public final class TrapezoidalDistribution extends AbstractContinuousDistribution {
    /** Lower limit of this distribution (inclusive). */
    private final double a;
    /** Start of the trapezoid constant density. */
    private final double b;
    /** End of the trapezoid constant density. */
    private final double c;
    /** Upper limit of this distribution (inclusive). */
    private final double d;
    /** Cached value (d + c - a - b). */
    private final double divisor;
    /** Cached value (b - a). */
    private final double bma;
    /** Cached value (d - c). */
    private final double dmc;
    /** Cumulative probability at b. */
    private final double cdfB;
    /** Cumulative probability at c. */
    private final double cdfC;
    /** Survival probability at b. */
    private final double sfB;
    /** Survival probability at c. */
    private final double sfC;

    /**
     * @param a Lower limit of this distribution (inclusive).
     * @param b Start of the trapezoid constant density.
     * @param c End of the trapezoid constant density.
     * @param d Upper limit of this distribution (inclusive).
     */
    private TrapezoidalDistribution(double a,
                                    double b,
                                    double c,
                                    double d) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;

        // Sum positive terms
        divisor = (d - a) + (c - b);
        bma = b - a;
        dmc = d - c;

        // Handle the special case for the triangular distribution (b == c)
        // where all computations for this range (and inversion) are skipped.

        // Compute probabilities at point C
        final double sf = dmc / divisor;
        final double cdf = 1 - sf;

        // Floating-point equality comparison is intentional.
        if (b == c) {
            // Triangular distribution.
            // Probability computation is skipped between b <= x < c.
            // Setting p(b) == p(c) ensures inversion uses the correct function.
            cdfB = cdfC = cdf;
            sfB = sfC = sf;
        } else {
            cdfB = bma / divisor;
            sfB = 1 - cdfB;
            cdfC = cdf;
            sfC = sf;
        }
    }

    /**
     * Creates a trapezoidal distribution.
     *
     * <p>The distribution density is represented as an up sloping line from
     * {@code a} to {@code b}, constant from {@code b} to {@code c}, and then a down
     * sloping line from {@code c} to {@code d}.
     *
     * @param a Lower limit of this distribution (inclusive).
     * @param b Start of the trapezoid constant density (shape parameter).
     * @param c End of the trapezoid constant density (shape parameter).
     * @param d Upper limit of this distribution (inclusive).
     * @return the distribution
     * @throws IllegalArgumentException if {@code a >= d}, if {@code b < a}, if
     * {@code c < b} or if {@code c > d}.
     */
    public static TrapezoidalDistribution of(double a,
                                             double b,
                                             double c,
                                             double d) {
        if (a >= d) {
            throw new DistributionException(DistributionException.INVALID_RANGE_LOW_GTE_HIGH,
                                            a, d);
        }
        if (b < a) {
            throw new DistributionException(DistributionException.TOO_SMALL,
                                            b, a);
        }
        if (c < b) {
            throw new DistributionException(DistributionException.TOO_SMALL,
                                            c, b);
        }
        if (c > d) {
            throw new DistributionException(DistributionException.TOO_LARGE,
                                            c, d);
        }
        return new TrapezoidalDistribution(a, b, c, d);
    }

    /**
     * Gets the start of the constant region of the density function.
     *
     * <p>This is the first shape parameter {@code b} of the distribution.
     *
     * @return the first shape parameter {@code b}
     */
    public double getB() {
        return b;
    }

    /**
     * Gets the end of the constant region of the density function.
     *
     * <p>This is the second shape parameter {@code c} of the distribution.
     *
     * @return the second shape parameter {@code c}
     */
    public double getC() {
        return c;
    }

    /** {@inheritDoc} */
    @Override
    public double density(double x) {
        if (x <= a) {
            return 0;
        }
        if (x < b) {
            final double divident = (x - a) / bma;
            return 2 * (divident / divisor);
        }
        if (x < c) {
            return 2 / divisor;
        }
        if (x < d) {
            final double divident = (d - x) / dmc;
            return 2 * (divident / divisor);
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double cumulativeProbability(double x)  {
        if (x <= a) {
            return 0;
        }
        if (x < b) {
            final double divident = (x - a) * (x - a) / bma;
            return divident / divisor;
        }
        if (x < c) {
            final double divident = 2 * x - b - a;
            return divident / divisor;
        }
        if (x < d) {
            final double divident = (d - x) * (d - x) / dmc;
            return 1 - divident / divisor;
        }
        return 1;
    }


    /** {@inheritDoc} */
    @Override
    public double survivalProbability(double x)  {
        // By symmetry:
        if (x <= a) {
            return 1;
        }
        if (x < b) {
            final double divident = (x - a) * (x - a) / bma;
            return 1 - divident / divisor;
        }
        if (x < c) {
            final double divident = 2 * x - b - a;
            return 1 - divident / divisor;
        }
        if (x < d) {
            final double divident = (d - x) * (d - x) / dmc;
            return divident / divisor;
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double inverseCumulativeProbability(double p) {
        ArgumentUtils.checkProbability(p);
        if (p == 0) {
            return a;
        }
        if (p == 1) {
            return d;
        }
        if (p < cdfB) {
            return a + Math.sqrt(p * divisor * bma);
        }
        if (p < cdfC) {
            return 0.5 * (a + b + (p * divisor));
        }
        return d - Math.sqrt((1 - p) * divisor * dmc);
    }

    /** {@inheritDoc} */
    @Override
    public double inverseSurvivalProbability(double p) {
        // By symmetry:
        ArgumentUtils.checkProbability(p);
        if (p == 1) {
            return a;
        }
        if (p == 0) {
            return d;
        }
        if (p >= sfB) {
            return a + Math.sqrt((1 - p) * divisor * bma);
        }
        if (p >= sfC) {
            return 0.5 * (a + b + ((1 - p) * divisor));
        }
        return d - Math.sqrt(p * divisor * dmc);
    }

    /**
     * {@inheritDoc}
     *
     * <p>For lower limit {@code a}, start of the density constant region {@code b},
     * end of the density constant region {@code c} and upper limit {@code d}, the
     * mean is:
     *
     * <p>\[ \frac{1}{3(d+c-b-a)}\left(\frac{d^3-c^3}{d-c}-\frac{b^3-a^3}{b-a}\right) \]
     */
    @Override
    public double getMean() {
        return nonCentralMoment(1);
    }

    /**
     * {@inheritDoc}
     *
     * <p>For lower limit {@code a}, start of the density constant region {@code b},
     * end of the density constant region {@code c} and upper limit {@code d}, the
     * variance is:
     *
     * <p>\[ \frac{1}{6(d+c-b-a)}\left(\frac{d^4-c^4}{d-c}-\frac{b^4-a^4}{b-a}\right) - \mu^2 \]
     *
     * <p>where \( \mu \) is the mean.
     */
    @Override
    public double getVariance() {
        final double mu = getMean();
        return nonCentralMoment(2) - mu * mu;
    }

    /**
     * Compute the {@code k}-th non-central moment.
     *
     * @param k Moment to compute
     * @return the moment
     */
    private double nonCentralMoment(int k) {
        final double dc = Math.pow(d, k + 2) - Math.pow(c, k + 2);
        final double ba = Math.pow(b, k + 2) - Math.pow(a, k + 2);
        return 2 * ((dc / dmc - ba / bma) / divisor / ((k + 1) * (k + 2)));
    }

    /**
     * {@inheritDoc}
     *
     * <p>The lower bound of the support is equal to the lower limit parameter
     * {@code a} of the distribution.
     */
    @Override
    public double getSupportLowerBound() {
        return a;
    }

    /**
     * {@inheritDoc}
     *
     * <p>The upper bound of the support is equal to the upper limit parameter
     * {@code d} of the distribution.
     */
    @Override
    public double getSupportUpperBound() {
        return d;
    }
}
