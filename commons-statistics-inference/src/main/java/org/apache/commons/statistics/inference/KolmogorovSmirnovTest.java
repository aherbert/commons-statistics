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
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;

import org.apache.commons.numbers.combinatorics.BinomialCoefficientDouble;
import org.apache.commons.numbers.combinatorics.Factorial;
import org.apache.commons.numbers.core.ArithmeticUtils;
import org.apache.commons.numbers.core.Sum;
import org.apache.commons.rng.UniformRandomProvider;

/**
 * Implements the Kolmogorov-Smirnov (K-S) test for equality of continuous distributions.
 *
 * <p>The K-S test uses a statistic based on the maximum deviation of the empirical distribution of
 * sample data points from the distribution expected under the null hypothesis. For one-sample tests
 * evaluating the null hypothesis that a set of sample data points follow a given distribution, the
 * test statistic is \(D_n=\sup_x |F_n(x)-F(x)|\), where \(F\) is the expected distribution and
 * \(F_n\) is the empirical distribution of the \(n\) sample data points. The distribution of
 * \(D_n\) is estimated using the method presented in [2].
 *
 * <p>Two-sample tests are also supported, evaluating the null hypothesis that the two samples
 * {@code x} and {@code y} come from the same underlying distribution. In this case, the test
 * statistic is \(D_{n,m}=\sup_t | F_n(t)-F_m(t)|\) where \(n\) is the length of {@code x}, \(m\) is
 * the length of {@code y}, \(F_n\) is the empirical distribution that puts mass \(1/n\) at each of
 * the values in {@code x} and \(F_m\) is the empirical distribution of the {@code y} values. The
 * default 2-sample test method, {@link #test(double[], double[], Options)} works as
 * follows:
 *
 * <ul>
 * <li>When both sample sizes are less than 10000, the method presented in [5]
 * is used to compute the exact p-value for the 2-sample test. The estimation of the p-value
 * can be expressed using strict or non-strict inequality.
 * <li>When the sample sizes, m and n, are larger the asymptotic
 * distribution of \(D_{n,m}\) is used. The p-value is \(1 - CDF(d, \sqrt{mn / (m + n)})\)
 * where CDF is the cumulative density function of the two-sided one-sample Kolmogorov-Smirnov
 * distribution.
 * </ul>
 *
 * <p>For small samples (former case), if the data contains ties, these can be resolved
 * using random ordering of tied values. This effectively changes tied values so that
 * they are considered different. This is to be used when samples should not match but do
 * due to the limited precision of their {@code double} representation. Alternatively,
 * the {@link #estimateP(double[], double[], UniformRandomProvider, int, AlternativeHypothesis, boolean)}
 * method, modeled after <a href="http://sekhon.berkeley.edu/matching/ks.boot.html">ks.boot</a>
 * in the R Matching package [3], can be used if ties are known to be present in the data.
 *
 * <p>In the two-sample case, \(D_{n,m}\) has a discrete distribution. This makes the p-value
 * associated with the null hypothesis \(H_0 : D_{n,m} &gt; d \) differ from \(H_0 : D_{n,m} \ge d \)
 * by the mass of the observed value \(d\). To distinguish these, the two-sample tests use a boolean
 * {@code strict} parameter. This parameter is ignored for large samples.
 *
 * <p>References:
 * <ol>
 * <li>
 * Marsaglia, G., Tsang, W. W., & Wang, J. (2003).
 * <a href="https://doi.org/10.18637/jss.v008.i18">Evaluating Kolmogorov's Distribution.</a>
 * Journal of Statistical Software, 8(18), 1–4.
 * <li>Simard, R., & L’Ecuyer, P. (2011).
 * <a href="https://doi.org/10.18637/jss.v039.i11">Computing the Two-Sided Kolmogorov-Smirnov Distribution.</a>
 * Journal of Statistical Software, 39(11), 1–18.
 * <li>Sekhon, J. S. (2011).
 * <a href="https://doi.org/10.18637/jss.v042.i07">
 * Multivariate and Propensity Score Matching Software with Automated Balance Optimization:
 * The Matching package for R.</a>
 * Journal of Statistical Software, 42(7), 1–52.
 * <li>Wilcox, Rand. 2012. Introduction to Robust Estimation and Hypothesis Testing,
 * Chapter 5, 3rd Ed. Academic Press.
 * <li>Viehmann, T (2021).
 * <a href="https://doi.org/10.48550/arXiv.2102.08037">
 * Numerically more stable computation of the p-values for the two-sample Kolmogorov-Smirnov test.</a>
 * arXiv:2102.08037
 * <li>Hodges, J. L. (1958).
 * <a href="https://doi.org/10.1007/BF02589501">
 * The significance probability of the smirnov two-sample test.</a>
 * Arkiv for Matematik, 3(5), 469-486.
 * </ol>
 *
 * <p>Note that [1] contains an error in computing h, refer to <a
 * href="https://issues.apache.org/jira/browse/MATH-437">MATH-437</a> for details.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test">
 * Kolmogorov-Smirnov (K-S) test (Wikipedia)</a>
 * @since 1.1
 */
public final class KolmogorovSmirnovTest {
    /** Name for sample 1. */
    private static final String SAMPLE_1_NAME = "Sample 1";
    /** Name for sample 2. */
    private static final String SAMPLE_2_NAME = "Sample 2";
    /** When the largest sample size exceeds this value, 2-sample K-S test uses asymptotic
     * distribution to compute the p-value. */
    private static final int LARGE_SAMPLE = 10000;
    /** Maximum finite factorial. */
    private static final int MAX_FACTORIAL = 170;

    /**
     * Options for the Kolmogorov-Smirnov test.
     *
     * <p>This class is immutable.
     */
    public static class Options {
        /** Default options. */
        private static final Options DEFAULT_OPTIONS = new Options();

        /** Alternative hypothesis. */
        private final AlternativeHypothesis alternative;
        /** Method to compute the p-value. */
        private final PValueMethod pValue;
        /** Use a strict inequality for the two-sample exact p-value. */
        private final boolean strictInequality;

        /**
         * Builder for the {@link Options}.
         */
        public static class Builder {
            /** Alternative hypothesis. */
            private AlternativeHypothesis alternative;
            /** Method to compute the p-value. */
            private PValueMethod pValue;
            /** Use a strict inequality for the two-sample exact p-value. */
            private boolean strictInequality;

            /**
             * @param source Source to copy.
             */
            Builder(Options source) {
                alternative = source.alternative;
                pValue = source.pValue;
                strictInequality = source.strictInequality;
            }

            /**
             * Sets the alternative hypothesis.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#getAlternative()
             */
            public Builder setAlternative(AlternativeHypothesis v) {
                this.alternative = Objects.requireNonNull(v);
                return this;
            }

            /**
             * Sets the method to compute the p-value.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#getPValueMethod()
             */
            public Builder setPValueMethod(PValueMethod v) {
                this.pValue = Objects.requireNonNull(v);
                return this;
            }

            /**
             * Set to {@code true} to compute the two-sample exact p-value using a strict inquality.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#isStrictInequality()
             */
            public Builder setStrictInequality(boolean v) {
                this.strictInequality = v;
                return this;
            }

            /**
             * Builds the options.
             *
             * @return the options
             */
            Options build() {
                return new Options(this);
            }
        }

        /**
         * Create the default options.
         */
        Options() {
            alternative = AlternativeHypothesis.TWO_SIDED;
            pValue = PValueMethod.AUTO;
            strictInequality = false;
        }

        /**
         * @param source Source to copy.
         */
        Options(Builder source) {
            alternative = source.alternative;
            pValue = source.pValue;
            strictInequality = source.strictInequality;
        }

        /**
         * Return the default options.
         *
         * <ul>
         * <li>{@link #getAlternative getAlternative = two-sided}
         * <li>{@link #getPValueMethod() getPValueMethod = auto}
         * <li>{@link #isStrictInequality() isStrictInequality = false}
         * </ul>
         *
         * @return the options
         */
        public static Options defaults() {
            return DEFAULT_OPTIONS;
        }

        /**
         * Create a new {@link Builder} with the default options.
         *
         * @return the builder
         */
        public static Builder builder() {
            return DEFAULT_OPTIONS.toBuilder();
        }

        /**
         * Create a {@link Builder} from the current options.
         *
         * @return the builder
         */
        public Builder toBuilder() {
            return new Builder(this);
        }

        /**
         * Return the alternative hypothesis.
         *
         * @return the alternative hypothesis
         */
        public AlternativeHypothesis getAlternative() {
            return alternative;
        }

        /**
         * Gets the method to compute the p-value.
         *
         * <p>For the two-sided test the exact p-value is only valid if there are no matching
         * samples {@code x[i] == y[j]}; otherwise the p-value resorts to the asymptotic
         * approximation.
         *
         * @return the p-value method
         */
        public PValueMethod getPValueMethod() {
            return pValue;
        }

        /**
         * Compute the p-value for the two-sample test as
         * \(P(D_{n,m} &gt; d)\) if {@code true}; otherwise \(P(D_{n,m} \ge d)\),
         * where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic, either the two-sided
         * \(D_{n,m}\) or one-sided \(D_{n,m}^+\}.
         *
         * <p>Applies to {@link KolmogorovSmirnovTest#test(double[], double[], Options)}.
         *
         * @return true to use a strict inequality
         */
        public boolean isStrictInequality() {
            return strictInequality;
        }
    }

    /**
     * Result for the one-sample Kolmogorov-Smirnov test.
     *
     * <p>This class is immutable.
     */
    public static class OneResult extends BaseSignificanceResult {
        /** Sign of the statistic. */
        private final int sign;

        /**
         * Create an instance.
         *
         * @param statistic Test statistic.
         * @param sign Sign of the statistic.
         * @param p Result p-value.
         */
        OneResult(double statistic, int sign, double p) {
            super(statistic, p);
            this.sign = sign;
        }

        /**
         * Gets the sign of the statistic. This is 1 for D+ and -1 for D-.
         * For a two sided-test this is zero if the magnitude of D+ and D- was equal.
         *
         * @return the sign
         */
        public int getSign() {
            return sign;
        }
    }

    /**
     * Result for the two-sample Kolmogorov-Smirnov test.
     *
     * <p>This class is immutable.
     */
    public static final class TwoResult extends OneResult {
        /** Count of ties in x with y. */
        private final int tiesX;
        /** Count of ties in y with x. */
        private final int tiesY;

        /**
         * Create an instance.
         *
         * @param statistic Test statistic.
         * @param sign Sign of the statistic.
         * @param tiesX Count of ties in x with y.
         * @param tiesY Count of ties in y with x.
         * @param p Result p-value.
         */
        TwoResult(double statistic, int sign, int tiesX, int tiesY, double p) {
            super(statistic, sign, p);
            this.tiesX = tiesX;
            this.tiesY = tiesY;
        }

        /**
         * Returns the count of the number of values in x that were equal to a value in y.
         * This is the number of values that must be removed from x to remove all ties
         * between samples if y is unchanged. If this value is non-zero then the p-value
         * is an estimate.
         *
         * @return the number of ties in X
         */
        public int getTiesX() {
            return tiesX;
        }

        /**
         * Returns the count of the number of values in y that were equal to a value in x.
         * This is the number of values that must be removed from y to remove all ties
         * between samples if x is unchanged. If this value is non-zero then the p-value
         * is an estimate.
         *
         * @return the number of ties in Y
         */
        public int getTiesY() {
            return tiesY;
        }
    }

    /** No instances. */
    private KolmogorovSmirnovTest() {}

    /**
     * Computes the one-sample Kolmogorov-Smirnov test statistic.
     *
     * <ul>
     * <li>two-sided: \(D_n=\sup_x |F_n(x)-F(x)|\)
     * <li>greater: \(D_n^+=\sup_x F_n(x)-F(x)\)
     * <li>less: \(D_n^-=\sup_x F(x)-F_n(x)\)
     * </ul>
     *
     * <p>where \(F\) is the distribution cumulative density function ({@code cdf}),
     * \(n\) is the length of {@code x} and \(F_n\) is the empirical distribution that puts
     * mass \(1/n\) at each of the values in {@code x}.
     *
     * <p>The cumulative distribution function should map a real value {@code x} to a probability
     * in [0, 1]. To use a reference distribution the CDF can be passed using a method reference:
     * <pre>
     * UniformContinuousDistribution dist = UniformContinuousDistribution.of(0, 1);
     * UniformRandomProvider rng = RandomSource.KISS.create(123);
     * double[] x = dist.sampler(rng).samples(100);
     * double d = KolmogorovSmirnovTest.statistic(x, dist::cumulativeProbability, AlternativeHypothesis.TWO_SIDED);
     * </pre>
     *
     * @param cdf Reference cumulative distribution function.
     * @param x Sample being evaluated.
     * @param alternative Alternative hypothesis.
     * @return Kolmogorov-Smirnov statistic
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; or contains NaN values.
     */
    public static double statistic(double[] x, DoubleUnaryOperator cdf,
                                   AlternativeHypothesis alternative) {
        return computeStatistic(x, cdf, alternative, null);
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic.
     *
     * <ul>
     * <li>two-sided: \(D_{n,m}=\sup_x |F_n(x)-F_m(x)|\)
     * <li>greater: \(D_{n,m}^+=\sup_x F_n(x)-F(x)\)
     * <li>less: \(D_{n,m}^-=\sup_x F(x)-F_n(x)\)
     * </ul>
     *
     * <p>where \(n\) is the length of {@code x}, \(m\) is the length of {@code y}, \(F_n\) is the
     * empirical distribution that puts mass \(1/n\) at each of the values in {@code x} and \(F_m\)
     * is the empirical distribution that puts mass \(1/\) of the {@code y} values.
     *
     * @param x First sample.
     * @param y Second sample.
     * @param alternative Alternative hypothesis.
     * @return Kolmogorov-Smirnov statistic
     * @throws IllegalArgumentException if either {@code x} or {@code y} do not have length at
     *         least 2; or contain NaN values.
     */
    public static double statistic(double[] x, double[] y,
                                   AlternativeHypothesis alternative) {
        final double n = checkArrayLength(x);
        final double m = checkArrayLength(y);
        // Clone to avoid destructive modification of input
        return computeIntegralKolmogorovSmirnovStatistic(x.clone(), y.clone(),
                alternative, null, null) / (n * m);
    }

    /**
     * Performs a one-sample Kolmogorov-Smirnov test evaluating the null hypothesis
     * that {@code x} conforms to the distribution cumulative density function ({@code cdf}).
     *
     * @param x Sample being being evaluated.
     * @param cdf Reference cumulative distribution function.
     * @return test result
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; or contains NaN values.
     * @see #test(double[], DoubleUnaryOperator, Options)
     */
    public static OneResult test(double[] x, DoubleUnaryOperator cdf) {
        return test(x, cdf, Options.defaults());
    }

    /**
     * Performs a one-sample Kolmogorov-Smirnov test evaluating the null hypothesis
     * that {@code x} conforms to the distribution cumulative density function ({@code cdf}).
     *
     * <p>The test is defined by the {@link AlternativeHypothesis}:
     * <ul>
     * <li>Two-sided evaluates the null hypothesis that the two distributions are
     * identical, \(F_n(i) = F(i)\) for all \( i \); the alternative is that the are not
     * identical. The statistic is \( max(D_n^+, D_n^-) \) and the sign of \( D \) is provided.
     * <li>Greater evaluates the null hypothesis that the \(F_n(i) &lt;= F(i)\) for all \( i \);
     * the alternative is \(F_n(i) &gt; F(i)\) for at least one \( i \). The statistic is \( D_n^+ \).
     * <li>Less evaluates the null hypothesis that the \(F_n(i) &gt;= F(i)\) for all \( i \);
     * the alternative is \(F_n(i) &lt; F(i)\) for at least one \( i \). The statistic is \( D_n^- \).
     * </ul>
     *
     * <p>The p-value method defaults to exact. The one-sided p-value uses Smirnov's stable formula:
     *
     * <p>\[ P(D_n^+ \ge x) = x \sum_{j=0}^{\floor{n(1-x)}} \binom(n, j) (\frac{j}{n} + x)^{j-1} (1-x-\frac{j}{n})^{n-j} \]
     *
     * <p>The two-sided test p-value is computed using methods described in
     * Simard &amp; L’Ecuyer (2011). The two-sided test supports an asymptotic p-value
     * using Kolmogorov's formula:
     *
     * <p>\[ \lim_{n\to\infty} P(\sqrt{n}D_n &gt; z) = 2 \sum_{i=1}^\infty (-1)^(i-1) e^{-2 i^2 z^2} \]
     *
     * @param x Sample being being evaluated.
     * @param cdf Reference cumulative distribution function.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; or contains NaN values.
     * @see #statistic(double[], DoubleUnaryOperator, AlternativeHypothesis)
     */
    public static OneResult test(double[] x, DoubleUnaryOperator cdf, Options options) {
        final AlternativeHypothesis alternative = options.getAlternative();
        final int[] sign = {0};
        final double d = computeStatistic(x, cdf, alternative, sign);
        double p;
        if (alternative == AlternativeHypothesis.TWO_SIDED) {
            PValueMethod method = options.getPValueMethod();
            if (method == PValueMethod.AUTO) {
                // No switch the asymptotic for large n
                method = PValueMethod.EXACT;
            }
            if (method == PValueMethod.ASYMPTOTIC) {
                // Kolmogorov's asymptotic formula using z = sqrt(n) * d
                p = KolmogorovSmirnovDistribution.ksSum(Math.sqrt(x.length) * d);
            } else {
                // exact
                p = KolmogorovSmirnovDistribution.Two.sf(d, x.length);
            }
        } else {
            // one-sided: always use exact
            // XXX - Support asymptotic here? What does R do?
            p = KolmogorovSmirnovDistribution.One.sf(d, x.length);
        }
        return new OneResult(d, sign[0], p);
    }

    /**
     * Performs a two-sample Kolmogorov-Smirnov test on samples {@code x} and {@code y}.
     *
     * @param x First sample.
     * @param y Second sample.
     * @return test result
     * @throws IllegalArgumentException if either {@code x} or {@code y} does
     * not have length at least 2; or contain NaN values.
     * @see #test(double[], double[], Options)
     */
    public static TwoResult test(double[] x, double[] y) {
        return test(x, y, Options.defaults());
    }

    /**
     * Performs a two-sample Kolmogorov-Smirnov test on samples {@code x} and {@code y}.
     * Test the empirical distributions \(F_n\) and \(F_m\) where \(n\) is the length
     * of {@code x}, \(m\) is the length of {@code y}, \(F_n\) is the empirical distribution
     * that puts mass \(1/n\) at each of the values in {@code x} and \(F_m\) is the empirical
     * distribution that puts mass \(1/\) of the {@code y} values.
     *
     * <p>The test is defined by the {@link AlternativeHypothesis}:
     * <ul>
     * <li>Two-sided evaluates the null hypothesis that the two distributions are
     * identical, \(F_n(i) = F_m(i)\) for all \( i \); the alternative is that the are not
     * identical. The statistic is \( max(D_n^+, D_n^-) \) and the sign of \( D \) is provided.
     * <li>Greater evaluates the null hypothesis that the \(F_n(i) &lt;= F_m(i)\) for all \( i \);
     * the alternative is \(F_n(i) &gt; F_m(i)\) for at least one \( i \). The statistic is \( D_n^+ \).
     * <li>Less evaluates the null hypothesis that the \(F_n(i) &gt;= F_m(i)\) for all \( i \);
     * the alternative is \(F_n(i) &lt; F_m(i)\) for at least one \( i \). The statistic is \( D_n^- \).
     * </ul>
     *
     * <p>If the {@link PValueMethod p-value method} is auto, then an exact p computation
     * is attempted if both sample sizes are less than 10000; otherwise an asymptotic p-value
     * is returned.
     *
     * <p>The exact p-value can be computed using a strict inquality as
     * \(P(D_{n,m} &gt; d)\); otherwise \(P(D_{n,m} \ge d)\). Computation of the exact
     * p-value requires no tied values <em>between</em> the two samples. The count of ties
     * for each sample is returned in the result. If there are ties, or the computation
     * of the exact p-value fails due to large sample sizes, then the computation
     * returns an asymptotic p-value.
     *
     * @param x First sample.
     * @param y Second sample.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if either {@code x} or {@code y} does not
     * have length at least 2; or contain NaN values.
     * @see #statistic(double[], double[], AlternativeHypothesis)
     */
    public static TwoResult test(double[] x, double[] y, Options options) {
        final int n = checkArrayLength(x);
        final int m = checkArrayLength(y);
        //final boolean exactOption = Math.max(n, m) < LARGE_SAMPLE;
        final AlternativeHypothesis alternative = options.getAlternative();
        PValueMethod method = options.getPValueMethod();
        final int[] sign = {0};
        final int[] ties = {0, 0};

        final long dnm = computeIntegralKolmogorovSmirnovStatistic(x.clone(), y.clone(),
                alternative, sign, ties);
        // Compute p-value
        // If there are ties the exact p-value is invalid.
        if (ties[0] != 0) {
            method = PValueMethod.ASYMPTOTIC;
        }
        if (method == PValueMethod.AUTO) {
            // Use exact for small samples
            method = Math.max(n, m) < LARGE_SAMPLE ?
                PValueMethod.EXACT :
                PValueMethod.ASYMPTOTIC;
        }
        final int gcd = ArithmeticUtils.gcd(n, m);
        // Note: Integer division using the gcd is intentional
        final double statistic = (dnm / gcd) / ((double) n * (m / gcd));

        double p = -1;
        final boolean twoSided = alternative == AlternativeHypothesis.TWO_SIDED;
        if (method == PValueMethod.EXACT) {
            p = twoSampleExactP(dnm, n, m, gcd, options.isStrictInequality(), twoSided);
        }
        if (p < 0) {
            p = twoSampleApproximateP(statistic, n, m, twoSided);
        }
        return new TwoResult(statistic, sign[0], ties[0], ties[1], p);
    }

    /**
     * Estimates the <i>p-value</i> of a two-sample Kolmogorov-Smirnov test
     * evaluating the null hypothesis that {@code x} and {@code y} are samples
     * drawn from the same probability distribution.
     *
     * <p>This method estimates the p-value by repeatedly sampling sets of size
     * {@code x.length} and {@code y.length} from the empirical distribution
     * of the combined sample. The memory requirement is an array copy of each of
     * the input arguments.
     *
     * <p>When {@code strict} is true, this is equivalent to the algorithm implemented
     * in the R function {@code ks.boot} as described in Sekhon (2011)
     * Journal of Statistical Software, 42(7), 1–52 [3].
     *
     * @param x First sample.
     * @param y Second sample.
     * @param rng Source of randomness for sampling.
     * @param iterations Number of bootstrap resampling iterations.
     * @param alternative Alternative hypothesis.
     * @param strict Whether or not the null hypothesis is expressed as a strict inequality.
     * @return the estimated p-value.
     * @throws IllegalArgumentException if either {@code x} or {@code y} does
     * not have length at least 2; contain NaN values; or the number of iterations is
     * not strictly positive.
     */
    public static double estimateP(double[] x, double[] y,
                                   UniformRandomProvider rng,
                                   int iterations,
                                   AlternativeHypothesis alternative,
                                   boolean strict) {
        checkArrayLength(x);
        checkArrayLength(y);
        InferenceUtils.checkNonNaN(x);
        InferenceUtils.checkNonNaN(y);
        InferenceUtils.checkStrictlyPositive(iterations);

        // Obtain the original sample statistic. Copy the input to avoid modifying it.
        final double[] sx = x.clone();
        final double[] sy = y.clone();
        final long dnm = computeIntegralKolmogorovSmirnovStatistic(sx, sy, alternative);

        // Test if the random statistic is greater (strict), or greater or equal to d
        final long d = strict ? dnm : dnm - 1;

        // Edge case where all possible d will be greater
        if (d < 0) {
            return 1;
        }

        // Sample randomly with replacement from the combined distribution.
        final DoubleSupplier gen = createSampler(x, y, rng);
        int count = 0;
        for (int i = iterations; i > 0; i--) {
            for (int j = 0; j < sx.length; j++) {
                sx[j] = gen.getAsDouble();
            }
            for (int j = 0; j < sy.length; j++) {
                sy[j] = gen.getAsDouble();
            }
            if (computeIntegralKolmogorovSmirnovStatistic(sx, sy, alternative) > d) {
                count++;
            }
        }
        return count / (double) iterations;
    }

    /**
     * Computes the magnitude of the one-sample Kolmogorov-Smirnov test statistic.
     * The sign of the statistic is optionally returned in {@code sign}. For the two-sided case
     * the sign is 0 if the magnitude of D+ and D- was equal; otherwise it indicates which D
     * had the larger magnitude.
     *
     * @param x Sample being evaluated.
     * @param cdf Reference cumulative distribution function.
     * @param alternative Alternative hypothesis.
     * @param sign Sign of the statistic (null or non-zero length).
     * @return Kolmogorov-Smirnov statistic
     * @throws IllegalArgumentException if {@code data} does not have length at least 2;
     * or contains NaN values.
     */
    private static double computeStatistic(double[] x, DoubleUnaryOperator cdf,
                                           AlternativeHypothesis alternative, int[] sign) {
        final int n = checkArrayLength(x);
        final double nd = n;
        final double[] sx = sort(x.clone(), "Sample");
        // Note: ties in the data do not matter as we compare the empirical CDF
        // immediately before the value (i/n) and at the value (i+1)/n. For ties
        // of length m this would be (i-m+1)/n and (i+1)/n and the result is the same.
        double d = 0;
        if (alternative == AlternativeHypothesis.GREATER_THAN) {
            // Compute D+
            for (int i = 0; i < n; i++) {
                final double yi = cdf.applyAsDouble(sx[i]);
                final double dp = (i + 1) / nd - yi;
                d = dp > d ? dp : d;
            }
            setSign(sign, 1);
        } else if (alternative == AlternativeHypothesis.LESS_THAN) {
            // Compute D-
            for (int i = 0; i < n; i++) {
                final double yi = cdf.applyAsDouble(sx[i]);
                final double dn = yi - i / nd;
                d = dn > d ? dn : d;
            }
            setSign(sign, -1);
        } else {
            // Two sided.
            // Compute both (as unsigned) and return the sign indicating the largest result.
            double plus = 0;
            double minus = 0;
            for (int i = 0; i < n; i++) {
                final double yi = cdf.applyAsDouble(sx[i]);
                final double dn = yi - i / nd;
                final double dp = (i + 1) / nd - yi;
                minus = dn > minus ? dn : minus;
                plus = dp > plus ? dp : plus;
            }
            setSign(sign, Double.compare(plus, minus));
            d = Math.max(plus, minus);
        }
        return d;
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic. The statistic is integral
     * and has a value in {@code [0, n*m]}. Not all values are possible; the smallest
     * increment is the greatest common divisor of {@code n} and {@code m}.
     *
     * <p>This method will destructively modify the input arrays (via a sort).
     *
     * <p>The sign of the statistic is optionally returned in {@code sign}. For the two-sided case
     * the sign is 0 if the magnitude of D+ and D- was equal; otherwise it indicates which D
     * had the larger magnitude. If the two-sided statistic is zero the two arrays are
     * identical, or are 'identical' data of a single value (sample sizes may be different),
     * or have a sequence of ties of 'identical' data with a net zero effect on the D statistic
     * e.g.
     * <pre>
     *  [1,2,3]           vs [1,2,3]
     *  [0,0,0,0]         vs [0,0,0]
     *  [0,0,0,0,1,1,1,1] vs [0,0,0,1,1,1]
     * </pre>
     *
     * <p>This method detects ties in the input data; if the {@code hasTies} array is not null
     * then the presence of ties is recorded in the first position of the {@code hasTies} array.
     *
     * @param x First sample (destructively modified by sort).
     * @param y Second sample  (destructively modified by sort).
     * @param alternative Alternative hypothesis.
     * @param sign Sign of the statistic (null or non-zero length).
     * @param ties Count of tiesFlag set to true if the input data has ties (null or non-zero length).
     * @return integral Kolmogorov-Smirnov statistic
     * @throws IllegalArgumentException if either {@code x} or {@code y} contain NaN values.
     */
    private static long computeIntegralKolmogorovSmirnovStatistic(double[] x, double[] y,
            AlternativeHypothesis alternative, int[] sign, int[] ties) {
        // Sort the sample arrays
        final double[] sx = sort(x, SAMPLE_1_NAME);
        final double[] sy = sort(y, SAMPLE_2_NAME);
        final int n = sx.length;
        final int m = sy.length;

        // CDFs range from 0 to 1 using increments of 1/n and 1/m for x and y respectively.
        // Scale by n*m to use increments of m and n for x and y.
        // Find the max difference between cdf_x and cdf_y
        int i = 0;
        int j = 0;
        long d = 0;
        long plus = 0;
        long minus = 0;
        // Tie counts
        int tx = 0;
        int ty = 0;
        do {
            // No NaNs so compare using < and >
            if (sx[i] < sy[j]) {
                final double z = sx[i];
                do {
                    i++;
                    d += m;
                } while (i < n && x[i] == z);
                plus = d > plus ? d : plus;
            } else if (sx[i] > sy[j]) {
                final double z = sy[j];
                do {
                    j++;
                    d -= n;
                } while (j < m && y[j] == z);
                minus = d < minus ? d : minus;
            } else {
                // Traverse to the end of the tied section
                // Count the ties in x and y
                final double z = sx[i];
                int k = i;
                do {
                    i++;
                } while (i < n && sx[i] == z);
                k = i - k;
                tx += k;
                d += k * (long) m;
                k = j;
                do {
                    j++;
                } while (j < m && sy[j] == z);
                k = j - k;
                ty += k;
                d -= k * (long) n;
                if (d > plus) {
                    plus = d;
                } else if (d < minus) {
                    minus = d;
                }
            }
        } while (i < n && j < m);
        if (ties != null) {
            ties[0] = tx;
            ties[1] = ty;
        }
        if (alternative == AlternativeHypothesis.GREATER_THAN) {
            setSign(sign, 1);
            return plus;
        } else if (alternative == AlternativeHypothesis.LESS_THAN) {
            setSign(sign, -1);
            return -minus;
        } else {
            // Two sided.
            setSign(sign, Double.compare(plus, -minus));
            return Math.max(plus, -minus);
        }
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic. The statistic is integral
     * and has a value in {@code [0, n*m]}. Not all values are possible; the smallest
     * increment is the greatest common divisor of {@code n} and {@code m}.
     *
     * <p>This method will destructively modify the input arrays (via a sort). This method
     * is a specialized version of
     * {@link #computeIntegralKolmogorovSmirnovStatistic(double[], double[], AlternativeHypothesis, int[], int[])}
     * for use in the estimation of p-values. It omits detection of ties and computation of the
     * sign of the two-sided statistic.
     *
     * @param x First sample (destructively modified by sort; must not contain NaN).
     * @param y Second sample  (destructively modified by sort; must not contain NaN).
     * @param alternative Alternative hypothesis.
     * @return integral Kolmogorov-Smirnov statistic
     */
    private static long computeIntegralKolmogorovSmirnovStatistic(double[] x, double[] y,
            AlternativeHypothesis alternative) {
        return computeIntegralKolmogorovSmirnovStatistic(x, y, alternative, null, null);
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic.
     *
     * <p>This method corrects ties between samples. Any sequence of tied values is randomly
     * shuffled and processed as if the randomly ordered tied values were unique.
     *
     * <p>Note: The input arrays are assumed to be sorted with no NaN values.
     * This is for convenience to allow arrays sorted by previous computation to be reused
     * when ties are detected.
     *
     * <p><strong>Warning: </strong>Note that the statistic is <em>signed</em>. Effectively it is
     * {@code max(|D+|, |D-|)}. If the statistic is zero the two arrays are identical, or
     * are 'identical' data of a single value (sample sizes may be different).
     *
     * @param sx First sorted sample
     * @param sy Second sorted sample
     * @param seed Random seed.
     * @return signed test statistic \(n m D_{n,m}\) used to evaluate the null hypothesis that
     * {@code x} and {@code y} represent samples from the same underlying distribution
     */
    // XXX - move this to an option in estimateP to walk up to n-random paths through tied sections
    static long integralKolmogorovSmirnovStatisticRandomTies(double[] sx, double[] sy, long seed) {
        final int n = sx.length;
        final int m = sy.length;

        // Small state size RNG for fast creation
        final UniformRandomProvider rng = new SplittableRandom(seed)::nextLong;

        int i = 0;
        int j = 0;
        long d = 0;
        long plus = 0;
        long minus = 0;
        // Count length of tied data
        int cx;
        int cy;
        do {
            // No NaN values so compare using <= (not Double.compare)
            final double z = sx[i] <= sy[j] ? sx[i] : sy[j];
            cx = i;
            cy = j;
            while (i < n && sx[i] <= z) {
                i++;
            }
            while (j < m && sy[j] <= z) {
                j++;
            }
            // lengths
            cx = i - cx;
            cy = j - cy;

            // Only cx or cy should be non-zero (not both), else there are ties between samples.
            // Detect using sign bit xor of adjusted counts. If both are above -1
            // the xor will be positive and ties are present.
            if (((cx - 1) ^ (cy - 1)) >= 0) {
                // Resolve tie randomly.
                do {
                    // A negative value picks x, else y.
                    if (rng.nextInt(-cx, cy) < 0) {
                        cx--;
                        d += m;
                        plus = d > plus ? d : plus;
                    } else {
                        cy--;
                        d -= n;
                        minus = d < minus ? d : minus;
                    }
                } while (((cx - 1) ^ (cy - 1)) >= 0);
            }

            // Here only 1 of cx or cy are non-zero.
            // Handle ties within the same sample using a multiple of the count.
            if (cx == 0) {
                d -= (long) cy * n;
                minus = d < minus ? d : minus;
            } else {
                d += (long) cx * m;
                plus = d > plus ? d : plus;
            }
        } while (i < n && j < m);
        // This returns the signed max difference
        return -minus > plus ? minus : plus;
    }


    /**
     * Creates a sampler to sample randomly from the combined distribution of the two samples.
     *
     * @param x First sample.
     * @param y Second sample.
     * @param rng Source of randomness.
     * @return the sampler
     */
    private static DoubleSupplier createSampler(double[] x, double[] y,
                                                UniformRandomProvider rng) {
        final int n = x.length;
        final int m = y.length;
        // Support sampling with maximum length arrays
        // (where a concatenated array is not possible)
        // by choosing one or the other.
        // - generate i in [-n, m)
        // - return i < 0 ? x[n + i] : y[i]
        // The sign condition is a 50-50 branch.
        // Perform branchless by extracting the sign bit to pick the array.
        final IntToDoubleFunction nextX = i -> x[n + i];
        final IntToDoubleFunction nextY = i -> y[i];
        // Arrange function which accepts the negative index at position [1]
        final IntToDoubleFunction[] next = {nextY, nextX};
        return () -> {
            final int i = rng.nextInt(-n, m);
            return next[i >>> 31].applyAsDouble(i);
        };
    }

    /**
     * Computes \(P(D_{n,m} &gt; d)\) if {@code strict} is {@code true}; otherwise \(P(D_{n,m} \ge
     * d)\), where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic, either the two-sided
     * \(D_{n,m}\) or one-sided \(D_{n,m}^+\}. See
     * {@link #statistic(double[], double[], AlternativeHypothesis)} for the definition of \(D_{n,m}\).
     *
     * <p>The returned probability is exact. If the value cannot be computed this returns -1.
     *
     * <p>Note: This requires the greatest common divisor of n and m. The integral D statistic
     * in the range [0, n*m] is separated by increments of the gcd. The method will only
     * compute p-values for valid values of D by calculating for D/gcd.
     * Strict inquality is performed using the next valid value for D.
     *
     * @param dnm Integral D-statistic value (in [0, n*m]).
     * @param n First sample size.
     * @param m Second sample size.
     * @param gcd Greatest common divisor of n and m.
     * @param strict whether or not the probability to compute is expressed as a strict inequality.
     * @param twoSided whether D refers to D or D+.
     * @return probability that a randomly selected m-n partition of m + n generates D
     *         greater than (resp. greater than or equal to) {@code d} (or -1)
     */
    static double twoSampleExactP(long dnm, int n, int m, int gcd, boolean strict, boolean twoSided) {
        // Create the statistic in [0, lcm]
        // For strict inequality D > d the result is the same if we compute for D >= (d+1)
        final long d = dnm / gcd + (strict ? 1 : 0);

        // P-value methods compute for d <= lcm (least common multiple)
        final long lcm = (long) n * (m / gcd);
        if (d > lcm) {
            return 0;
        }

        // Note: Some methods require m >= n, others n >= m
        final int a = Math.min(n, m);
        final int b = Math.max(n, m);

        if (twoSided) {
            // Any two-sided statistic dnm cannot be less than min(n, m) in the absence of ties.
            if (d * gcd <= a) {
                return 1;
            }
            // Here d in [2, lcm]
            if (n == m) {
                return twoSampleTwoSidedPOutsideSquare(d, n);
            }
            return twoSampleTwoSidedPStabilizedInner(d, b, a, gcd);
        }
        // Any one-sided statistic cannot be less than 0
        if (d <= 0) {
            return 1;
        }
        // Here d in [1, lcm]
        if (n == m) {
            return twoSampleOneSidedPOutsideSquare(d, n);
        }
        return twoSampleOneSidedPOutside(d, a, b, gcd);
    }

    /**
     * Computes \(P(D_{n,m} \ge d)\), where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic.
     *
     * <p>The returned probability is exact, implemented using the stabilized inner method
     * presented in Viehmann (2021).
     *
     * @param d Integral D-statistic value (in [2, lcm])
     * @param n First sample size.
     * @param m Second sample size.
     * @param gcd Greatest common divisor of n and m.
     * @return probability that a randomly selected m-n partition of m + n generates \(D_{n,m}\)
     *         greater than or equal to {@code d}
     */
    private static double twoSampleTwoSidedPStabilizedInner(long d, int n, int m, int gcd) {
        // Total paths is binom(m+n, n). Check this is possible.
        // XXX - binom is not actually used in the computation.
        // Update this when better limits are known
        final long lm = m;
        if (n + lm > Integer.MAX_VALUE) {
            return -1;
        }
        final double binom = binom(m + n, n);
        if (binom == Double.POSITIVE_INFINITY) {
            return -1;
        }

        // This could be updated to use d in [1, lcm].
        // Currently it uses d in [gcd, n*m].
        // Largest intermediate value is (dnm + im + n) which is within 2^63
        // if n and m are 2^31-1, i = n, dnm = n*m: (2^31-1)^2 + (2^31-1)^2 + 2^31-1 < 2^63
        final long dnm = d * gcd;

        // Viehmann (2021): Updated for i in [0, n], j in [0, m]
        // C_i,j = 1                                      if |i/n - j/m| >= d
        //       = 0                                      if |i/n - j/m| < d and (i=0 or j=0)
        //       = C_i-1,j * i/(i+j) + C_i,j-1 * j/(i+j)  otherwise
        // P2 = C_m,n
        // Note: The python listing in Viehmann used d in [0, 1]. This uses dnm in [0, nm]
        // so updates the scaling to compute the ranges. Also note that the listing uses
        // dist > d or dist < -d where this uses |dist| >= d to compute P(D >= d) (non-strict inequality).
        // The provided listing is explicit in the values for each j in the range.
        // It can be optimized given the known start and end j for each iteration as only
        // j where |i/n - j/m| < d must be processed.

        // First iteration with i = 0
        // if j/m >= d: C_i,j=1, else 0.
        // j/m >= d => j*n >= dnm
        // j = ceil(dnm / n)
        int endJ = Math.min(m + 1, (int) ((dnm + n - 1) / n));

        // Only require 1 array to store C_i-1,j as the startJ only ever increases
        // and we update lower indices using higher ones.
        // The maximum value addressed is j=m or less using j*n <= 2 * d*m*n : j = 2 * ceil(2*d*m) + 1
        // size = int(2*m*d + 2)
        final double[] cij = new double[Math.min(m + 1, 2 * endJ + 1)];

        // Each iteration fills C_i,j with values and the remaining values are
        // kept as 1 for |i/n - j/m| >= d
        int length = endJ;
        for (int j = endJ; j < cij.length; j++) {
            assert j * (long) n >= dnm;
            cij[j] = 1;
        }
        for (int j = endJ; j-- > 0;) {
            assert j * (long) n < dnm;
        }

        int startJ = 0;
        double val = -1;
        long im = 0;
        for (int i = 1; i <= n; i++) {
            im += m;
            final int lastStartJ = startJ;
            final int lastLength = length;

            // startJ where: im - jn < dnm : jn > im - dnm
            // j = floor((im - dnm) / n) + 1      in [0, m]
            startJ = im < dnm ? 0 : Math.min(m, (int) ((im - dnm) / n) + 1);

            // endJ where: jn - im >= dnm
            // j = ceil((dnm + im) / n)         in [0, m+1]
            endJ = Math.min(m + 1, (int) ((dnm + im + n - 1) / n));

            // Compute C_i,j for minJ <= j < maxJ
            if (endJ <= startJ) {
                // No possible paths inside the boundary
                return 1;
            }

            // Initialize previous value C_i,j-1
            val = startJ == 0 ? 0 : 1;

            assert startJ <= 0 || Math.abs(im - (startJ - 1) * (long) n) >= dnm : "startJ " + startJ;
            for (int j = startJ; j < endJ; j++) {
                assert j == 0 || Math.abs(im - j * (long) n) < dnm : "startJ <= j < endJ";
                // C_i,j = C_i-1,j * i/(i+j) + C_i,j-1 * j/(i+j)
                val = (cij[j - lastStartJ] * i + val * j) / ((double) i + j);
                cij[j - startJ] = val;
            }

            for (int j = endJ; j < cij.length; j++) {
                assert Math.abs(im - j * (long) n) >= dnm : "j >= endJ";
            }

            // Must keep the remaining values in C_i,j as 1 to allow
            // cij[j - lastStartJ] * i == i when (j-lastStartJ) > lastLength
            length = endJ - startJ;
            for (int j = lastLength - length - 1; j >= 0; j--) {
                cij[length + j] = 1;
            }

//            // startJ = int(m * (i/n + d)) + 1-size
//            // => int(mi/n + dnm/n) + 1 - int(2*dnm/n + 2)
//            // => int(mi/n + dnm/n) - int(2*dnm/n) - 1
//            // If this is too low the loop just does more work.
//            int startJ2 = Math.max((int) ((im - dnm) / n) - 1, 0);
//            // j is limited to the range [0, m]
//            startJ2 = Math.min(startJ2, m);
//
//            // First iteration with jj = 0. j = startJ.
//            // Note: Two conditions can be dropped:
//            // 1. max(startJ - lastStartJ) = ceil(m/n)
//            // => j - lastStartJ >= size is not possible
//            // 2. if startJ > 0 it occurs when
//            // (j+1)*n <= im - dnm => im - j*n > dnm
//            // if |im - jn| <= dnm then j must be 0 and val=0.
//            // Cannot check j==0 as im > d is possible.
//            long jn = startJ2 * (long) n;
//            // dist = |i/n - j/m| => |i*m - j*n|
//            val = Math.abs(im - jn) >= dnm ? 1 : 0;
//            row2[0] = val;
//
//            // Remaining loop omits check for i=0 or j=0
//            // Only loop over indices that address the row.
//            int j;
//            final int max = size + lastStartJ2 - startJ2;
//            for (int jj = 1; jj < max; jj++) {
//                j = jj + startJ2;
//                jn += n;
//                // dist = |i/n - j/m| => |i*m - j*n|
//                if (Math.abs(im - jn) >= dnm) {
//                    val = 1;
//                // else if j==0:                       [Skip as j > 0]
//                //   val = 0
//                // else if (j - lastStartJ >= size):   [Skip due to max]
//                //   val = (i + val * j) / ((double) i + j)
//                } else {
//                    val = (row2[j - lastStartJ2] * i + val * j) / ((double) i + j);
//                }
//                row2[jj] = val;
//            }
//            // Fill in the remaining values for (j - lastStartJ >= size)
//            for (int jj = max; jj < size; jj++) {
//                j = jj + startJ2;
//                jn += n;
//                if (Math.abs(im - jn) >= dnm) {
//                    // val=1 and the remaining values are all 1
//                    Arrays.fill(row2, jj, size, 1);
//                    break;
//                }
//                val = (i + val * j) / ((double) i + j);
//                row2[jj] = val;
//            }
//            lastStartJ2 = startJ2;

            // What is different between row and row2 ?
//            System.out.printf("row  %s %d-%d%nrow2 %s %d (%d)%n",
//                    Arrays.toString(cij), startJ, endJ,
//                    Arrays.toString(row2), startJ2, max, size);
        }
//        System.out.printf("%s vs %s%n", cij[endJ - startJ - 1], row2[m - lastStartJ2]);
        // Return the most recently written value
        return val; //cij[endJ - startJ - 1];
        //return row2[m - lastStartJ2];
    }

    /**
     * Computes \(P(D_{n,m}^+ \ge d)\), where \(D_{n,m}^+\) is the 2-sample one-sided
     * Kolmogorov-Smirnov statistic.
     *
     * <p>The returned probability is exact, implemented using the outer method
     * presented in Hodges (1958).
     *
     * <p>This method will fail-fast and return -1 if the computation of the
     * numbers of paths overflows.
     *
     * @param d Integral D-statistic value (in [0, lcm])
     * @param n First sample size.
     * @param m Second sample size.
     * @param gcd Greatest common divisor of n and m.
     * @return probability that a randomly selected m-n partition of m + n generates \(D_{n,m}\)
     *         greater than or equal to {@code d}
     */
    private static double twoSampleOneSidedPOutside(long d, int n, int m, int gcd) {
        // Hodges, Fig.2
        // Lower boundary: (nx - my)/nm >= d : (nx - my) >= dnm
        // B(x, y) is the number of ways from (0, 0) to (x, y) without previously
        // reaching the boundary.
        // B(x, y) = binom(x+y, y) - [number of ways which previously reached the boundary]
        // Total paths:
        // sum_y { B(x, y) binom(m+n-x-y, n-y) }

        // Normalized by binom(m+n, n). Check this is possible.
        final long lm = m;
        if (n + lm > Integer.MAX_VALUE) {
            return -1;
        }
        final double binom = binom(m + n, n);
        if (binom == Double.POSITIVE_INFINITY) {
            return -1;
        }

        // This could be updated to use d in [1, lcm].
        // Currently it uses d in [gcd, n*m].
        final long dnm = d * gcd;

        // Visit all x in [0, m] where (nx - my) >= d for each increasing y in [0, n].
        // x = ceil( (d + my) / n ) = (d + my + n - 1) / n
        // y = ceil( (nx - d) / m ) = (nx - d + m - 1) / m
        // Note: n m integer, d in [0, nm], the intermediate cannot overflow a long.
        // x | y=0 = (d + n - 1) / n
        final int x0 = (int) ((dnm + n - 1) / n);
        if (x0 >= m) {
            return 1 / binom;
        }
        // The y above is the y *on* the boundary. Set the limit as the next y above:
        // y | x=m = 1 + floor( (nx - d) / m ) = 1 + (nm - d) / m
        final int maxy = (int) ((n * lm - dnm + m) / m);
        // Compute x and B(x, y) for visited B(x,y)
        final int[] xy = new int[maxy];
        final double[] bxy = new double[maxy];
        xy[0] = x0;
        bxy[0] = 1;
        for (int y = 1; y < maxy; y++) {
            final int x = (int) ((dnm + lm * y + n - 1) / n);
            // B(x, y) = binom(x+y, y) - [number of ways which previously reached the boundary]
            // Add the terms to subtract as a negative sum.
            final Sum b = Sum.create();
            for (int yy = 0; yy < y; yy++) {
                // Here: previousX = x - xy[yy] : previousY = y - yy
                // bxy[yy] is the paths to (previousX, previousY)
                // binom represent the paths from (previousX, previousY) to (x, y)
                b.addProduct(bxy[yy], -binom(x - xy[yy] + y - yy, y - yy));
            }
            b.add(binom(x + y, y));
            xy[y] = x;
            bxy[y] = b.getAsDouble();
        }
        // sum_y { B(x, y) binom(m+n-x-y, n-y) }
        final Sum sum = Sum.create();
        for (int y = 0; y < maxy; y++) {
            sum.addProduct(bxy[y], binom(m + n - xy[y] - y, n - y));
        }
        // No individual term should have overflowed since binom is finite.
        // Any sum above 1 is floating-point error.
        return KolmogorovSmirnovDistribution.clipProbability(sum.getAsDouble() / binom);
    }

    /**
     * Computes \(P(D_{n,n}^+ \ge d)\), where \(D_{n,n}^+\) is the 2-sample one-sided
     * Kolmogorov-Smirnov statistic.
     *
     * <p>The returned probability is exact, implemented using the outer method
     * presented in Hodges (1958).
     *
     * @param d Integral D-statistic value (in [1, lcm])
     * @param n Sample size.
     * @return probability that a randomly selected m-n partition of m + n generates \(D_{n,m}\)
     *         greater than or equal to {@code d}
     */
    private static double twoSampleOneSidedPOutsideSquare(long d, int n) {
        // Hodges (1958) Eq. 2.3:
        // p = binom(2n, n-a) / binom(2n, n)
        // a in [1, n] == d * n == dnm / n
        final int a = (int) d;

        // Rearrange:
        // p = ( 2n! / ((n-a)! (n+a)!) ) / ( 2n! / (n! n!) )
        //   = n! n! / ( (n-a)! (n+a)! )
        // Perform using pre-computed factorials if possible.
        if (n + a <= MAX_FACTORIAL) {
            final double x = Factorial.doubleValue(n);
            final double y = Factorial.doubleValue(n - a);
            final double z = Factorial.doubleValue(n + a);
            return (x / y) * (x / z);
        }
        // p = n! / (n-a)!  *  n! / (n+a)!
        //       n * (n-1) * ... * (n-a+1)
        //   = -----------------------------
        //     (n+a) * (n+a-1) * ... * (n+1)

        double p = 1;
        for (int i = 0; i < a && p != 0; i++) {
            p *= (n - i) / (1.0 + n + i);
        }
        return p;
    }

    /**
     * Computes \(P(D_{n,n}^+ \ge d)\), where \(D_{n,n}^+\) is the 2-sample two-sided
     * Kolmogorov-Smirnov statistic.
     *
     * <p>The returned probability is exact, implemented using the outer method
     * presented in Hodges (1958).
     *
     * @param d Integral D-statistic value (in [1, n])
     * @param n Sample size.
     * @return probability that a randomly selected m-n partition of n + n generates \(D_{n,n}\)
     *         greater than or equal to {@code d}
     */
    private static double twoSampleTwoSidedPOutsideSquare(long d, int n) {
        // Hodges (1958) Eq. 2.4:
        // p = 2 [ binom(2n, n-a) - binom(2n, n-2a) + binom(2n, n-3a) - ... ] / binom(2n, n)
        // a in [1, n] == d * n == dnm / n

        // As per twoSampleOneSidedPOutsideSquare, divide by binom(2n, n) and each term
        // can be expressed as a product:
        //         (             n - i                    n - i                   n - i         )
        // p = 2 * ( prod_i=0^a --------- - prod_i=0^2a --------- + prod_i=0^3a --------- + ... )
        //         (           1 + n + i                1 + n + i               1 + n + i       )
        // for ja in [1, ..., n/a]
        // Avoid repeat computation of terms by extracting common products:
        // p = 2 * ( p0a * (1 - p1a * (1 - p2a * (1 - ... ))) )
        // where each term pja is prod_i={ja}^{ja+a} for all j in [1, n / a]

        // The first term is the one-sided p.
        final double p0a = twoSampleOneSidedPOutsideSquare(d, n);
        if (p0a == 0) {
            // Underflow - nothing more to do
            return 0;
        }
        // Compute the inner-terms from small to big.
        // j = n / (d/n) ~ n*n / d
        // j is a measure of how extreme the d value is (small j is extreme d).
        // When j is above 0 a path may traverse from the lower boundary to the upper boundary.
        final int a = (int) d;
        double p = 0;
        for (int j = n / a; j > 0; j--) {
            double pja = 1;
            final int jaa = j * a + a;
            // Since p0a did not underflow we avoid the check for pj != 0
            for (int i = j * a; i < jaa; i++) {
                pja *= (n - i) / (1.0 + n + i);
            }
            p = pja * (1 - p);
        }
        p = p0a * (1 - p);
        return Math.min(1, 2 * p);
    }

    /**
     * Compute the binomial coefficient binom(n, k).
     *
     * @param n N.
     * @param k K.
     * @return binom(n, k)
     */
    private static double binom(int n, int k) {
        return BinomialCoefficientDouble.value(n, k);
    }

    /**
     * Uses the Kolmogorov-Smirnov distribution to approximate \(P(D_{n,m} &gt; d)\) where \(D_{n,m}\)
     * is the 2-sample Kolmogorov-Smirnov statistic. See
     * {@link #statistic(double[], double[], AlternativeHypothesis)} for the definition of \(D_{n,m}\).
     *
     * <p>Specifically, what is returned is \(1 - CDF(d, \sqrt{mn / (m + n)})\) where CDF
     * is the cumulative density function of the two-sided one-sample Kolmogorov-Smirnov
     * distribution.
     *
     * @param d D-statistic value.
     * @param n First sample size.
     * @param m Second sample size.
     * @param twoSided True to compute the two-sided p-value; else one-sided.
     * @return approximate probability that a randomly selected m-n partition of m + n generates
     *         \(D_{n,m}\) greater than {@code d}
     */
    static double twoSampleApproximateP(double d, int n, int m, boolean twoSided) {
        final double nn = Math.min(n, m);
        final double mm = Math.max(n, m);
        if (twoSided) {
            // Smirnov's asymptotic formula:
            // P(sqrt(N) D_n > x) = 2 \sum_{i=1}^\infty (-1)^(i-1) e^{-2 i^2 x^2}
            // x^2 = N * d * d
            // N = m*n/(m+n)
            // Comparison of twoSampleExactP(d*n*m, n, m, ...) with Two.sf or an implementation of
            // the KS sum over a range of N and d where the p-value is a typical alpha threshold
            // of 0.001 to 0.1 shows that the Two.sf has lower RMSD relative error except when N is
            // very small (e.g. 4); here neither approximate p-value is close to the exact P.
            return KolmogorovSmirnovDistribution.Two.sf(d, (int) Math.round(mm * nn / (mm + nn)));
        }
        // one-sided
        // Use Hodges Eq 5.3. Requires m >= n
        // Correct for m=n, m an integral multiple of n, and 'on the average' for m nearly equal to n
        final double z = d * Math.sqrt(nn * mm / (nn + mm));
        return Math.exp(-2 * z * z - 2 * z * (mm + 2 * nn) / Math.sqrt(mm * nn * (mm + nn)) / 3);
    }

    /**
     * Verifies that {@code array} has length at least 2.
     *
     * @param array Array to test.
     * @return the length
     * @throws IllegalArgumentException if array is too short
     */
    private static int checkArrayLength(double[] array) {
        final int n = array.length;
        if (n <= 1) {
            throw new InferenceException(InferenceException.TWO_VALUES_REQUIRED, n);
        }
        return n;
    }

    /**
     * Sort the input array. Throws an exception if NaN values are
     * present. It is assumed the array is non-zero length.
     *
     * @param x Input array.
     * @param name Name of the array.
     * @return a reference to the input (sorted) array
     * @throws IllegalArgumentException if {@code x} contains NaN values.
     */
    private static double[] sort(double[] x, String name) {
        Arrays.sort(x);
        // NaN will be at the end
        if (Double.isNaN(x[x.length - 1])) {
            throw new InferenceException(name + " contains NaN");
        }
        return x;
    }

    /**
     * Sets the {@code sign} to the specified {@code value}.
     *
     * @param sign Sign (can be null).
     * @param value Value
     */
    private static void setSign(int[] sign, int value) {
        if (sign != null) {
            sign[0] = value;
        }
    }
}
