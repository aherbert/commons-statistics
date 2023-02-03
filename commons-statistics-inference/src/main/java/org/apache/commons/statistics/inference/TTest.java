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

import java.util.Objects;
import org.apache.commons.statistics.distribution.TDistribution;

/**
 * Implements Student's t-test statistics.
 *
 * <p>Tests can be:
 * <ul>
 * <li>One-sample or two-sample
 * <li>One-sided or two-sided
 * <li>Paired or unpaired (for two-sample tests)
 * <li>Homoscedastic (equal variance assumption) or heteroscedastic (for two sample tests)
 * <li>Fixed significance level (boolean-valued) or returning p-values
 * </ul>
 *
 * <p>Test statistics are available for all tests. Methods including "Test" in their
 * names perform tests, all other methods return t-statistics. Among the "Test" methods,
 * {@code double-}valued methods return p-values; {@code boolean-}valued methods
 * perform fixed significance level tests. Significance levels are always specified as
 * numbers between 0 and 0.5 (e.g. tests at the 95% level use {@code alpha=0.05}).
 *
 * <p>Input to tests can be either {@code double[]} arrays or the mean, variance, and size
 * of the sample.
 *
 * <p>Uses {@link org.apache.commons.statistics.distribution.TDistribution} to estimate
 * exact p-values.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Student%27s_t-test">Student&#39;s t-test (Wikipedia)</a>
 * @since 1.1
 */
public final class TTest {
    /**
     * Options for the t-test.
     *
     * <p>This class is immutable.
     */
    public static class Options {
        /** Default options. */
        private static final Options DEFAULT_OPTIONS = new Options();

        /** Alternative hypothesis. */
        private final AlternativeHypothesis alternative;
        /** Assume the two samples have the same population variance. */
        private final boolean equalVariances;
        /** The true value of the mean (or difference in means for a two sample test). */
        private final double mu;

        /**
         * Builder for the {@link Options}.
         */
        public static class Builder {
            /** Alternative hypothesis. */
            private AlternativeHypothesis alternative;
            /** Assume the two samples have the same population variance. */
            private boolean equalVariances;
            /** The true value of the mean (or difference in means for a two sample test). */
            private double mu;

            /**
             * @param source Source to copy.
             */
            Builder(Options source) {
                alternative = source.alternative;
                equalVariances = source.equalVariances;
                mu = source.mu;
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
             * Set the assumption of equal variances.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#isEqualVariances()
             */
            public Builder setEqualVariances(boolean v) {
                equalVariances = v;
                return this;
            }

            /**
             * Set the expected value of the mean (or difference in means for a two-sample test).
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#getMu()
             */
            public Builder setMu(double v) {
                mu = v;
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
            equalVariances = false;
            mu = 0;
        }

        /**
         * @param source Source to copy.
         */
        Options(Builder source) {
            alternative = source.alternative;
            equalVariances = source.equalVariances;
            mu = source.mu;
        }

        /**
         * Return the default options.
         *
         * <ul>
         * <li>{@link #getAlternative getAlternative = two-sided}
         * <li>{@link #isEqualVariances() isEqualVariances = false}
         * <li>{@link #getMu() getMu = 0}
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
         * If {@code true}, perform the independent t-test under the assumption of equal
         * sub-population variances (homoscedastic t-test).
         *
         * <p>If {@code false}, perform the independent t-test without the assumption of equal
         * sub-population variances (heteroscedastic t-test).
         *
         * <p>Applies to {@link TTest#test(double[], double[], Options)}.
         *
         * @return true the variance are equal
         */
        public boolean isEqualVariances() {
            return equalVariances;
        }

        /**
         * Return the expected value of the mean (or difference in means for a two-sample test).
         *
         * @return the expected mean
         */
        public double getMu() {
            return mu;
        }
    }

    /**
     * Result for the t-test.
     *
     * <p>This class is immutable.
     */
    public static final class Result extends BaseSignificanceResult {
        /** Degrees of freedom. */
        private final double degreesOfFreedom;

        /**
         * Create an instance.
         *
         * @param statistic Test statistic.
         * @param degreesOfFreedom Degrees of freedom.
         * @param p Result p-value.
         */
        Result(double statistic, double degreesOfFreedom, double p) {
            super(statistic, p);
            this.degreesOfFreedom = degreesOfFreedom;
        }

        /**
         * Gets the degrees of freedom.
         *
         * @return the degrees of freedom
         */
        public double getDegreesOfFreedom() {
            return degreesOfFreedom;
        }
    }

    /** No instances. */
    private TTest() {}

    /**
     * Computes a one-sample t statistic comparing the mean of the dataset to {@code mu}.
     *
     * <p>The returned t-statistic is:
     *
     * <p>\[ t = \frac{m - \mu}{ \sqrt{ \frac{v}{n} } \]
     *
     * @param mu Expected mean.
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variance is negative
     */
    public static double statistic(double mu, double m, double v, long n) {
        InferenceUtils.checkNonNegative(v);
        checkSampleSize(n);
        return computeT(m - mu, v, n);
    }

    /**
     * Computes a one-sample t statistic comparing the mean of the sample to {@code mu}.
     *
     * @param mu Expected mean.
     * @param x Sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}
     * @see #statistic(double, double, double, long)
     */
    public static double statistic(double mu, double[] x) {
        final long n = checkSampleSize(x.length);
        final double m = StatisticUtils.mean(x);
        final double v = StatisticUtils.variance(x, m);
        return computeT(m - mu, v, n);
    }

    /**
     * Computes a paired two-sample t-statistic comparing the mean difference between the
     * samples to {@code mu}.
     *
     * <p>The t-statistic returned is equivalent to what would be returned by computing
     * the one-sample t-statistic {@link #statistic(double, double[])}, with
     * the sample array consisting of the (signed) differences between corresponding
     * entries in {@code x} and {@code y}.
     *
     * @param mu Expected mean difference.
     * @param x First sample values.
     * @param y Second sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     */
    public static double pairedStatistic(double mu, double[] x, double[] y) {
        final long n = checkSampleSize(x.length);
        // Update all x with mu
        final double[] xx = StatisticUtils.subtract(x, mu);
        final double m = StatisticUtils.meanDifference(xx, y);
        final double v = StatisticUtils.varianceDifference(xx, y, m);
        return computeT(m, v, n);
    }

    /**
     * Computes a two-sample t statistic, comparing the difference in means of the
     * datasets to {@code mu}.
     *
     * <p>Use the {@code equalVariances} flag to control the computation of the variance.
     * Use {@code false} to compare the means without the assumption of equal
     * sub-population variances (heteroscedastic); otherwise the means are compared
     * under the assumption of equal sub-population variances (homoscedastic).
     *
     * <p>The heteroscedastic t-statistic is:
     *
     * <p>\[ t = \frac{m1 - m2 - \mu}{ \sqrt{ \frac{v_1}{n_1} + \frac{v_2}{n_2} } \]
     *
     * <p>The homoscedastic t-statistic is:
     *
     * <p>\[ t = \frac{m1 - m2 - \mu}{ \sqrt{v} \sqrt{ \frac{1}{n_1} + \frac{1}{n_2} } } \]
     *
     * <p>where \( v \) is the pooled variance estimate:
     *
     * <p>\[ v = \frac{(n_1-1)v_1 + (n_2-1)v_2}{n_1 + n_2 - 2} \]
     *
     * @param mu Expected difference between means.
     * @param m1 First sample mean.
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param m2 Second sample mean.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @param equalVariances Set to {@code true} to assume equal variances.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     */
    public static double statistic(double mu,
                                   double m1, double v1, long n1,
                                   double m2, double v2, long n2,
                                   boolean equalVariances) {
        InferenceUtils.checkNonNegative(v1);
        InferenceUtils.checkNonNegative(v2);
        checkSampleSize(n1);
        checkSampleSize(n2);
        return equalVariances ?
            computeHomoscedasticT(m1 - mu, v1, n1, m2, v2, n2) :
            computeT(m1 - mu, v1, n1, m2, v2, n2);
    }

    /**
     * Computes a two-sample t statistic, comparing the difference in means of the
     * samples to {@code mu}.
     *
     * <p>Use the {@code equalVariances} flag to control the computation of the variance.
     * Use {@code false} to compare the means without the assumption of equal
     * sub-population variances (heteroscedastic); otherwise the means are compared
     * under the assumption of equal sub-population variances (homoscedastic).
     *
     * @param mu Expected difference between means.
     * @param x First sample values.
     * @param y Second sample values.
     * @param equalVariances Set to {@code true} to assume equal variances.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples in either dataset is {@code < 2}
     */
    public static double statistic(double mu, double[] x, double[] y, boolean equalVariances) {
        final long n1 = checkSampleSize(x.length);
        final long n2 = checkSampleSize(y.length);
        final double[] s1 = StatisticUtils.subtract(x, mu);
        final double m1 = StatisticUtils.mean(s1);
        final double m2 = StatisticUtils.mean(y);
        final double v1 = StatisticUtils.variance(s1, m1);
        final double v2 = StatisticUtils.variance(y, m2);
        return equalVariances ?
            computeHomoscedasticT(m1, v1, n1, m2, v2, n2) :
            computeT(m1, v1, n1, m2, v2, n2);
    }

    /**
     * Perform a two-sided one-sample t-test comparing the mean of the dataset to {@code 0}.
     *
     * <p>Degrees of freedom are \( v = n - 1).
     *
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variance is negative
     * @see #statistic(double, double, double, long)
     * @see #test(double, double, long, Options)
     */
    public static Result test(double m, double v, long n) {
        return test(m, v, n, Options.defaults());
    }

    /**
     * Perform a one-sample t-test comparing the mean of the dataset to {@code mu}.
     *
     * <p>Degrees of freedom are \( v = n - 1 \).
     *
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variance is negative
     * @see #statistic(double, double, double, long)
     */
    public static Result test(double m, double v, long n, Options options) {
        final double t = statistic(options.getMu(), m, v, n);
        final double df = n - 1.0;
        final double p = computeP(t, df, options.getAlternative());
        return new Result(t, df, p);
    }

    /**
     * Performs a two-sided one-sample t-test comparing the mean of the sample to {@code 0}.
     *
     * <p>Degrees of freedom are \( v = n - 1 \).
     *
     * @param sample Sample values.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     * @see #statistic(double, double[])
     * @see #test(double[], Options)
     */
    public static Result test(double[] sample) {
        return test(sample, Options.defaults());
    }

    /**
     * Performs a one-sample t-test comparing the mean of the sample to {@code mu}.
     *
     * <p>Degrees of freedom are \( v = n - 1 \).
     *
     * @param sample Sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     * @see #statistic(double, double[])
     */
    public static Result test(double[] sample, Options options) {
        final double t = statistic(options.getMu(), sample);
        final double df = sample.length - 1.0;
        final double p = computeP(t, df, options.getAlternative());
        return new Result(t, df, p);
    }

    /**
     * Performs a two-sided paired two-sample t-test comparing the mean difference between the
     * samples to {@code 0}.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     * @see #pairedStatistic(double, double[], double[])
     * @see #pairedTest(double[], double[], Options)
     */
    public static Result pairedTest(double[] x, double[] y) {
        return pairedTest(x, y, Options.defaults());
    }

    /**
     * Performs a paired two-sample t-test comparing the mean difference between the
     * samples to {@code mu}.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     * @see #pairedStatistic(double, double[], double[])
     */
    public static Result pairedTest(double[] x, double[] y, Options options) {
        final double t = pairedStatistic(options.getMu(), x, y);
        final double df = x.length - 1.0;
        final double p = computeP(t, df, options.getAlternative());
        return new Result(t, df, p);
    }

    /**
     * Performs a two-sided two-sample t-test, comparing the difference in means of the
     * datasets to {@code 0} without the assumption of equal sub-population variances.
     *
     * @param m1 First sample mean.
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param m2 Second sample mean.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @return test result
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     * @see #statistic(double, double, double, long, double, double, long, boolean)
     * @see #test(double, double, long, double, double, long, Options)
     */
    public static Result test(double m1, double v1, long n1,
                              double m2, double v2, long n2) {
        return test(m1, v1, n1, m2, v2, n2, Options.defaults());
    }

    /**
     * Performs a two-sample t-test, comparing the difference in means of the
     * datasets to {@code mu}.
     *
     * <p>Use the {@code options} to select a homoscedastic test under the assumption of equal
     * sub-population variances; or a heteroscedastic test without the
     * assumption of equal sub-population variances.
     *
     * <p>The heteroscedastic degrees of freedom are estimated using the
     * Welch-Satterthwaite approximation:
     *
     * <p>\[ v = \frac{ (\frac{v_1}{n_1} + \frac{v_2}{n_2})^2 }
     *                { \frac{(v_1/n_1)^2}{n_1-1} + \frac{(v_2/n_2)^2}{n_2-1} } \]
     *
     * <p>The homoscedastic degrees of freedom are \( v = n_1 + n_2 - 2).
     *
     * @param m1 First sample mean.
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param m2 Second sample mean.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     * @see #statistic(double, double, double, long, double, double, long, boolean)
     */
    public static Result test(double m1, double v1, long n1,
                              double m2, double v2, long n2,
                              Options options) {
        final double t = statistic(options.getMu(), m1, v1, n1, m2, v2, n2, options.isEqualVariances());
        final double df = options.isEqualVariances() ?
                -2.0 + n1 + n2 :
                computeDf(v1, n1, v2, n2);
        final double p = computeP(t, df, options.getAlternative());
        return new Result(t, df, p);
    }

    /**
     * Performs a two-sided two-sample t-test on two independent samples comparing the difference
     * in means of the datasets to {@code 0} without the assumption of equal sub-population
     * variances.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples in either dataset
     * is {@code < 2}
     * @see #statistic(double, double[], double[], boolean)
     * @see #test(double[], double[], Options)
     */
    public static Result test(double[] x, double[] y) {
        return test(x, y, Options.defaults());
    }

    /**
     * Performs a two-sample t-test on two independent samples comparing the difference in means of
     * the samples to {@code mu}.
     *
     * <p>Use the {@code options} to select a homoscedastic test under the assumption of equal
     * sub-population variances; or a heteroscedastic test without the
     * assumption of equal samples sizes or sub-population variances.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples in either dataset
     * is {@code < 2}
     * @see #statistic(double, double[], double[], boolean)
     * @see #test(double, double, long, double, double, long, Options)
     */
    public static Result test(double[] x, double[] y, Options options) {
        return test(x, y, options.getAlternative(),
            options.isEqualVariances() ? DataDispersion.HOMOSCEDASTIC : DataDispersion.HETEROSCEDASTIC,
            Difference.of(options.getMu()));
    }

    /**
     * Performs a two-sample t-test on two independent samples comparing the difference in means of
     * the samples to {@code mu}.
     *
     * <p>Use the {@code options} to select a homoscedastic test under the assumption of equal
     * sub-population variances; or a heteroscedastic test without the
     * assumption of equal samples sizes or sub-population variances.
     *
     * <p>Supports the following options (with defaults):
     *
     * <ul>
     * <li>{@link AlternativeHypothesis AlternativeHypothesis = two-sided}
     * <li>{@link DataDispersion DataDispersion = heteroscedastic}
     * <li>{@link Difference Difference mu = 0}
     * </ul>
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples in either dataset
     * is {@code < 2}
     * @see #statistic(double, double[], double[], boolean)
     * @see #test(double, double, long, double, double, long, Options)
     */
    public static Result test(double[] x, double[] y, Object... options) {
        // Here we do not call statistic(double[], double[], boolean) because the degreesOfFreedom
        // requires the variance. So repeat the computation and compute p.
        final long n1 = checkSampleSize(x.length);
        final long n2 = checkSampleSize(y.length);
        final double mu = Options2.orElse(Difference.ZERO, options).doubleValue();
        final double[] xx = StatisticUtils.subtract(x, mu);
        final double m1 = StatisticUtils.mean(xx);
        final double m2 = StatisticUtils.mean(y);
        final double v1 = StatisticUtils.variance(xx, m1);
        final double v2 = StatisticUtils.variance(y, m2);
        double t;
        double df;
        if (Options2.isPresent(DataDispersion.HOMOSCEDASTIC, options)) {
            t = computeHomoscedasticT(m1, v1, n1, m2, v2, n2);
            df = -2.0 + n1 + n2;
        } else {
            t = computeT(m1, v1, n1, m2, v2, n2);
            df = computeDf(v1, n1, v2, n2);
        }
        final AlternativeHypothesis alternative = Options2.orElse(AlternativeHypothesis.TWO_SIDED, options);
        final double p = computeP(t, df, alternative);
        return new Result(t, df, p);
    }

    /**
     * Computes t statistic for one-sample t-test.
     *
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return t test statistic
     */
    private static double computeT(double m, double v, long n) {
        return m / Math.sqrt(v / n);
    }

    /**
     * Computes t statistic for two-sample t-test without the assumption of equal
     * samples sizes or sub-population variances.
     *
     * @param m1 First sample mean.
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param m2 Second sample mean.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @return t test statistic
     */
    private static double computeT(double m1, double v1, long n1,
                                   double m2, double v2, long n2)  {
        return (m1 - m2) / Math.sqrt((v1 / n1) + (v2 / n2));
    }

    /**
     * Computes approximate degrees of freedom for two-sample t-test without the
     * assumption of equal samples sizes or sub-population variances.
     *
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @return approximate degrees of freedom
     */
    private static double computeDf(double v1, long n1,
                                    double v2, long n2) {
        // Sample sizes are specified as a double to avoid integer overflow
        final double d1 = n1;
        final double d2 = n2;
        return (((v1 / d1) + (v2 / d2)) * ((v1 / d1) + (v2 / d2))) /
            ((v1 * v1) / (d1 * d1 * (n1 - 1)) + (v2 * v2) / (d2 * d2 * (n2 - 1)));
    }

    /**
     * Computes t statistic for two-sample t-test under the hypothesis of equal
     * sub-population variances.
     *
     * @param m1 First sample mean.
     * @param v1 First sample variance.
     * @param n1 First sample size.
     * @param m2 Second sample mean.
     * @param v2 Second sample variance.
     * @param n2 Second sample size.
     * @return t test statistic
     */
    private static double computeHomoscedasticT(double m1, double v1, long n1,
                                                double m2, double v2, long n2)  {
        final double pooledVariance = ((n1 - 1) * v1 + (n2 - 1) * v2) / (-2.0 + n1 + n2);
        return (m1 - m2) / Math.sqrt(pooledVariance * (1.0 / n1 + 1.0 / n2));
    }

    /**
     * Computes p-value for the specified t statistic.
     *
     * @param t T statistic.
     * @param degreesOfFreedom Degrees of freedom.
     * @param alternative Alternative hypothesis.
     * @return p-value for t-test
     */
    private static double computeP(double t, double degreesOfFreedom, AlternativeHypothesis alternative) {
        if (alternative == AlternativeHypothesis.LESS_THAN) {
            return TDistribution.of(degreesOfFreedom).cumulativeProbability(t);
        }
        if (alternative == AlternativeHypothesis.GREATER_THAN) {
            return TDistribution.of(degreesOfFreedom).survivalProbability(t);
        }
        // two-sided
        return 2.0 * TDistribution.of(degreesOfFreedom).survivalProbability(Math.abs(t));
    }

    /**
     * Check sample data size.
     *
     * @param n Data size.
     * @return the sample size
     * @throws IllegalArgumentException if the number of samples {@code < 2}
     */
    private static long checkSampleSize(long n) {
        if (n <= 1) {
            throw new InferenceException(InferenceException.TWO_VALUES_REQUIRED, n);
        }
        return n;
    }
}
