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
public class TTest {
    /**
     * Options for the t-test.
     *
     * <p>This class is immutable.
     */
    public static class Options {
        /** Default options. */
        private static final Options DEFAULTS = new Options();

        /** Alternative hypothesis. */
        private final AlternativeHypothesis alternative;
        /** Assume the two samples have the same population variance. */
        private final boolean equalVariances;

        /**
         * Builder for the {@link Options}.
         */
        public static class Builder {
            /** Alternative hypothesis. */
            private AlternativeHypothesis alternative;
            /** Assume the two samples have the same population variance. */
            private boolean equalVariances;

            /**
             * @param source Source to copy.
             */
            Builder(Options source) {
                alternative = source.alternative;
                equalVariances = source.equalVariances;
            }

            /**
             * Sets the alternative hypothesis.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#getAlternative()
             */
            public Builder setAlternative(AlternativeHypothesis v) {
                this.alternative = v;
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
        }

        /**
         * @param source Source to copy.
         */
        Options(Builder source) {
            alternative = source.alternative;
            equalVariances = source.equalVariances;
        }

        /**
         * Return the default options.
         *
         * @return the options
         */
        public static Options defaults() {
            return DEFAULTS;
        }

        /**
         * Create a new {@link Builder} with the default options.
         *
         * @return the builder
         */
        public static Builder builder() {
            return DEFAULTS.toBuilder();
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
         * <p>Applies to the {@link TTest#tTest(double[], double[])}.
         *
         * @return true the variance are equal
         */
        public boolean isEqualVariances() {
            return equalVariances;
        }
    }

    /**
     * Performs a two-sample t-test on two independent samples.
     *
     * <p>Use the {@code options} to select a homoscedastic test under the assumption of equal
     * sub-population variances; or a heteroscedastic test without the
     * assumption of equal samples sizes or sub-population variances.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples in either dataset
     * is {@code < 2}
     */
    public SignificanceResult independent(double[] sample1, double[] sample2, Options options) {
        // Here we do not call t(double[], double[]) because the degreesOfFreedom
        // requires the variance. So repeat the computation and compute p.
        final long n1 = checkSampleSize(sample1.length);
        final long n2 = checkSampleSize(sample2.length);
        final double m1 = InferenceUtils.mean(sample1);
        final double m2 = InferenceUtils.mean(sample2);
        final double v1 = InferenceUtils.variance(sample1, m1);
        final double v2 = InferenceUtils.variance(sample2, m2);
        double t;
        double df;
        if (options.isEqualVariances()) {
            t = computeHomoscedasticT(m1, m2, v1, v2, n1, n2);
            df = -2.0 + n1 + n2;
        } else {
            t = computeT(m1, m2, v1, v2, n1, n2);
            df = computeDf(v1, v2, n1, n2);
        }
        double p = computeP(t, df);
        return new BaseSignificanceResult(t, p);
    }

    /**
     * Performs a two-sample t-test on two related, or paired, samples.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     */
    public SignificanceResult paired(double[] sample1, double[] sample2, Options options) {
        final long n = checkSampleSize(sample1.length);
        final double m = InferenceUtils.meanDifference(sample1, sample2);
        final double v = InferenceUtils.varianceDifference(sample1, sample2, m);
        double t = computeT(0, m, v, n);
        double df = n - 1;
        double p = computeP(t, df);
        return new BaseSignificanceResult(t, p);
    }

    /**
     * Performs a one-sample t-test comparing the mean of the sample with the constant
     * {@code mu}.
     *
     * @param mu Constant value to compare sample mean against.
     * @param sample Sample values.
     * @param options Test options.
     * @return the test result
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     */
    public SignificanceResult one(double mu, double[] sample, Options options) {
        final long n = checkSampleSize(sample.length);
        final double m = InferenceUtils.mean(sample);
        final double v = InferenceUtils.variance(sample, m);
        double t = computeT(0, m, v, n);
        double df = n - 1;
        double p = computeP(t, df);
        return new BaseSignificanceResult(t, p);
    }

    /**
     * Computes a 2-sample t statistic, comparing the means of the datasets without the
     * assumption of equal sample sizes or sub-population variances. Use
     * {@link #homoscedasticT(double, double, double, double, long, long)} to compute a
     * t-statistic under the equal variances assumption.
     *
     * <p>This statistic can be used to perform a two-sample t-test to compare sample
     * means.
     *
     * <p>The returned t-statistic is:
     *
     * <p>\[ t = \frac{m1 - m2}{ \sqrt{ \frac{v_1}{n_1} + \frac{v_2}{n_2} } \]
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean..
     * @param v1 First sample variance..
     * @param v2 Second sample variance..
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return t statistic
     *
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     */
    public double t(double m1, double m2,
                    double v1, double v2,
                    long n1, long n2) {
        InferenceUtils.checkNonNegative(v1);
        InferenceUtils.checkNonNegative(v2);
        checkSampleSize(n1);
        checkSampleSize(n2);
        return computeT(m1, m2, v1, v2, n1, n2);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a two-sample, two-tailed t-test comparing the means of the datasets without the
     * assumption of equal samples sizes or sub-population variances. Use
     * {@link #homoscedasticTTest(double, double, double, double, long, long)} to perform a
     * t-test under the equal variances assumption.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the two means are equal in favor of the two-sided
     * alternative that they are different. For a one-sided test, divide the returned
     * value by 2.
     *
     * <p>See {@link #t(double, double, double, double, long, long)} for the formula used to
     * compute the t-statistic. Degrees of freedom are approximated using the
     * Welch-Satterthwaite approximation:
     *
     * <p>\[ v = \frac{ (\frac{v_1}{n_1} + \frac{v_2}{n_2})^2 }
     *                { \frac{(v_1/n_1)^2}{n_1-1} + \frac{(v_2/n_2)^2}{n_2-1} } \]
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return p-value for t-test
     *
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     * @see #t(double, double, double, double, long, long)
     */
    public double tTest(double m1, double m2,
                        double v1, double v2,
                        long n1, long n2) {
        final double degreesOfFreedom = computeDf(v1, v2, n1, n2);
        return computeP(t(m1, m2, v1, v2, n1, n2), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that samples 1 and 2
     * describe datasets drawn from populations with the same mean, with significance
     * level {@code alpha}. This test does not assume that the samples sizes or
     * sub-population variances are equal. To perform the test under the equal variances
     * assumption, use
     * {@link #homoscedasticTTest(double, double, double, double, long, long, double)}.
     *
     * <p>Returns {@code true} iff the null hypothesis that the means are equal can be
     * rejected with confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * <p><strong>Examples:</strong>
     * <ol>
     * <li>To test the (2-sided) hypothesis {@code mean 1 = mean 2 } at the 95% level, use<br>
     *     {@code tTest(m1, m2, v1, v2, n1, n2, 0.05)}
     * <li>To test the (one-sided) hypothesis {@code m1 < m2 } at the 99% level, first
     *     verify that the measured mean of {@code sample 1} is less than the mean of
     *     {@code sample 2} and then use<br>
     *     {@code tTest(m1, m2, v1, v2, n1, n2, 0.02)}
     * </ol>
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; the variances are negative; or {@code alpha} is not in the range
     * {@code (0, 0.5]}
     * @see #tTest(double, double, double, double, long, long)
     */
    public boolean tTest(double m1, double m2,
                         double v1, double v2,
                         long n1, long n2,
                         double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return tTest(m1, m2, v1, v2, n1, n2) < alpha;
    }

    /**
     * Computes a t statistic to use in comparing the mean of the dataset to {@code mu}.
     *
     * <p>This statistic can be used to perform a one sample t-test for the mean.
     *
     * <p>The returned t-statistic is:
     *
     * <p>\[ t = \frac{m - \mu}{ \sqrt{ \frac{v}{n} } \]
     *
     * @param mu comparison constant
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variance is negative
     */
    public double t(double mu, double m, double v, long n) {
        InferenceUtils.checkNonNegative(v);
        checkSampleSize(n);
        return computeT(mu, m, v, n);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a one-sample, two-tailed t-test comparing the mean of the dataset with the constant
     * {@code mu}.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the mean equals {@code mu} in favor of the two-sided
     * alternative that the mean is different from {@code mu}. For a one-sided test,
     * divide the returned value by 2.
     *
     * <p>See {@link #t(double, double, double, long)} for the formula used to
     * compute the t-statistic. Degrees of freedom are \( v = n - 1).
     *
     * @param mu Constant value to compare sample mean against.
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variance is negative
     * @see #t(double, double, double, long)
     */
    public double tTest(double mu, double m, double v, long n) {
        final double degreesOfFreedom = n - 1.0;
        return computeP(t(mu, m, v, n), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that the mean of the
     * population from which the dataset is drawn equals {@code mu}.
     *
     * <p>Returns {@code true} iff the null hypothesis can be rejected with
     * confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * <p><strong>Examples:</strong>
     * <ol>
     * <li>To test the (2-sided) hypothesis {@code sample mean = mu } at
     *     the 95% level, use<br>
     *     {@code tTest(mu, m, v, n, 0.05)}
     * <li>To test the (one-sided) hypothesis {@code  sample mean &lt; mu }
     *     at the 99% level, first verify that the measured sample mean is less
     *     than {@code mu} and then use<br>
     *     {@code tTest(mu, m, v, n, 0.02)}
     * </ol>
     *
     * @param mu Constant value to compare sample mean against.
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; the
     * variance is negative; or {@code alpha} is not in the range {@code (0, 0.5]}
     * @see #tTest(double, double, double, long)
     */
    public boolean tTest(double mu, double m, double v, long n,
                         double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return tTest(mu, m, v, n) < alpha;
    }

    /**
     * Computes a 2-sample t statistic, comparing the means of the datasets under the
     * assumption of equal sub-population variances. Samples sizes can be equal or
     * unequal. To compute a t-statistic without the equal variances assumption, use
     * {@link #t(double, double, double, double, long, long)}.
     *
     * <p>This statistic can be used to perform a (homoscedastic) two-sample t-test to
     * compare sample means.
     *
     * <p>The t-statistic returned is:
     *
     * <p>\[ t = \frac{m1 - m2}{ \sqrt{v} \sqrt{ \frac{1}{n_1} + \frac{1}{n_2} } } \]
     *
     * <p>where \( v \) is the pooled variance estimate:
     *
     * <p>\[ v = \sqrt{ \frac{(n_1-1)v_1 + (n_2-1)v_2}{n_1 + n_2 - 2} }\]
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * variances are negative
     */
    public double homoscedasticT(double m1, double m2,
                                 double v1, double v2,
                                 long n1, long n2) {
        InferenceUtils.checkNonNegative(v1);
        InferenceUtils.checkNonNegative(v2);
        checkSampleSize(n1);
        checkSampleSize(n2);
        return computeHomoscedasticT(m1, m2, v1, v2, n1, n2);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a two-sample, two-tailed t-test comparing the means of the datasets, under the
     * hypothesis of equal sub-population variances. Samples sizes can be equal or
     * unequal. To perform a test without the equal variances assumption, use
     * {@link #tTest(double, double, double, double, long, long)}.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the two means are equal in favor of the two-sided
     * alternative that they are different. For a one-sided test, divide the returned
     * value by 2.
     *
     * <p>See {@link #homoscedasticT(double, double, double, double, long, long)} for the
     * formula used to compute the t-statistic. Degrees of freedom are \( v = n_1 + n_2 -
     * 2).
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; or the variances are negative.
     * @see #homoscedasticT(double, double, double, double, long, long)
     */
    public double homoscedasticTTest(double m1, double m2,
                                     double v1, double v2,
                                     long n1, long n2) {
        final double degreesOfFreedom = -2.0 + n1 + n2;
        return computeP(homoscedasticT(m1, m2, v1, v2, n1, n2), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that {@code sample1} and
     * {@code sample2} are drawn from populations with the same mean, with significance
     * level {@code alpha}, assuming that the sub-population variances are equal. Samples
     * sizes can be equal or unequal. Use
     * {@link #tTest(double, double, double, double, long, long, double)} to perform the
     * test without the assumption of equal variances.
     *
     * <p>Returns {@code true} iff the null hypothesis that the means are equal can be
     * rejected with confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * <p><strong>Examples:</strong>
     * <ol>
     * <li>To test the (2-sided) hypothesis {@code mean 1 = mean 2 } at
     *     the 95% level, use <br>
     *     {@code tTest(sample1, sample2, 0.05). }
     * <li>To test the (one-sided) hypothesis {@code  mean 1 &lt; mean 2, }
     *     at the 99% level, first verify that the measured mean of
     *     {@code sample 1} is less than the mean of {@code sample 2}
     *     and then use<br>
     *     {@code tTest(sample1, sample2, 0.02) }
     * </ol>
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples in either dataset is
     * {@code < 2}; the variances are negative; or {@code alpha} is not in the range
     * {@code (0, 0.5]}
     * @see #homoscedasticT(double, double, double, double, long, long)
     */
    public boolean homoscedasticTTest(double m1, double m2,
                                      double v1, double v2,
                                      long n1, long n2,
                                      double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return homoscedasticTTest(m1, m2, v1, v2, n1, n2) < alpha;
    }

    /**
     * Computes a 2-sample t statistic, comparing the means of the datasets without the
     * assumption of equal sample sizes or sub-population variances. Use
     * {@link #homoscedasticT(double[], double[])} to compute a
     * t-statistic under the equal variances assumption.
     *
     * <p>This statistic can be used to perform a two-sample t-test to compare
     * sample means.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples in either dataset is {@code < 2}
     * @see #t(double, double, double, double, long, long)
     */
    public double t(double[] sample1, double[] sample2) {
        final long n1 = checkSampleSize(sample1.length);
        final long n2 = checkSampleSize(sample2.length);
        final double m1 = InferenceUtils.mean(sample1);
        final double m2 = InferenceUtils.mean(sample2);
        final double v1 = InferenceUtils.variance(sample1, m1);
        final double v2 = InferenceUtils.variance(sample2, m2);
        return computeT(m1, m2, v1, v2, n1, n2);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a two-sample, two-tailed t-test comparing the means of the datasets without the
     * assumption of equal samples sizes or sub-population variances. Use
     * {@link #homoscedasticTTest(double[], double[])} to perform a
     * t-test under the equal variances assumption.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the two means are equal in favor of the two-sided
     * alternative that they are different. For a one-sided test, divide the returned
     * value by 2.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples in either dataset is {@code < 2}
     * @see #t(double[], double[])
     * @see #tTest(double, double, double, double, long, long)
     */
    public double tTest(double[] sample1, double[] sample2) {
        // Here we do not call t(double[], double[]) because the degreesOfFreedom
        // requires the variance. So repeat the computation and compute p.
        final long n1 = checkSampleSize(sample1.length);
        final long n2 = checkSampleSize(sample2.length);
        final double m1 = InferenceUtils.mean(sample1);
        final double m2 = InferenceUtils.mean(sample2);
        final double v1 = InferenceUtils.variance(sample1, m1);
        final double v2 = InferenceUtils.variance(sample2, m2);
        final double degreesOfFreedom = computeDf(v1, v2, n1, n2);
        return computeP(computeT(m1, m2, v1, v2, n1, n2), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that samples 1 and 2
     * describe datasets drawn from populations with the same mean, with significance
     * level {@code alpha}. This test does not assume that the samples sizes or
     * sub-population variances are equal. To perform the test under the equal variances
     * assumption, use {@link #homoscedasticTTest(double[], double[], double)}.
     *
     * <p>Returns {@code true} iff the null hypothesis that the means are equal can be
     * rejected with confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * <p><strong>Examples:</strong>
     * <ol>
     * <li>To test the (2-sided) hypothesis {@code mean 1 = mean 2 } at the 95% level, use<br>
     *     {@code tTest(sample1, sample2, 0.05)}
     * <li>To test the (one-sided) hypothesis {@code m1 < m2 } at the 99% level, first
     *     verify that the measured mean of {@code sample1} is less than the mean of
     *     {@code sample2} and then use<br>
     *     {@code tTest(sample1, sample2, 0.02)}
     * </ol>
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples in either dataset is {@code < 2},
     * or {@code alpha} is not in the range {@code (0, 0.5]}
     * @see #tTest(double[], double[])
     * @see #tTest(double, double, double, double, long, long, double)
     */
    public boolean tTest(double[] sample1, double[] sample2,
                         double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return tTest(sample1, sample2) < alpha;
    }

    /**
     * Computes a t statistic to use in comparing the mean of the dataset to {@code mu}.
     *
     * <p>This statistic can be used to perform a one sample t-test for the mean.
     *
     * @param mu comparison constant
     * @param sample Sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}
     * @see #t(double, double, double, long)
     */
    public double t(double mu, double[] sample) {
        final long n = checkSampleSize(sample.length);
        final double m = InferenceUtils.mean(sample);
        final double v = InferenceUtils.variance(sample, m);
        return computeT(mu, m, v, n);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a one-sample, two-tailed t-test comparing the mean of the dataset with the constant
     * {@code mu}.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the mean equals {@code mu} in favor of the two-sided
     * alternative that the mean is different from {@code mu}. For a one-sided test,
     * divide the returned value by 2.
     *
     * @param mu Constant value to compare sample mean against.
     * @param sample Sample values.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples is {@code < 2}
     * @see #t(double, double[])
     * @see #tTest(double, double, double, long)
     */
    public double tTest(double mu, double[] sample) {
        final double degreesOfFreedom = sample.length - 1.0;
        return computeP(t(mu, sample), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that the mean of the
     * population from which the dataset is drawn equals {@code mu}.
     *
     * <p>Returns {@code true} iff the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}. To perform a 1-sided test, use {@code alpha * 2}.
     *
     * <p><strong>Examples:</strong>
     * <ol>
     * <li>To test the (2-sided) hypothesis {@code sample mean = mu } at
     *     the 95% level, use<br>
     *     {@code tTest(mu, sample, 0.05)}
     * <li>To test the (one-sided) hypothesis {@code  sample mean &lt; mu }
     *     at the 99% level, first verify that the measured sample mean is less
     *     than {@code mu} and then use<br>
     *     {@code tTest(mu, sample, 0.02)}
     * </ol>
     *
     * @param mu Constant value to compare sample mean against.
     * @param sample Sample values.
     * @param alpha Significance level of the test.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or
     * {@code alpha} is not in the range {@code (0, 0.5]}
     * @see #tTest(double, double[])
     * @see #tTest(double, double, double, long, double)
     */
    public boolean tTest(double mu, double[] sample, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return tTest(mu, sample) < alpha;
    }

    /**
     * Computes a paired, 2-sample t-statistic.
     *
     * <p>The t-statistic returned is equivalent to what would be returned by computing
     * the one-sample t-statistic {@link #t(double, double[])}, with {@code mu = 0} and
     * the sample array consisting of the (signed) differences between corresponding
     * entries in {@code sample1} and {@code sample2}.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     */
    public double pairedT(double[] sample1, double[] sample2) {
        final long n = checkSampleSize(sample1.length);
        final double m = InferenceUtils.meanDifference(sample1, sample2);
        final double v = InferenceUtils.varianceDifference(sample1, sample2, m);
        return computeT(0, m, v, n);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i> p-value</i>, associated with
     * a paired, two-sample, two-tailed t-test.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the mean of the paired differences is 0 in favor of the
     * two-sided alternative that the mean paired difference is not equal to 0. For a
     * one-sided test, divide the returned value by 2.
     *
     * <p>This test is equivalent to a one-sample t-test computed using
     * {@link #tTest(double, double[])} with {@code mu = 0} and the sample array
     * consisting of the (signed) differences between corresponding elements of
     * {@code sample1} and {@code sample2}.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; or the
     * the size of the samples is not equal
     * @see #pairedT(double[], double[])
     */
    public double pairedTTest(double[] sample1, double[] sample2) {
        final double degreesOfFreedom = sample1.length - 1.0;
        return computeP(pairedT(sample1, sample2), degreesOfFreedom);
    }

    /**
     * Performs a paired t-test evaluating the null hypothesis that the mean of the paired
     * differences between {@code sample1} and {@code sample2} is 0 in favor of the
     * two-sided alternative that the mean paired difference is not equal to 0, with
     * significance level {@code alpha}.
     *
     * <p>Returns {@code true} iff the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}. To perform a 1-sided test, use {@code alpha * 2}
     *
     * <p>This test is equivalent to a one-sample t-test computed using
     * {@link #tTest(double, double[], double)} with {@code mu = 0} and the sample array
     * consisting of the (signed) differences between corresponding elements of
     * {@code sample1} and {@code sample2}.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples is {@code < 2}; the
     * the size of the samples is not equal; or {@code alpha} is not in the range
     * {@code (0, 0.5]}
     * @see #pairedTTest(double[], double[])
     */
    public boolean pairedTTest(double[] sample1, double[] sample2,
                               double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return pairedTTest(sample1, sample2) < alpha;
    }

    /**
     * Computes a 2-sample t statistic, comparing the means of the datasets under the
     * assumption of equal sub-population variances. Samples sizes can be equal or
     * unequal. To compute a t-statistic without the equal variances assumption, use
     * {@link #t(double[], double[])}.
     *
     * <p>This statistic can be used to perform a (homoscedastic) two-sample t-test to
     * compare sample means.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return t statistic
     * @throws IllegalArgumentException if the number of samples {@code < 2}
     * @see #homoscedasticT(double, double, double, double, long, long)
     */
    public double homoscedasticT(double[] sample1, double[] sample2) {
        final long n1 = checkSampleSize(sample1.length);
        final long n2 = checkSampleSize(sample2.length);
        final double m1 = InferenceUtils.mean(sample1);
        final double m2 = InferenceUtils.mean(sample2);
        final double v1 = InferenceUtils.variance(sample1, m1);
        final double v2 = InferenceUtils.variance(sample2, m2);
        return computeHomoscedasticT(m1, m2, v1, v2, n1, n2);
    }

    /**
     * Returns the <i>observed significance level</i>, or <i>p-value</i>, associated with
     * a two-sample, two-tailed t-test comparing the means of the datasets, under the
     * hypothesis of equal sub-population variances. Samples sizes can be equal or
     * unequal. To perform a test without the equal variances assumption, use
     * {@link #tTest(double, double, double, double, long, long)}.
     *
     * <p>The returned p-value is the smallest significance level at which one can reject
     * the null hypothesis that the two means are equal in favor of the two-sided
     * alternative that they are different. For a one-sided test, divide the returned
     * value by 2.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return p-value for t-test
     * @throws IllegalArgumentException if the number of samples {@code < 2}
     * @see #homoscedasticT(double[], double[])
     * @see #homoscedasticTTest(double[], double[])
     */
    public double homoscedasticTTest(double[] sample1, double[] sample2) {
        final double degreesOfFreedom = -2.0 + sample1.length + sample2.length;
        return computeP(homoscedasticT(sample1, sample2), degreesOfFreedom);
    }

    /**
     * Performs a two-sided t-test evaluating the null hypothesis that {@code sample1} and
     * {@code sample2} are drawn from populations with the same mean, with significance
     * level {@code alpha}, assuming that the sub-population variances are equal. Samples
     * sizes can be equal or unequal. Use
     * {@link #tTest(double, double, double, double, long, long, double)} to perform the
     * test without the assumption of equal variances.
     *
     * <p>Returns {@code true} iff the null hypothesis that the means are equal can be
     * rejected with confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence {@code 1 - alpha}
     * @throws IllegalArgumentException if the number of samples {@code < 2}; or
     * {@code alpha} is not in the range {@code (0, 0.5]}
     * @see #homoscedasticTTest(double[], double[])
     * @see #homoscedasticTTest(double, double, double, double, long, long, double)
     */
    public boolean homoscedasticTTest(double[] sample1, double[] sample2,
                                      double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return homoscedasticTTest(sample1, sample2) < alpha;
    }

    /**
     * Computes t test statistic for 1-sample t-test.
     *
     * @param mu Constant to test against.
     * @param m Sample mean.
     * @param v Sample variance.
     * @param n Sample size.
     * @return t test statistic
     */
    private static double computeT(double mu, double m,
                                   double v, long n) {
        return (m - mu) / Math.sqrt(v / n);
    }

    /**
     * Computes t test statistic for 2-sample t-test without the assumption of equal
     * samples sizes or sub-population variances.
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return t test statistic
     */
    private static double computeT(double m1, double m2,
                                   double v1, double v2,
                                   long n1, long n2)  {
        return (m1 - m2) / Math.sqrt((v1 / n1) + (v2 / n2));
    }

    /**
     * Computes approximate degrees of freedom for 2-sample t-test without the assumption
     * of equal samples sizes or sub-population variances.
     *
     * <p>Note: Sample sizes are specified as a double to avoid integer overflow.
     *
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return approximate degrees of freedom
     */
    private static double computeDf(double v1, double v2,
                                    double n1, double n2) {
        return (((v1 / n1) + (v2 / n2)) * ((v1 / n1) + (v2 / n2))) /
            ((v1 * v1) / (n1 * n1 * (n1 - 1)) + (v2 * v2) / (n2 * n2 * (n2 - 1)));
    }

    /**
     * Computes t test statistic for 2-sample t-test under the hypothesis of equal
     * sub-population variances.
     *
     * @param m1 First sample mean.
     * @param m2 Second sample mean.
     * @param v1 First sample variance.
     * @param v2 Second sample variance.
     * @param n1 First sample size.
     * @param n2 Second sample size.
     * @return t test statistic
     */
    private static double computeHomoscedasticT(double m1, double m2,
                                                double v1, double v2,
                                                long n1, long n2)  {
        final double pooledVariance = ((n1 - 1) * v1 + (n2 - 1) * v2) / (-2.0 + n1 + n2);
        return (m1 - m2) / Math.sqrt(pooledVariance * (1.0 / n1 + 1.0 / n2));
    }

    /**
     * Computes p-value for the specified t statistic.
     *
     * @param t T statistic.
     * @param degreesOfFreedom Degrees of freedom.
     * @return p-value for t-test
     */
    private static double computeP(double t, double degreesOfFreedom) {
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
