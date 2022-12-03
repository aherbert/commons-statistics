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

import org.apache.commons.statistics.distribution.ChiSquaredDistribution;

/**
 * Implements Chi-square test statistics.
 *
 * <p>This implementation handles both known and unknown distributions.
 *
 * <p>Two samples tests can be used when the distribution is unknown <i>a priori</i>
 * but provided by one sample, or when the hypothesis under test is that the two
 * samples come from the same underlying distribution.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Chi-squared_test">Chi-square test (Wikipedia)</a>
 * @since 1.1
 */
public final class ChiSquareTest {
    /** Name for the row. */
    private static final String ROW = "row";
    /** Name for the column. */
    private static final String COLUMN = "column";
    /**
     * Options for the Chi-square test.
     *
     * <p>This class is immutable.
     */
    public static class Options {
        /** Default options. */
        private static final Options DEFAULT_OPTIONS = new Options();

        /** Adjustment for the degrees of freedom. */
        private final int adjust;

        /**
         * Builder for the {@link Options}.
         */
        public static class Builder {
            /** Adjustment for the degrees of freedom. */
            private int adjust;

            /**
             * @param source Source to copy.
             */
            Builder(Options source) {
                adjust = source.adjust;
            }

            /**
             * Sets the adjustment to the degrees of freedom.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @throw IllegalArgumentException if the adjustment is negative.
             * @see Options#getDegreesOfFreedomAdjustment()
             */
            public Builder setDegreesOfFreedomAdjustment(int v) {
                InferenceUtils.checkNonNegative(v);
                this.adjust = v;
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
            adjust = 0;
        }

        /**
         * @param source Source to copy.
         */
        Options(Builder source) {
            adjust = source.adjust;
        }

        /**
         * Return the default options.
         *
         * <ul>
         * <li>{@link #getDegreesOfFreedomAdjustment getDegreesOfFreedomAdjustment = 0}
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
         * Return the adjustment to the degrees of freedom.
         *
         * <p>The default degrees of freedom for a sample of length {@code n} are
         * {@code n - 1}. An intrinsic null hypothesis is one where you estimate one or
         * more parameters from the data in order to get the numbers for your null
         * hypothesis. For a distribution with {@code p} parameters where up to
         * {@code p} parameters have been estimated from the data the degrees of freedom
         * is in the range {@code [n - 1 - p, n - 1]}.
         *
         * @return the adjustment
         */
        public int getDegreesOfFreedomAdjustment() {
            return adjust;
        }
    }

    /** No instances. */
    private ChiSquareTest() {}

    /**
     * Computes the Chi-square goodness-of-fit statistic comparing the {@code observed} counts to a
     * uniform expected value (each category is equally likely).
     *
     * <p>Note: This is a specialized version of a comparison of {@code observed}
     * with an {@code expected} array of uniform values. The result is faster than
     * calling {@link #statistic(double[], long[])} and the statistic is the same,
     * with an allowance for accumulated floating-point error due to the optimized
     * routine.
     *
     * @param observed Observed frequency counts.
     * @return Chi-square statistic
     * @throws IllegalArgumentException if the sample size is less than 2;
     * {@code observed} has negative entries; or all the the observations are zero.
     */
    public static double statistic(long[] observed) {
        InferenceUtils.checkValuesRequiredSize(observed.length, 2);
        InferenceUtils.checkNonNegative(observed);
        final double e = StatisticUtils.mean(observed);
        if (e == 0) {
            throw new InferenceException(InferenceException.NO_DATA);
        }
        // chi2 = sum{ (o-e)^2 / e }. Use a single division at the end.
        double chi2 = 0;
        for (final long o : observed) {
            final double d = o - e;
            chi2 += d * d;
        }
        return chi2 / e;
    }

    /**
     * Computes the Chi-square goodness-of-fit statistic comparing {@code observed} and
     * {@code expected} frequency counts.
     *
     * <p><strong>Note:</strong>This implementation rescales the {@code expected}
     * array if necessary to ensure that the sum of the expected and observed counts
     * are equal.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return Chi-square statistic
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; {@code observed} has negative entries; or all the the observations are zero.
     */
    public static double statistic(double[] expected, long[] observed) {
        final double ratio = StatisticUtils.computeRatio(expected, observed);
        // chi2 = sum{ (o-e)^2 / e }
        double chi2 = 0;
        for (int i = 0; i < observed.length; i++) {
            final double e = ratio * expected[i];
            final double d = observed[i] - e;
            chi2 += d * d / e;
        }
        return chi2;
    }

    /**
     * Computes the Chi-square statistic associated with a Chi-square test of
     * independence based on the input {@code counts} array, viewed as a two-way
     * table in row-major format.
     *
     * @param counts 2-way table.
     * @return Chi-square statistic
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     */
    public static double statistic(long[][] counts) {
        InferenceUtils.checkCategoriesRequiredSize(counts.length, 2);
        InferenceUtils.checkValuesRequiredSize(counts[0].length, 2);
        InferenceUtils.checkRectangular(counts);
        InferenceUtils.checkNonNegative(counts);

        final int nRows = counts.length;
        final int nCols = counts[0].length;

        // compute row, column and total sums
        final double[] rowSum = new double[nRows];
        final double[] colSum = new double[nCols];
        double sum = 0;
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                rowSum[row] += counts[row][col];
                colSum[col] += counts[row][col];
            }
            checkNonZero(rowSum[row], ROW, row);
            sum += rowSum[row];
        }

        for (int col = 0; col < nCols; col++) {
            checkNonZero(colSum[col], COLUMN, col);
        }

        // Compute expected counts and Chi-square
        double chi2 = 0;
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                final double e = (rowSum[row] * colSum[col]) / sum;
                final double d = counts[row][col] - e;
                chi2 += d * d / e;
            }
        }
        return chi2;
    }

    /**
     * Computes a Chi-square statistic associated with a Chi-square test of
     * independence of frequency counts in {@code observed1} and {@code observed2}.
     * The sums of frequency counts in the two samples are not required to be the
     * same. The formula used to compute the test statistic is:
     *
     * <p>\[ sum_i{ \frac{(K * a_i - b_i / K)^2}{a_i + b_i} } \]
     *
     * <p>where
     *
     * <p>\[ K = \sqrt{ \sum_i{a_i} / \sum_i{b_i} } \]
     *
     * <p>Note: This is a specialized version of a 2-by-n contingency table. The
     * result is faster than calling {@link #statistic(long[][])} with the table
     * composed as {@code new long[][]{observed1, observed2}}. The statistic is the
     * same, with an allowance for accumulated floating-point error due to the
     * optimized routine.
     *
     * @param observed1 Observed frequency counts of the first data set.
     * @param observed2 Observed frequency counts of the second data set.
     * @return Chi-square statistic
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; either array has entries that are negative; either all
     * counts of {@code observed1} or {@code observed2} are zero; or if the count at
     * some index is zero for both arrays.
     */
    public static double statistic(long[] observed1, long[] observed2) {
        InferenceUtils.checkValuesRequiredSize(observed1.length, 2);
        InferenceUtils.checkValuesSizeMatch(observed1.length, observed2.length);
        InferenceUtils.checkNonNegative(observed1);
        InferenceUtils.checkNonNegative(observed2);

        // Compute and compare count sums
        long colSum1 = 0;
        long colSum2 = 0;
        for (int i = 0; i < observed1.length; i++) {
            final long obs1 = observed1[i];
            final long obs2 = observed2[i];
            checkNonZero(obs1 | obs2, ROW, i);
            colSum1 += obs1;
            colSum2 += obs2;
        }
        // Create the same exception message as chiSquare(long[][])
        checkNonZero(colSum1, COLUMN, 0);
        checkNonZero(colSum2, COLUMN, 1);

        // Compare and compute weight only if different
        final boolean unequalCounts = colSum1 != colSum2;
        final double weight = unequalCounts ?
            Math.sqrt((double) colSum1 / (double) colSum2) : 1;
        // Compute Chi-square
        // This exploits an algebraic rearrangement of the generic n*m contingency table case
        // for a single sum squared addition per row.
        double chi2 = 0;
        for (int i = 0; i < observed1.length; i++) {
            final double obs1 = observed1[i];
            final double obs2 = observed2[i];
            // apply weights
            final double d = unequalCounts ?
                    obs1 / weight - obs2 * weight :
                    obs1 - obs2;
            chi2 += (d * d) / (obs1 + obs2);
        }
        return chi2;
    }

    /**
     * Perform a Chi-square goodness-of-fit test evaluating the null hypothesis that
     * the {@code observed} counts conform to a uniform distribution (each category
     * is equally likely).
     *
     * @param observed Observed frequency counts.
     * @return test result
     * @throws IllegalArgumentException if the sample size is less than 2;
     * {@code observed} has negative entries; or all the the observations are zero.
     */
    public static SignificanceResult test(long[] observed) {
        return test(observed, Options.defaults());
    }

    /**
     * Perform a Chi-square goodness-of-fit test evaluating the null hypothesis that
     * the {@code observed} counts conform to a uniform distribution (each category
     * is equally likely).
     *
     * @param observed Observed frequency counts.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if the sample size is less than 2;
     * {@code observed} has negative entries; all the the observations are zero; or
     * the adjusted degrees of freedom are not strictly positive
     */
    public static SignificanceResult test(long[] observed, Options options) {
        final int df = StatisticUtils.computeDegreesOfFreedom(observed.length, options.getDegreesOfFreedomAdjustment());
        final double chi2 = statistic(observed);
        final double p = computeP(chi2, df);
        return new BaseSignificanceResult(chi2, p);
    }

    /**
     * Perform a Chi-square goodness-of-fit test evaluating the null hypothesis that
     * the {@code observed} counts conform to the {@code expected} counts.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return test result
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; {@code observed} has negative entries; or all the the observations
     * are zero.
     */
    public static SignificanceResult test(double[] expected, long[] observed) {
        return test(expected, observed, Options.defaults());
    }

    /**
     * Perform a Chi-square test evaluating the null hypothesis that the {@code observed}
     * counts conform to the {@code expected} counts.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; {@code observed} has negative entries; all the the observations are zero; or
     * the adjusted degrees of freedom are not strictly positive
     */
    public static SignificanceResult test(double[] expected, long[] observed, Options options) {
        final int df = StatisticUtils.computeDegreesOfFreedom(observed.length, options.getDegreesOfFreedomAdjustment());
        final double chi2 = statistic(expected, observed);
        final double p = computeP(chi2, df);
        return new BaseSignificanceResult(chi2, p);
    }

    /**
     * Perform a Chi-square test of independence based on the input {@code counts} array,
     * viewed as a two-way table.
     *
     * @param counts 2-way table.
     * @return test result
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     */
    public static SignificanceResult test(long[][] counts) {
        final double chi2 = statistic(counts);
        final double df = (counts.length - 1.0) * (counts[0].length - 1.0);
        final double p = computeP(chi2, df);
        return new BaseSignificanceResult(chi2, p);
    }

    /**
     * Perform a Chi-square test of independence of frequency counts in
     * {@code observed1} and {@code observed2}.
     *
     * <p>Note: This is a specialized version of a 2-by-n contingency table.
     *
     * @param observed1 Observed frequency counts of the first data set.
     * @param observed2 Observed frequency counts of the second data set.
     * @return test result
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; either array has entries that are negative; either all
     * counts of {@code observed1} or {@code observed2} are zero; or if the count at
     * some index is zero for both arrays.
     */
    public static SignificanceResult test(long[] observed1, long[] observed2) {
        final double chi2 = statistic(observed1, observed2);
        final double p = computeP(chi2, observed1.length - 1.0);
        return new BaseSignificanceResult(chi2, p);
    }

    /**
     * Compute the Chi-square test p-value.
     *
     * @param chi2 Chi-square statistic.
     * @param degreesOfFreedom Degrees of freedom.
     * @return p-value
     */
    private static double computeP(double chi2, double degreesOfFreedom) {
        return ChiSquaredDistribution.of(degreesOfFreedom).survivalProbability(chi2);
    }

    /**
     * Check the array value is non-zero.
     *
     * @param value Value
     * @param name Name of the array
     * @param index Index in the array
     * @throws IllegalArgumentException if the value is zero
     */
    private static void checkNonZero(double value, String name, int index) {
        if (value == 0) {
            throw new InferenceException(InferenceException.ZERO_AT, name, index);
        }
    }
}
