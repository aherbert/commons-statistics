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
public class ChiSquareTest {
    /** Name for the row. */
    private static final String ROW = "row";
    /** Name for the column. */
    private static final String COLUMN = "column";

    /**
     * Computes the Chi-square statistic comparing {@code observed} and
     * {@code expected} frequency counts.
     *
     * <p>This statistic can be used to perform a Chi-square test evaluating the
     * null hypothesis that the observed counts follow the expected distribution.
     *
     * <p><strong>Note:</strong>This implementation rescales the {@code expected}
     * array if necessary to ensure that the sum of the expected and observed counts
     * are equal.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return Chi-square test statistic
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; or {@code observed} has negative entries.
     */
    public double chiSquare(double[] expected, long[] observed) {
        final double ratio = InferenceUtils.computeRatio(expected, observed);
        // Compute Chi-square
        double sumSq = 0;
        for (int i = 0; i < observed.length; i++) {
            final double e = ratio * expected[i];
            final double dev = observed[i] - e;
            sumSq += dev * dev / e;
        }
        return sumSq;
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated with a
     * Chi-square goodness of fit test comparing the {@code observed} frequency
     * counts to those in the {@code expected} array.
     *
     * <p>The number returned is the smallest significance level at which one can
     * reject the null hypothesis that the observed counts conform to the frequency
     * distribution described by the expected counts.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return p-value
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; or {@code observed} has negative entries.
     * @see #chiSquare(double[], long[])
     */
    public double chiSquareTest(double[] expected, long[] observed) {
        final double chi2 = chiSquare(expected, observed);
        return ChiSquaredDistribution.of(expected.length - 1.0).survivalProbability(chi2);
    }

    /**
     * Performs a Chi-square goodness of fit test evaluating the null hypothesis
     * that the observed counts conform to the frequency distribution described by
     * the expected counts, with significance level {@code alpha}. Returns true iff
     * the null hypothesis can be rejected with 100 * (1 - alpha) percent
     * confidence.
     *
     * <p><strong>Example:</strong>
     *
     * <p>To test the hypothesis that {@code observed} follows {@code expected} at
     * the 99% level, use {@code chiSquareTest(expected, observed, 0.01)}
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @param alpha significance level of the test
     * @return true iff null hypothesis can be rejected with confidence 1 - alpha
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; {@code observed} has negative entries; or {@code alpha} is not in
     * the range {@code (0, 0.5]}.
     * @see #chiSquareTest(double[], long[])
     */
    public boolean chiSquareTest(double[] expected, long[] observed,
                                 double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return chiSquareTest(expected, observed) < alpha;
    }

    /**
     * Computes the Chi-square statistic associated with a Chi-square test of
     * independence based on the input {@code counts} array, viewed as a two-way
     * table in row-major format.
     *
     * <p>This statistic can be used to perform a Chi-square test evaluating the
     * null hypothesis of independence.
     *
     * @param counts 2-way table.
     * @return Chi-square test statistic
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     */
    public double chiSquare(long[][] counts) {
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
        double sumSq = 0;
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                final double expected = (rowSum[row] * colSum[col]) / sum;
                sumSq += ((counts[row][col] - expected) *
                        (counts[row][col] - expected)) / expected;
            }
        }
        return sumSq;
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated with a
     * Chi-square test of independence based on the input {@code counts} array,
     * viewed as a two-way table.
     *
     * <p>The number returned is the smallest significance level at which one
     * can reject the null hypothesis of independence.
     *
     * @param counts 2-way table.
     * @return p-value
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     * @see #chiSquare(long[][])
     */
    public double chiSquareTest(long[][] counts) {
        final double chi2 = chiSquare(counts);
        final double df = (counts.length - 1.0) * (counts[0].length - 1.0);
        return ChiSquaredDistribution.of(df).survivalProbability(chi2);
    }

    /**
     * Performs a Chi-square test of independence evaluating the null hypothesis
     * that the classifications represented by the counts in the columns of the
     * input 2-way table are independent of the rows, with significance level
     * {@code alpha}. Returns true iff the null hypothesis can be rejected with 100
     * (1 - alpha) percent confidence.
     *
     * @param counts 2-way table.
     * @param alpha significance level of the test
     * @return true iff null hypothesis can be rejected with confidence 1 - alpha
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; the sum
     * of a row or column is zero; or {@code alpha} is not in the range
     * {@code (0, 0.5]}.
     * @see #chiSquareTest(long[][])
     */
    public boolean chiSquareTest(long[][] counts, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return chiSquareTest(counts) < alpha;
    }

    /**
     * Computes a Chi-square two sample test statistic comparing bin frequency
     * counts in {@code observed1} and {@code observed2}. The sums of frequency
     * counts in the two samples are not required to be the same. The formula used
     * to compute the test statistic is:
     *
     * <p>\[ sum_i{ \frac{(K * a_i - b_i / K)^2}{a_i + b_i} } \]
     *
     * <p>where
     *
     * <p>\[ K = \sqrt{ \sum_i{a_i} / \sum_i{b_i} } \]
     *
     * <p>This statistic can be used to perform a Chi-square test evaluating the
     * null hypothesis that both observed counts follow the same distribution.
     *
     * <p>Note: This is a specialized version of a 2-by-n contingency table. The
     * result is faster than calling {@link #chiSquare(long[][])} with the table
     * composed as {@code new long[][]{observed1, observed2}}. The statistic is the
     * same, with an allowance for accumulated floating-point error due to the
     * optimized routine.
     *
     * @param observed1 Observed frequency counts of the first data set.
     * @param observed2 Observed frequency counts of the second data set.
     * @return Chi-square test statistic
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; either array has entries that are negative; either all
     * counts of {@code observed1} or {@code observed2} are zero; or if the count at
     * some index is zero for both arrays.
     * @see #chiSquare(long[][])
     */
    public double chiSquareDataSetsComparison(long[] observed1, long[] observed2) {
        InferenceUtils.checkValuesRequiredSize(observed1.length, 2);
        InferenceUtils.checkValuesSizeMatch(observed1.length, observed2.length);
        InferenceUtils.checkNonNegative(observed1);
        InferenceUtils.checkNonNegative(observed2);

        // Compute and compare count sums
        long colSum1 = 0;
        long colSum2 = 0;
        for (int i = 0; i < observed1.length; i++) {
            final double obs1 = observed1[i];
            final double obs2 = observed2[i];
            checkNonZero(obs1 + obs2, ROW, i);
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
        double sumSq = 0;
        for (int i = 0; i < observed1.length; i++) {
            final double obs1 = observed1[i];
            final double obs2 = observed2[i];
            // apply weights
            final double dev = unequalCounts ?
                    obs1 / weight - obs2 * weight :
                    obs1 - obs2;
            sumSq += (dev * dev) / (obs1 + obs2);
        }
        return sumSq;
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated
     * with a Chi-square two sample test comparing bin frequency counts in
     * {@code observed1} and {@code observed2}.
     *
     * <p>The number returned is the smallest significance level at which one can
     * reject the null hypothesis that the observed counts conform to the same
     * distribution.
     *
     * @param observed1 Observed frequency counts of the first data set.
     * @param observed2 Observed frequency counts of the second data set.
     * @return p-value
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; either array has entries that are negative; either all
     * counts of {@code observed1} or {@code observed2} are zero; or if the count at
     * some index is zero for both arrays.
     * @see #chiSquareDataSetsComparison(long[], long[])
     */
    public double chiSquareTestDataSetsComparison(long[] observed1, long[] observed2) {
        final double chi2 = chiSquareDataSetsComparison(observed1, observed2);
        return ChiSquaredDistribution.of(observed1.length - 1.0).survivalProbability(chi2);
    }

    /**
     * Performs a Chi-square two sample test comparing bin frequency counts in
     * {@code observed1} and {@code observed2}. The test evaluates the null
     * hypothesis that the two lists of observed counts conform to the same
     * frequency distribution, with significance level {@code alpha}. Returns true
     * iff the null hypothesis can be rejected with 100 * (1 - alpha) percent
     * confidence.
     *
     * @param observed1 Observed frequency counts of the first data set.
     * @param observed2 Observed frequency counts of the second data set.
     * @param alpha significance level of the test
     * @return true iff null hypothesis can be rejected with confidence 1 - alpha
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; either array has entries that are negative; either all
     * counts of {@code observed1} or {@code observed2} are zero; the count at some
     * index is zero for both arrays; or {@code alpha} is not in the range
     * {@code (0, 0.5]}.
     * @see #chiSquareTestDataSetsComparison(long[], long[])
     */
    public boolean chiSquareTestDataSetsComparison(long[] observed1,
                                                   long[] observed2,
                                                   double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return chiSquareTestDataSetsComparison(observed1, observed2) < alpha;
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
