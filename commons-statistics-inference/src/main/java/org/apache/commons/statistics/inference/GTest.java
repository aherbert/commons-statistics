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

import org.apache.commons.numbers.core.Sum;
import org.apache.commons.statistics.distribution.ChiSquaredDistribution;

/**
 * Implements G-test statistics.
 *
 * <p>This is known in statistical genetics as the McDonald-Kreitman test.
 * The implementation handles both known and unknown distributions.
 *
 * <p>Two samples tests can be used when the distribution is unknown <i>a priori</i>
 * but provided by one sample, or when the hypothesis under test is that the two
 * samples come from the same underlying distribution.
 *
 * @see <a href="https://en.wikipedia.org/wiki/G-test">G-test (Wikipedia)</a>
 * @since 1.1
 */
public class GTest {
    // Note:
    // The g-test statistic is a summation of terms with positive and negative sign
    // and thus the sum may exhibit cancellation. This class uses separate high precision
    // sums of the positive and negative terms which are then combined.
    // Total cancellation for a large number of terms will not impact
    // p-values of interest around critical alpha values as the Chi^2
    // distribution exhibits strong concentration around its mean (degrees of freedom, k).
    // The summation only need maintain enough bits in the final sum to distinguish
    // g values around critical alpha values where 0 << chisq.sf(g, k) << 0.5: g > k,
    // with k = number of terms - 1.

    /**
     * Computes the Goodness of Fit comparing {@code observed} and {@code expected}
     * frequency counts.
     *
     * <p>This statistic can be used to perform a G test (Log-Likelihood Ratio Test)
     * evaluating the null hypothesis that the observed counts follow the expected
     * distribution.
     *
     * <p><strong>Note:</strong>This implementation rescales the values
     * if necessary to ensure that the sum of the expected and observed counts
     * are equal.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return G-test statistic
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; or {@code observed} has negative entries.
     */
    public double g(double[] expected, long[] observed) {
        // g = 2 * sum{o * ln(o/e)}
        // The sum of o and e must be the same.
        final double ratio = InferenceUtils.computeRatio(expected, observed);
        // High precision sum to reduce cancellation.
        // Separate sum for positive and negative terms.
        final Sum sum = Sum.create();
        final Sum sum2 = Sum.create();
        for (int i = 0; i < observed.length; i++) {
            final long o = observed[i];
            // Process non-zero counts to avoid 0 * -inf = NaN
            if (o != 0) {
                final double term = o * Math.log(o / (ratio * expected[i]));
                if (term < 0) {
                    sum2.add(term);
                } else {
                    sum.add(term);
                }
            }
        }
        return sum.add(sum2).getAsDouble() * 2;
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated with a
     * G-test for goodness of fit comparing the {@code observed} frequency counts to
     * those in the {@code expected} array.
     *
     * <p>The number returned is the smallest significance level at which one can
     * reject the null hypothesis that the observed counts conform to the frequency
     * distribution described by the expected counts.
     *
     * <p>Note: The probability returned is the tail probability beyond
     * {@link #g(double[], long[]) g(expected, observed)} in the Chi-square
     * distribution with degrees of freedom <i>one</i> less than the common length
     * of {@code expected} and {@code observed}.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @return p-value
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; or {@code observed} has negative entries.
     * @see #g(double[], long[])
     */
    public double gTest(double[] expected, long[] observed) {
        final double g = g(expected, observed);
        return ChiSquaredDistribution.of(expected.length - 1.0).survivalProbability(g);
    }

//    /**
//     * Returns the intrinsic <i>observed significance level</i>, or p-value,
//     * associated with a G-test for goodness of fit comparing the {@code observed}
//     * frequency counts to those in the {@code expected} array.
//     *
//     * <p>The probability returned is the tail probability beyond
//     * {@link #g(double[], long[]) g(expected, observed)} in the ChiSquare
//     * distribution with degrees of freedom <i>two</i> less than the common length
//     * of {@code expected} and {@code observed}.
//     *
//     * @param expected Expected frequency counts.
//     * @param observed Observed frequency counts.
//     * @return p-value
//     * @throws IllegalArgumentException if the sample size is less than 2; the array
//     * sizes do not match; {@code expected} has entries that are not strictly
//     * positive; or {@code observed} has negative entries.
//     * @see #g(double[], long[])
//     */
//    // XXX should this be made available via an option?
//    //
//    public double gTestIntrinsic(double[] expected, long[] observed) {
//        final double g = g(expected, observed);
//        return ChiSquaredDistribution.of(expected.length - 2.0).survivalProbability(g);
//    }

    /**
     * Performs a G-test (Log-Likelihood Ratio Test) for goodness of fit evaluating
     * the null hypothesis that the observed counts conform to the frequency
     * distribution described by the expected counts, with significance level
     * {@code alpha}. Returns true iff the null hypothesis can be rejected with
     * {@code 100 * (1 - alpha)} percent confidence.
     *
     * <p><strong>Example:</strong>
     *
     * <p>To test the hypothesis that {@code observed} follows {@code expected} at
     * the 99% level, use {@code gTest(expected, observed, 0.01)}.
     *
     * @param expected Expected frequency counts.
     * @param observed Observed frequency counts.
     * @param alpha significance level of the test
     * @return true iff null hypothesis can be rejected with confidence 1 - alpha
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; {@code observed} has negative entries; or {@code alpha} is not in
     * the range {@code (0, 0.5]}.
     * @see #gTest(double[], long[])
     */
    public boolean gTest(double[] expected, long[] observed,
                         double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return gTest(expected, observed) < alpha;
    }

    /**
     * <p>Computes a G statistic (Log-Likelihood Ratio) associated with a G-test of
     * independence based on the input {@code counts} array, viewed as a two-way
     * table. The formula used to compute the test statistic is:
     *
     * <p>\[ G = 2 \cdot \sum_{ij}{O_{ij}} \cdot \left[ H(r) + H(c) - H(r,c) \right] \]
     *
     * <p>and \( H \) is the <a
     * href="https://en.wikipedia.org/wiki/Entropy_%28information_theory%29">
     * Shannon Entropy</a> of the random variable formed by viewing the elements of
     * the argument array as incidence counts:
     *
     * <p>\[ H(X) = - {\sum_{x \in \text{Supp}(X)} p(x) \ln p(x)} \]
     *
     * <p>This statistic can be used to perform a G-test evaluating the null
     * hypothesis that both observed counts are independent
     *
     * @param counts 2-way table.
     * @return G-test statistic
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     */
    public double g(long[][] counts) {
        InferenceUtils.checkCategoriesRequiredSize(counts.length, 2);
        InferenceUtils.checkValuesRequiredSize(counts[0].length, 2);
        InferenceUtils.checkRectangular(counts);
        InferenceUtils.checkNonNegative(counts);

        final int nRows = counts.length;
        final int nCols = counts[0].length;

        // compute row, column and total sums
        final double[] rowSum = new double[nRows];
        final double[] colSum = new double[nCols];
        double n = 0;
        // We can sum data on the first pass. See below for computation details.
        final Sum sum = Sum.create();
        for (int row = 0; row < nRows; row++) {
            for (int col = 0; col < nCols; col++) {
                final long c = counts[row][col];
                rowSum[row] += c;
                colSum[col] += c;
                if (c > 1) {
                    sum.add(c * Math.log(c));
                }
            }
            checkNonZero(rowSum[row], "row", row);
            n += rowSum[row];
        }

        for (int col = 0; col < nCols; col++) {
            checkNonZero(colSum[col], "column", col);
        }

        // There are alternative forms available:
        // https://en.wikipedia.org/wiki/G-test#Relation_to_mutual_information
        // This computes a modified form of the Shannon entropy H without requiring
        // normalisation of observations to probabilities and without negation,
        // i.e. we compute n * [ H(r) + H(c) - H(r,c) ] as [ H'(r,c) - H'(r) - H'(c) ].

        // H  = -sum (p * log(p))
        // H' = n * sum (p * log(p))
        //    = n * sum (o/n * log(o/n))
        //    = n * [ sum(o/n * log(o)) - sum(o/n * log(n)) ]
        //    = sum(o * log(o)) - n log(n)

        // After 3 modified entropy sums H'(r,c) - H'(r) - H'(c) compensation is (-1 + 2) * n log(n)
        sum.add(n * Math.log(n));
        // Negative terms
        final Sum sum2 = Sum.create();
        // All these counts are above zero so no check for zeros
        for (final double c : rowSum) {
            sum2.add(c * -Math.log(c));
        }
        for (final double c : colSum) {
            sum2.add(c * -Math.log(c));
        }

        return sum.add(sum2).getAsDouble() * 2;
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated with a
     * G-test (Log-Likelihood Ratio Test) of independence based on the input
     * {@code counts} array, viewed as a two-way table.
     *
     * <p>The number returned is the smallest significance level at which one can
     * reject the null hypothesis of independence.
     *
     * @param counts 2-way table.
     * @return p-value
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; or the
     * sum of a row or column is zero.
     * @see #g(long[][])
     */
    public double gTest(long[][] counts) {
        final double g = g(counts);
        final double df = (counts.length - 1.0) * (counts[0].length - 1.0);
        return ChiSquaredDistribution.of(df).survivalProbability(g);
    }

    /**
     * Performs a G-test (Log-Likelihood Ratio Test) of independence evaluating the
     * null hypothesis that the classifications represented by the counts in the
     * columns of the input 2-way table are independent of the rows, with
     * significance level {@code alpha}. Returns true iff the null hypothesis can be
     * rejected with 100 (1 - alpha) percent confidence.
     *
     * @param counts 2-way table.
     * @param alpha significance level of the test
     * @return true iff null hypothesis can be rejected with confidence 1 - alpha
     * @throws IllegalArgumentException if the number of rows or columns is less
     * than 2; the array is non-rectangular; the array has negative entries; the sum
     * of a row or column is zero; or {@code alpha} is not in the range
     * {@code (0, 0.5]}.
     */
    public boolean gTest(long[][] counts, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return gTest(counts) < alpha;
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
