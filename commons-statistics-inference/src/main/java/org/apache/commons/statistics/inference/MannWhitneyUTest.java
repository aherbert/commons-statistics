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
import java.util.stream.IntStream;
import org.apache.commons.numbers.combinatorics.BinomialCoefficientDouble;
import org.apache.commons.statistics.distribution.NormalDistribution;
import org.apache.commons.statistics.ranking.NaNStrategy;
import org.apache.commons.statistics.ranking.NaturalRanking;
import org.apache.commons.statistics.ranking.RankingAlgorithm;
import org.apache.commons.statistics.ranking.TiesStrategy;

/**
 * Implements the Mann-Whitney U test (also called Wilcoxon rank-sum test).
 *
 * @see <a href="https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test">
 * Mann-Whitney U test (Wikipedia)</a>
 * @since 1.1
 */
public final class MannWhitneyUTest {
    /** Limit on sample size for the exact p-value computation for {@code n+m}. */
    private static final int EXACT_LIMIT = 1029;
    /** Singleton default instance. */
    private static final MannWhitneyUTest INSTANCE = new MannWhitneyUTest(
        new NaturalRanking(NaNStrategy.FIXED, TiesStrategy.AVERAGE));

    /** Ranking algorithm. */
    private final RankingAlgorithm ranking;

    /**
     * @param ranking Ranking algorithm.
     */
    private MannWhitneyUTest(RankingAlgorithm ranking) {
        this.ranking = ranking;
    }

    /**
     * Return the default test instance.
     *
     * <p>Uses a ranking algorithm based on the
     * {@link org.apache.commons.statistics.ranking.NaturalRanking natural order} of
     * values. {@link Double#NaN NaN} values are
     * {@link org.apache.commons.statistics.ranking.NaNStrategy#FIXED left in place} and
     * ties get the {@link org.apache.commons.statistics.ranking.TiesStrategy#AVERAGE
     * average} of applicable ranks.
     *
     * <p>To change the behaviour use {@link #create(RankingAlgorithm)}.
     *
     * @return the test
     * @see #create(RankingAlgorithm)
     */
    public static MannWhitneyUTest instance() {
        return INSTANCE;
    }

    /**
     * Return a new test instance with the configured {@code ranking} algorithm.
     *
     * @param ranking Ranking algorithm.
     * @return the test
     * @throws NullPointerException if the {@code ranking} is null
     */
    public static MannWhitneyUTest create(RankingAlgorithm ranking) {
        return new MannWhitneyUTest(Objects.requireNonNull(ranking, "ranking"));
    }

    /**
     * Computes the Mann-Whitney U statistic comparing mean for two independent
     * samples possibly of different length.
     *
     * <p>This statistic can be used to perform a Mann-Whitney U test evaluating the
     * null hypothesis that the two independent samples has equal mean.
     *
     * <p>This returns the {@code U1} statistic. Compute the {@code U2} statistic
     * using:
     * <pre>
     * u2 = (long) x.length * y.length - u1;
     * </pre>
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @return Mann-Whitney U1 statistic
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length.
     */
    public double mannWhitneyU(double[] x, double[] y) {
        ensureDataConformance(x, y);

        final double[] z = concatenateSamples(x, y);
        final double[] ranks = ranking.apply(z);

        // The ranks for x is in the first x.length entries in ranks because x
        // is in the first x.length entries in z
        final double sumRankX = Arrays.stream(ranks).limit(x.length).sum();

        // U1 = R1 - (n1 * (n1 + 1)) / 2 where R1 is sum of ranks for sample 1,
        // e.g. x, n1 is the number of observations in sample 1.
        return sumRankX - ((long) x.length * (x.length + 1)) * 0.5;
    }

    /**
     * Returns the asymptotic <i>observed significance level</i>, or p-value,
     * associated with a Mann-Whitney U statistic comparing mean for two independent
     * samples.
     *
     * <p>The returned p-value is the smallest significance level at which one can
     * reject the null hypothesis that the two means are equal in favor of the
     * two-sided alternative that they are different.
     *
     * <p>Computation of the exact p-value requires the sample size {@code <= 1023}.
     * Exact computation requires tabulation of values not exceeding size
     * {@code n*m*u} where {@code u} is the minimum of {@code u1} or {@code u2}.
     * Exact computation is only valid if there are no no tied ranks in the data. If
     * these conditions are not satisfied then the p-value resorts to the asymptotic
     * approximation using a tie correction.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @param exactPValue Set to {@code true} to compute the exact p-value.
     * @return asymptotic p-value
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length.
     * @see #mannWhitneyU(double[], double[])
     */
    public double mannWhitneyUTest(double[] x, double[] y,
                                   boolean exactPValue) {
        ensureDataConformance(x, y);

        // Computation as above. The ranks are required for tie correction
        final double[] z = concatenateSamples(x, y);
        final double[] ranks = ranking.apply(z);
        final double sumRankX = Arrays.stream(ranks).limit(x.length).sum();
        final double u1 = sumRankX - ((long) x.length * (x.length + 1)) * 0.5;

        final double c = WilcoxonSignedRankTest.calculateTieCorrection(ranks);
        // Exact p-value is limited by binom(n+m, m)
        if (exactPValue && Integer.toUnsignedLong(x.length + y.length) <= EXACT_LIMIT && c == 0) {
            return calculateExactPValue((int) u1, x.length, y.length);
        }
        return calculateAsymptoticPValue(u1, x.length, y.length, c);
    }

    /**
     * Performs a two-sided Mann-Whitney U evaluating the null hypothesis that
     * samples 1 and 2 describe datasets drawn from populations with the same mean,
     * with significance level {@code alpha}.
     *
     * <p>Returns {@code true} iff the null hypothesis that the means are equal can
     * be rejected with confidence {@code 1 - alpha}. To perform a 1-sided test, use
     * {@code alpha * 2}.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param exactPValue Set to {@code true} to compute the exact p-value.
     * @param alpha Significance level of the test.
     * @return true if the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length,
     * or {@code alpha} is not in the range {@code (0, 0.5]}
     * @see #mannWhitneyUTest(double[], double[], boolean)
     */
    public boolean mannWhitneyUTest(double[] sample1, double[] sample2,
                                    boolean exactPValue, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return mannWhitneyUTest(sample1, sample2, exactPValue) < alpha;
    }

    /**
     * Ensures that the provided arrays fulfil the assumptions.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length.
     */
    private static void ensureDataConformance(double[] x, double[] y) {
        InferenceUtils.checkValuesRequiredSize(x.length, 1);
        InferenceUtils.checkValuesRequiredSize(y.length, 1);
    }

    /**
     * Concatenate the samples into one array.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @return concatenated array
     */
    private static double[] concatenateSamples(double[] x, double[] y) {
        final double[] z = new double[x.length + y.length];
        System.arraycopy(x, 0, z, 0, x.length);
        System.arraycopy(y, 0, z, x.length, y.length);
        return z;
    }

    /**
     * @param u Mann-Whitney U value.
     * @param n1 Number of subjects in first sample.
     * @param n2 Number of subjects in second sample.
     * @param c Tie-correction
     * @return two-sided asymptotic p-value
     */
    private static double calculateAsymptoticPValue(double u, int n1, int n2, double c) {
        // Use long to avoid overflow
        final long n1n2 = (long) n1 * n2;
        final long n = (long) n1 + n2;

        // https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction
        final double e = n1n2 * 0.5;
        final double var = (n1n2 / 12.0) * ((n + 1.0) - c / n / (n - 1));

        double x = u - e;
        // +/- 0.5 is a continuity correction towards the expected
        // XXX - could be made optional
        // Use of signum ignores x==0
        x = (x - Math.signum(x) * 0.5) / Math.sqrt(var);

        final NormalDistribution standardNormal = NormalDistribution.of(0, 1);

        return 2 * standardNormal.survivalProbability(Math.abs(x));
    }

    /**
     * Calculate the exact two-sided p-value.
     * This assume that {@code n+m <= 1029}.
     *
     * @param u Mann-Whitney U value.
     * @param m Number of subjects in first sample.
     * @param n Number of subjects in second sample.
     * @return two-sided exact p-value
     */
    private static double calculateExactPValue(int u, int m, int n) {
        // Use u_min. No overflow if n+m <= 1029.
        final int u2 = m * n - u;
        // Use m < n to support symmetry
        return 2 * cdf(Math.min(u, u2), Math.min(m, n), Math.max(m, n));
    }

    /**
     * Compute the cumulative density function of the Mann-Whitney U statistic.
     *
     * <p>This should be called with the lower of U1 or U2 for computational efficiency.
     * The input value m+n is limited to 1029.
     *
     * <p>Uses the recursive formula provided in Bucchianico, A.D, (1999)
     * Combinatorics, computer algebra and the Wilcoxon-Mann-Whitney test, Journal
     * of Statistical Planning and Inference, Volume 79, Issue 2, 349-364.
     *
     * @param k Mann-Whitney U statistic
     * @param m First sample size.
     * @param n Second sample size.
     * @return {@code Pr(X <= k)}
     */
    private static double cdf(int k, int m, int n) {
        // Recursively compute f(m, n, k)
        // m+n <= 1029; k < mn/2 (due to symmetry using min(u1, u2))
        // Max size is m=n=515: approximately 516^2 * 515^2/2 = 398868 doubles ~ 3.04 GiB
        final double[][][] f = new double[m + 1][n + 1][k + 1];
        // Initialize as not computed, or using the base recursion condition for k=0
        for (final double[][] a : f) {
            for (final double[] b : a) {
                Arrays.fill(b, -1);
                // f(m, n, 0) == 1 if m >= 0, n >= 0
                b[0] = 1;
            }
        }

        // P(X=k) = f(m, n, k) / binom(m+n, m)
        // P(X<=k) = sum_0^k (P(X=i))

        // Arguments assume that binom will not overflow a double
        final double binom = BinomialCoefficientDouble.value(m + n, m);
        return IntStream.rangeClosed(0, k).mapToDouble(i -> f(f, m, n, i)).sum() / binom;
    }

    /**
     * Compute f(m; n; k), the number of subsets of {0; 1; ...; n} with m elements such
     * that the elements of this subset add up to k.
     *
     * <p>The function is computed recursively.
     *
     * @param f Tabulated values of f[m][n][k].
     * @param m M
     * @param n N
     * @param k K
     * @return f(m; n; k)
     */
    private static double f(double[][][] f, int m, int n, int k) {
        // Theorem 2.5:
        // Omit conditions that will not be met: k > mn
        // f(m, n, k) = 0 if k < 0, m < 0, n < 0
        if ((k | m | n) < 0) {
            return 0;
        }
        // Compute on demand
        double fmnk = f[m][n][k];
        if (fmnk < 0) {
            // f(m, n, 0) == 1 if m >= 0, n >= 0
            // This is already computed.

            // Recursion from formula (3):
            // f(m, n, k) = f(m-1, n, k-n) + f(m, n-1, k)
            f[m][n][k] = fmnk = f(f, m - 1, n, k - n) + f(f, m, n - 1, k);
        }
        return fmnk;
    }
}
