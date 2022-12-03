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
import org.apache.commons.statistics.distribution.NormalDistribution;
import org.apache.commons.statistics.ranking.NaNStrategy;
import org.apache.commons.statistics.ranking.NaturalRanking;
import org.apache.commons.statistics.ranking.RankingAlgorithm;
import org.apache.commons.statistics.ranking.TiesStrategy;

/**
 * Implements the Wilcoxon signed-rank test.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test">Wilcoxon signed-rank test (Wikipedia)</a>
 * @since 1.1
 */
public final class WilcoxonSignedRankTest {
    /** Limit on sample size for the exact p-value computation. */
    private static final int EXACT_LIMIT = 1023;
    /** Singleton default instance. */
    // XXX Sould this use NaNStrategy.FAILED as NaN would invalidate the p-value computation.
    // Any TiesStratgey other than AVERAGE invalidates the asymptotic p-value computation
    // tie correction. So perhaps this should not be configurable at all.
    private static final WilcoxonSignedRankTest INSTANCE = new WilcoxonSignedRankTest(
        new NaturalRanking(NaNStrategy.FIXED, TiesStrategy.AVERAGE));

    /** Ranking algorithm. */
    private final RankingAlgorithm ranking;

    /**
     * @param ranking Ranking algorithm.
     */
    private WilcoxonSignedRankTest(RankingAlgorithm ranking) {
        this.ranking = ranking;
    }

    /**
     * Return the default test instance. This computes an <i>average rank</i> procedure.
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
    public static WilcoxonSignedRankTest instance() {
        return INSTANCE;
    }

    /**
     * Return a new test instance with the configured {@code ranking} algorithm.
     *
     * @param ranking Ranking algorithm.
     * @return the test
     * @throws NullPointerException if the {@code ranking} is null
     */
    public static WilcoxonSignedRankTest create(RankingAlgorithm ranking) {
        return new WilcoxonSignedRankTest(Objects.requireNonNull(ranking, "ranking"));
    }

    /**
     * Computes the Wilcoxon signed ranked statistic comparing mean for two related
     * samples or repeated measurements on a single sample.
     *
     * <p>This statistic can be used to perform a Wilcoxon signed ranked test
     * evaluating the null hypothesis that the two related samples or repeated
     * measurements on a single sample has equal mean.
     *
     * <p>Let X<sub>i</sub> denote the i'th individual of the first sample and
     * Y<sub>i</sub> the related i'th individual in the second sample. Let
     * Z<sub>i</sub> = X<sub>i</sub> - Y<sub>i</sub>.
     *
     * <p><strong>Preconditions</strong>:
     * <ul>
     * <li>The differences Z<sub>i</sub> must be independent.
     * <li>Each Z<sub>i</sub> comes from a continuous population (they must be
     * identical) and is symmetric about a common median.
     * <li>The values that X<sub>i</sub> and Y<sub>i</sub> represent are
     * ordered, so the comparisons greater than, less than, and equal to are
     * meaningful.
     * </ul>
     *
     * <p>This method handles matching samples {@code x[i] == y[i]} (zero difference)
     * by including them in the ranking of samples but excludes them from the test statistic
     * (<i>signed-rank zero procedure</i>).
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @return Wilcoxon <i>positive-rank sum</i> statistic (W+)
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or do not
     * have the same length.
     */
    public double wilcoxonSignedRank(double[] sample1, double[] sample2) {
        ensureDataConformance(sample1, sample2);
        final double[] z = calculateDifferences(sample1, sample2);
        final double[] zAbs = calculateAbsoluteDifferences(z);
        final double[] ranks = ranking.apply(zAbs);
        return calculateW(z, ranks);
    }

    /**
     * Returns the <i>observed significance level</i>, or p-value, associated with a
     * Wilcoxon signed ranked statistic comparing mean for two related
     * samples or repeated measurements on a single sample.
     *
     * <p>The returned p-value is the smallest significance level at which one can
     * reject the null hypothesis that the two related samples or repeated
     * measurements on a single sample have equal mean in favor of the
     * two-sided alternative that they are different.
     *
     * <p>Computation of the exact p-value requires the sample size {@code <= 1023}.
     * Exact computation requires tabulation of values not exceeding size {@code n(n+1)/2}
     * and computes in order n^2/2. Exact computation is only valid if there are no
     * matching samples {@code x[i] == y[i]} and no tied ranks in the data. If these
     * conditions are not satisfied then the p-value resorts to the Cureton approximation
     * using a tie correction.
     *
     * @param sample1 First sample values.
     * @param sample2 Second sample values.
     * @param exactPValue Set to {@code true} to compute the exact p-value
     * @return p-value
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or do not
     * have the same length
     * @see #wilcoxonSignedRank(double[], double[])
     */
    public double wilcoxonSignedRankTest(double[] sample1, double[] sample2,
                                         boolean exactPValue) {
        ensureDataConformance(sample1, sample2);
        final double[] z = calculateDifferences(sample1, sample2);
        final double[] zAbs = calculateAbsoluteDifferences(z);
        final double[] ranks = ranking.apply(zAbs);
        final double wPlus = calculateW(z, ranks);

        // Exact p has strict requirements for no zeros, no ties
        final int zeros = countZeros(z);
        final double c = calculateTieCorrection(ranks);
        final int n = sample1.length;
        if (exactPValue && n <= EXACT_LIMIT && zeros + c == 0) {
            return calculateExactPValue((int) wPlus, n);
        }
        return calculateAsymptoticPValue(wPlus, n, zeros, c);
    }


    /**
     * Performs a two-sided Wilcoxon signed ranked test evaluating the null
     * hypothesis that the two related samples or repeated measurements on a single
     * sample has equal mean with significance level {@code alpha}.
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
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length;
     * do not have the same length; or {@code alpha} is not in the range
     * {@code (0, 0.5]}
     * @see #wilcoxonSignedRankTest(double[], double[], boolean)
     */
    public boolean wilcoxonSignedRankTest(double[] sample1, double[] sample2,
                                          boolean exactPValue, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return wilcoxonSignedRankTest(sample1, sample2, exactPValue) < alpha;
    }

    /**
     * Ensures that the provided arrays fulfil the assumptions.
     *
     * @param x First sample.
     * @param y Second sample.
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or do not
     * have the same length
     */
    private static void ensureDataConformance(double[] x, double[] y) {
        InferenceUtils.checkValuesRequiredSize(x.length, 1);
        InferenceUtils.checkValuesRequiredSize(y.length, 1);
        InferenceUtils.checkValuesSizeMatch(x.length, y.length);
    }

    /**
     * Calculates x[i] - y[i] for all i.
     *
     * @param x First sample.
     * @param y Second sample.
     * @return z = x - y
     */
    private static double[] calculateDifferences(double[] x, double[] y) {
        final double[] z = new double[x.length];
        for (int i = 0; i < x.length; ++i) {
            z[i] = x[i] - y[i];
        }
        return z;
    }

    /**
     * Calculates |z[i]| for all i.
     *
     * @param z Sample.
     * @return |z|
     */
    private static double[] calculateAbsoluteDifferences(double[] z) {
        final double[] zAbs = new double[z.length];
        for (int i = 0; i < z.length; ++i) {
            zAbs[i] = Math.abs(z[i]);
        }
        return zAbs;
    }

    /**
     * Calculate the Wilcoxon <i>positive-rank sum</i> statistic.
     *
     * @param obs Observed signed value.
     * @param ranks Ranks (including averages for ties)
     * @return Wilcoxon <i>positive-rank sum</i> statistic (W+)
     */
    private static double calculateW(final double[] obs, final double[] ranks) {
        double wPlus = 0;
        for (int i = 0; i < obs.length; ++i) {
            // Must be positive (excludes zeros)
            if (obs[i] > 0) {
                wPlus += ranks[i];
            }
        }
        return wPlus;
    }

    /**
     * Count the number of zeros in the data.
     *
     * @param z Input data.
     * @return the zero count
     */
    private static int countZeros(final double[] z) {
        int c = 0;
        for (final double v : z) {
            if (v == 0) {
                c++;
            }
        }
        return c;
    }

    /**
     * Calculate the tie correction.
     * Destructively modifies ranks (by sorting).
     * <pre>
     * c = sum(t^3 - t)
     * </pre>
     * <p>where t is the size of each group of tied observations.
     *
     * @param ranks Ranks
     * @return the tie correction
     */
    static double calculateTieCorrection(double[] ranks) {
        double c = 0;
        int ties = 1;
        Arrays.sort(ranks);
        double last = Double.NaN;
        for (final double rank : ranks) {
            // Deliberate use of equals
            if (last == rank) {
                // Extend the tied group
                ties++;
            } else {
                if (ties != 1) {
                    c += (double) ties * ties * ties - ties;
                    ties = 1;
                }
                last = rank;
            }
        }
        // Final ties count
        c += (double) ties * ties * ties - ties;
        return c;
    }

    /**
     * Compute the asymptotic p-value using the Cureton normal approximation. This
     * corrects for zeros in the signed-rank zero procedure and/or ties corrected
     * using the average method.
     *
     * @param wPlus Wilcoxon signed rank value (W+).
     * @param n Number of subjects.
     * @param z Count of number of zeros
     * @param c Tie-correction
     * @return two-sided asymptotic p-value
     */
    private static double calculateAsymptoticPValue(double wPlus, int n, double z, double c) {
        // E[W+] = n * (n + 1) / 4 - z * (z + 1) / 4
        final double e = (n * (n + 1.0) - z * (z + 1.0)) * 0.25;

        final double var = ((n * (n + 1.0) * (2 * n + 1.0)) -
                            (z * (z + 1.0) * (2 * z + 1.0)) -
                             c * 0.5) / 24;

        double x = wPlus - e;
        // +/- 0.5 is a continuity correction towards the expected
        // XXX - could be made optional
        // Use of signum ignores x==0
        x = (x - Math.signum(x) * 0.5) / Math.sqrt(var);

        final NormalDistribution standardNormal = NormalDistribution.of(0, 1);

        return 2 * standardNormal.survivalProbability(Math.abs(x));
    }

    /**
     * Compute the exact p-value for a two-sided test.
     *
     * <p>This computation requires that no zeros or ties are found in the data.
     * The input value n is limited to 1023.
     *
     * @param w Wilcoxon signed rank value (W+, or W-).
     * @param n Number of subjects.
     * @return two-sided exact p-value
     */
    private static double calculateExactPValue(int w, int n) {
        // T+ plus T- equals the sum of the ranks: n(n+1)/2
        // Compute using the lower half.
        // No overflow here if n <= 1023.
        final int sum = n * (n + 1) / 2;
        final int t = Math.min(w,  sum - w);
        // Two-sided test
        return 2 * cdf(t, n);
    }

    /**
     * Compute the cumulative density function for the distribution of the Wilcoxon
     * signed rank statistic. This is a discrete distribution and is only valid
     * when no zeros or ties are found in the data.
     *
     * <p>This should be called with the lower of W+ or W- for computational efficiency.
     * The input value n is limited to 1023.
     *
     * <p>Uses recursion to compute the density for {@code X <= t} and sums the values.
     * See: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test#Computing_the_null_distribution
     *
     * @param t Smallest Wilcoxon signed rank value (W+, or W-).
     * @param n Number of subjects.
     * @return {@code Pr(T <= t)}
     */
    private static double cdf(int t, int n) {
        // Currently limited to n=1023.
        // Note:
        // The limit for t is n(n+1)/2.
        // The highest possible sum is bounded by the normalisation factor 2^n.
        // n         t              sum          support
        // 31        [0, 496]       < 2^31       int
        // 63        [0, 2016]      < 2^63       long
        // 1023      [0, 523766]    < 2^1023     double
        // Support for n up to 1023 is possible using a double for 2^n and the table.
        assert n >= 2 && n <= EXACT_LIMIT;

        if (t == 0) {
            // No recursion required
            return Math.scalb(1, -n);
        }

        // Define u_n(t) as the number of sign combinations for T = t
        // Pr(T == t) = u_n(t) / 2^n
        // Sum them to create the cumulative probability Pr(T <= t).
        //
        // Recursive formula:
        // u_n(t) = u_{n-1}(t) + u_{n-1}(t-n)
        // u_0(0) = 1
        // u_0(t) = 0 : t != 0
        // u_n(t) = 0 : t < 0 || t > n(n+1)/2

        // Compute all u_n(t) up to t.
        final double[] u = new double[t + 1];
        // Initialize u_1(t) using base cases for recursion
        u[0] = u[1] = 1;

        // Each u_n(t) is created using the current correct values for u_{n-1}(t)
        for (int nn = 2; nn < n + 1; nn++) {
            // u[t] holds the correct value for u_{n-1}(t)
            // u_n(t) = u_{n-1}(t) + u_{n-1}(t-n)
            for (int tt = t; tt >= nn; tt--) {
                u[tt] += u[tt - nn];
            }
        }
        final double sum = Arrays.stream(u).sum();

        // Finally divide by the number of possible sums: 2^n
        return Math.scalb(sum, -n);
    }
}
