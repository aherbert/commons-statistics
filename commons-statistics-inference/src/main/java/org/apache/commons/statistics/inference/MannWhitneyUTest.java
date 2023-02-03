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

import java.lang.ref.SoftReference;
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
    /** Limit on sample size for the exact p-value computation for the auto mode. */
    private static final int AUTO_LIMIT = 50;
    /** Ranking instance. */
    private static final RankingAlgorithm RANKING = new NaturalRanking(NaNStrategy.FAILED, TiesStrategy.AVERAGE);
    /** Value for an unset f computation. */
    private static final double UNSET = -1;
    /** An object to use for synchonization when accessing the cache of F. */
    private static final Object LOCK = new Object();
    /** A reference to a previously computed storage for f.
     * Use of a SoftReference ensures this is garbage collected before an OutOfMemoryError.
     * The value should only be accessed, checked for size and optionally
     * modified when holding the lock. */
    private static SoftReference<double[][][]> cacheF = new SoftReference<>(null);

    /**
     * Options for the Mann-Whitney U test.
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
        /** Perform continuity correction. */
        private final boolean continuityCorrection;
        /** Expected location shift. */
        private final double mu;

        /**
         * Builder for the {@link Options}.
         */
        public static class Builder {
            /** Alternative hypothesis. */
            private AlternativeHypothesis alternative;
            /** Method to compute the p-value. */
            private PValueMethod pValue;
            /** Perform continuity correction. */
            private boolean continuityCorrection;
            /** Expected location shift. */
            private double mu;

            /**
             * @param source Source to copy.
             */
            Builder(Options source) {
                alternative = source.alternative;
                pValue = source.pValue;
                continuityCorrection = source.continuityCorrection;
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
             * Set to {@code true} to use a continuity correction for the approximate
             * p-value computation.
             *
             * @param v Value.
             * @return a reference to {@code this}
             * @see Options#isContinuityCorrection()
             */
            public Builder setContinuityCorrection(boolean v) {
                continuityCorrection = v;
                return this;
            }

            /**
             * Set the expected location shift.
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
            pValue = PValueMethod.AUTO;
            continuityCorrection = true;
            mu = 0;
        }

        /**
         * @param source Source to copy.
         */
        Options(Builder source) {
            alternative = source.alternative;
            pValue = source.pValue;
            continuityCorrection = source.continuityCorrection;
            mu = source.mu;
        }

        /**
         * Return the default options.
         *
         * <ul>
         * <li>{@link #getAlternative getAlternative = two-sided}
         * <li>{@link #getPValueMethod() getPValueMethod = auto}
         * <li>{@link #isContinuityCorrection() isContinuityCorrection = true}
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
         * Gets the method to compute the p-value.
         *
         * @return the p-value method
         */
        public PValueMethod getPValueMethod() {
            return pValue;
        }

        /**
         * If {@code true}, adjust the Wilcoxon rank statistic by 0.5 towards the
         * mean value when computing the z-statistic if a normal approximation is used
         * to compute the p-value.
         *
         * @return true to perform continuity correction
         */
        public boolean isContinuityCorrection() {
            return continuityCorrection;
        }

        /**
         * Return the location shift.
         *
         * @return the location shift
         */
        public double getMu() {
            return mu;
        }
    }

    /**
     * Result for the Mann-Whitney U test.
     *
     * <p>This class is immutable.
     */
    public static final class Result extends BaseSignificanceResult {
        /** Flag indicating the data has tied values. */
        private final boolean tiedValues;

        /**
         * Create an instance.
         *
         * @param statistic Test statistic.
         * @param tiedValues Flag indicating the data has tied values.
         * @param p Result p-value.
         */
        Result(double statistic, boolean tiedValues, double p) {
            super(statistic, p);
            this.tiedValues = tiedValues;
        }

        /**
         * Return {@code true} if the data had tied values.
         *
         * <p>Note: The exact computation cannot be used when there are tied values.
         *
         * @return {@code true} if there were tied values
         */
        public boolean hasTiedValues() {
            return tiedValues;
        }
    }

    /** No instances. */
    private MannWhitneyUTest() {}

    /**
     * Computes the Mann-Whitney U statistic comparing two independent
     * samples possibly of different length.
     *
     * <p>This statistic can be used to perform a Mann-Whitney U test evaluating the
     * null hypothesis that the two independent samples differ by a location shift of {@code mu}.
     *
     * <p>This returns the {@code U1} statistic. Compute the {@code U2} statistic
     * using:
     * <pre>
     * u2 = (long) x.length * y.length - u1;
     * </pre>
     *
     * @param mu Expected location shift.
     * @param x First sample values.
     * @param y Second sample values.
     * @return Mann-Whitney U1 statistic
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or contain
     * NaN values.
     */
    public static double statistic(double mu, double[] x, double[] y) {
        ensureDataConformance(x, y);

        final double[] z = concatenateSamples(mu, x, y);
        final double[] ranks = RANKING.apply(z);

        // The ranks for x is in the first x.length entries in ranks because x
        // is in the first x.length entries in z
        final double sumRankX = Arrays.stream(ranks).limit(x.length).sum();

        // U1 = R1 - (n1 * (n1 + 1)) / 2 where R1 is sum of ranks for sample 1,
        // e.g. x, n1 is the number of observations in sample 1.
        return sumRankX - ((long) x.length * (x.length + 1)) * 0.5;
    }

    /**
     * Performs a Mann-Whitney U test comparing the location for two independent
     * samples.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @return test result
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or contain
     * NaN values.
     * @see #statistic(double, double[], double[])
     * @see #test(double[], double[], Options)
     */
    public static Result test(double[] x, double[] y) {
        return test(x, y, Options.defaults());
    }

    /**
     * Performs a Mann-Whitney U test comparing the location for two independent
     * samples. The location is specified using {@link Options#getMu()}.
     *
     * <p>If the p-value method is {@link PValueMethod#AUTO auto} an exact p-value is
     * computed if the samples contain less than 50 values; otherwise a normal
     * approximation is used.
     *
     * <p>Computation of the exact p-value is only valid if there are no tied
     * ranks in the data; otherwise the p-value resorts to the asymptotic
     * approximation using a tie correction.
     *
     * <p><strong>Note: </strong>
     * Exact computation requires tabulation of values not exceeding size
     * {@code (n+1)*(m+1)*(u+1)} where {@code u} is the minimum of {@code u1} or {@code u2};
     * and {@code n} and {@code m} are the sample sizes. This may use a very large amount
     * of memory and result in an {@link OutOfMemoryError}.
     * Exact computation requires a finite binomial coefficient {@code binom(n+m, m)}
     * which is limited to {@code n+m <= 1029} for any @{@code n} and {@code m}; or
     * {@code min(n, m) <= 37}.
     *
     * @param x First sample values.
     * @param y Second sample values.
     * @param options Test options.
     * @return test result
     * @throws IllegalArgumentException if {@code x} or {@code y} are zero-length; or contain
     * NaN values.
     * @see #statistic(double, double[], double[])
     */
    public static Result test(double[] x, double[] y, Options options) {
        // Computation as above. The ranks are required for tie correction.
        ensureDataConformance(x, y);
        final double[] z = concatenateSamples(options.getMu(), x, y);
        final double[] ranks = RANKING.apply(z);
        final double sumRankX = Arrays.stream(ranks).limit(x.length).sum();
        final double u1 = sumRankX - ((long) x.length * (x.length + 1)) * 0.5;

        final double c = WilcoxonSignedRankTest.calculateTieCorrection(ranks);
        final boolean tiedValues = c != 0;

        PValueMethod method = options.getPValueMethod();
        final int n = x.length;
        final int m = y.length;
        if (method == PValueMethod.AUTO && Math.max(n, m) < AUTO_LIMIT) {
            method = PValueMethod.EXACT;
        }
        // Exact p requires no ties.
        // The method will fail-fast if the computation is not possible due
        // to the size of the data.
        double p = method == PValueMethod.EXACT && !tiedValues ?
            calculateExactPValue(u1, n, m, options) : -1;
        if (p < 0) {
            p = calculateAsymptoticPValue(u1, n, m, c, options);
        }
        return new Result(u1, tiedValues, p);
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
     * Concatenate the samples into one array. Subtract {@code mu} from the first sample.
     *
     * @param mu Expected difference between means.
     * @param x First sample values.
     * @param y Second sample values.
     * @return concatenated array
     */
    private static double[] concatenateSamples(double mu, double[] x, double[] y) {
        final double[] z = new double[x.length + y.length];
        System.arraycopy(x, 0, z, 0, x.length);
        System.arraycopy(y, 0, z, x.length, y.length);
        if (mu != 0) {
            for (int i = 0; i < x.length; i++) {
                z[i] -= mu;
            }
        }
        return z;
    }

    /**
     * Calculate the asymptotic p-value using a Normal approximation.
     *
     * @param u Mann-Whitney U value.
     * @param n1 Number of subjects in first sample.
     * @param n2 Number of subjects in second sample.
     * @param options Test options.
     * @param c Tie-correction
     * @return two-sided asymptotic p-value
     */
    private static double calculateAsymptoticPValue(double u, int n1, int n2, double c, Options options) {
        // Use long to avoid overflow
        final long n1n2 = (long) n1 * n2;
        final long n = (long) n1 + n2;

        // https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction
        final double e = n1n2 * 0.5;
        final double var = (n1n2 / 12.0) * ((n + 1.0) - c / n / (n - 1));

        double z = u - e;
        if (options.isContinuityCorrection()) {
            // +/- 0.5 is a continuity correction towards the expected.
            if (options.getAlternative() == AlternativeHypothesis.GREATER_THAN) {
                z -= 0.5;
            } else if (options.getAlternative() == AlternativeHypothesis.LESS_THAN) {
                z += 0.5;
            } else {
                // two-sided. Shift towards the expected of zero.
                // Use of signum ignores x==0 (i.e. not copySign(0.5, z))
                z -= Math.signum(z) * 0.5;
            }
        }
        z /= Math.sqrt(var);

        final NormalDistribution standardNormal = NormalDistribution.of(0, 1);
        if (options.getAlternative() == AlternativeHypothesis.GREATER_THAN) {
            return standardNormal.survivalProbability(z);
        }
        if (options.getAlternative() == AlternativeHypothesis.LESS_THAN) {
            return standardNormal.cumulativeProbability(z);
        }
        // two-sided
        return 2 * standardNormal.survivalProbability(Math.abs(z));
    }

    /**
     * Calculate the exact p-value.
     * If the value cannot be computed this returns -1.
     *
     * <p>Note: Computation may run out of memory during array allocation, or method recursion.
     *
     * @param u Mann-Whitney U value.
     * @param m Number of subjects in first sample.
     * @param n Number of subjects in second sample.
     * @param options Test options.
     * @return exact p-value (or -1) (two-sided, greater, or less using the options)
     */
    static double calculateExactPValue(double u, int m, int n, Options options) {
        // Check the computation can be attempted.
        // u must be an integer
        if ((int) u != u) {
            return -1;
        }
        // Note: n+m will not overflow as we concatenated the samples to a single array.
        final double binom = BinomialCoefficientDouble.value(n + m, m);
        if (binom == Double.POSITIVE_INFINITY) {
            return -1;
        }

        // Use u_min for the CDF.
        final int u1 = (int) u;
        final int u2 = (int) ((long) m * n - u1);
        // Use m < n to support symmetry.
        final int n1 = Math.min(m, n);
        final int n2 = Math.max(m, n);

        // Return the correct side:
        if (options.getAlternative() == AlternativeHypothesis.GREATER_THAN) {
            // sf(u1 - 1)
            return sf(u1 - 1, u2 + 1, n1, n2, binom);
        }
        if (options.getAlternative() == AlternativeHypothesis.LESS_THAN) {
            // cdf(u1)
            return cdf(u1, u2, n1, n2, binom);
        }
        // two-sided: 2 * sf(max(u1, u2) - 1) or 2 * cdf(min(u1, u2))
        final double p = 2 * computeCdf(Math.min(u1, u2), n1, n2, binom);
        // Clip to range: [0, 1]
        return Math.min(1, p);
    }

    /**
     * Compute the cumulative density function of the Mann-Whitney U1 statistic.
     * The U2 statistic is passed for convenience to exploit symmetry in the distribution.
     *
     * @param u1 Mann-Whitney U1 statistic
     * @param u2 Mann-Whitney U2 statistic
     * @param m First sample size.
     * @param n Second sample size.
     * @param binom binom(n+m, m) (must be finite)
     * @return {@code Pr(X <= k)}
     */
    private static double cdf(int u1, int u2, int m, int n, double binom) {
        // Exploit symmetry. Note the distribution is discrete thus requiring (u2 - 1).
        return u2 > u1 ?
            computeCdf(u1, m, n, binom) :
            1 - computeCdf(u2 - 1, m, n, binom);
    }

    /**
     * Compute the survival function of the Mann-Whitney U1 statistic.
     * The U2 statistic is passed for convenience to exploit symmetry in the distribution.
     *
     * @param u1 Mann-Whitney U1 statistic
     * @param u2 Mann-Whitney U2 statistic
     * @param m First sample size.
     * @param n Second sample size.
     * @param binom binom(n+m, m) (must be finite)
     * @return {@code Pr(X > k)}
     */
    private static double sf(int u1, int u2, int m, int n, double binom) {
        // Opposite of the CDF
        return u2 > u1 ?
            1 - computeCdf(u1, m, n, binom) :
            computeCdf(u2 - 1, m, n, binom);
    }

    /**
     * Compute the cumulative density function of the Mann-Whitney U statistic.
     *
     * <p>This should be called with the lower of U1 or U2 for computational efficiency.
     *
     * <p>Uses the recursive formula provided in Bucchianico, A.D, (1999)
     * Combinatorics, computer algebra and the Wilcoxon-Mann-Whitney test, Journal
     * of Statistical Planning and Inference, Volume 79, Issue 2, 349-364.
     *
     * @param k Mann-Whitney U statistic
     * @param m First sample size.
     * @param n Second sample size.
     * @param binom binom(n+m, m) (must be finite)
     * @return {@code Pr(X <= k)}
     */
    private static double computeCdf(int k, int m, int n, double binom) {
        // Theorem 2.5:
        // f(m, n, k) = 0 if k < 0, m < 0, n < 0, k > nm
        if (k < 0) {
            return 0;
        }
        // Recursively compute f(m, n, k)
        final double[][][] f = getF(m, n, k);

        // P(X=k) = f(m, n, k) / binom(m+n, m)
        // P(X<=k) = sum_0^k (P(X=i))

        // Called with k = min(u1, u2) : max(p) ~ 0.5 so no need to clip to [0, 1]
        return IntStream.rangeClosed(0, k).mapToDouble(i -> fmnk(f, m, n, i)).sum() / binom;
    }

    /**
     * Gets the storage for f(m, n, k).
     *
     * <p>This may be cached for performance.
     *
     * @param m M.
     * @param n N.
     * @param k K.
     * @return the storage for f
     */
    private static double[][][] getF(int m, int n, int k) {
        // Obtain any previous computation of f and expand it if required.
        // F is only modified within this synchronized block.
        // Any concurrent threads using a reference returned by this method
        // will not receive an index out-of-bounds as f is only ever expanded.
        synchronized (LOCK) {
            // Note: f(x<m, y<n, z<k) is always the same.
            // Cache the array and re-use any previous computation.
            double[][][] f = cacheF.get();

            // Require:
            // f = new double[m + 1][n + 1][k + 1]
            // f(m, n, 0) == 1; otherwise -1 if not computed
            // m+n <= 1029 for any m,n; k < mn/2 (due to symmetry using min(u1, u2))
            // Size m=n=515: approximately 516^2 * 515^2/2 = 398868 doubles ~ 3.04 GiB
            if (f == null) {
                f = new double[m + 1][n + 1][k + 1];
                for (final double[][] a : f) {
                    for (final double[] b : a) {
                        initialize(b);
                    }
                }
                // Cache for reuse.
                cacheF = new SoftReference<>(f);
                return f;
            }

            // Grow if required: m1 < m+1 => m1-(m+1) < 0 => m1 - m < 1
            final int m1 = f.length;
            final int n1 = f[0].length;
            final int k1 = f[0][0].length;
            final boolean growM = m1 - m < 1;
            final boolean growN = n1 - n < 1;
            final boolean growK = k1 - k < 1;
            if (growM | growN | growK) {
                // Some part of the previous f is too small.
                // Atomically grow without destroying the previous computation.
                // Any other thread using the previous f will not go out of bounds
                // by keeping the new f dimensions at least as large.
                // Note: Doing this in-place allows the memory to be gradually
                // increased rather than allocating a new [m + 1][n + 1][k + 1]
                // and copying all old values.
                final int sn = Math.max(n1, n + 1);
                final int sk = Math.max(k1, k + 1);
                if (growM) {
                    // Entirely new region
                    f = Arrays.copyOf(f, m + 1);
                    for (int x = m1; x <= m; x++) {
                        f[x] = new double[sn][sk];
                        for (final double[] b : f[x]) {
                            initialize(b);
                        }
                    }
                }
                // Expand previous in place if required
                if (growN) {
                    for (int x = 0; x < m1; x++) {
                        f[x] = Arrays.copyOf(f[x], sn);
                        for (int y = n1; y < sn; y++) {
                            final double[] b = f[x][y] = new double[sk];
                            initialize(b);
                        }
                    }
                }
                if (growK) {
                    for (int x = 0; x < m1; x++) {
                        for (int y = 0; y < n1; y++) {
                            final double[] b = f[x][y] = Arrays.copyOf(f[x][y], sk);
                            for (int z = k1; z < sk; z++) {
                                b[z] = UNSET;
                            }
                        }
                    }
                }
                // Avoided an OutOfMemoryError. Cache for reuse.
                cacheF = new SoftReference<>(f);
            }
            return f;
        }
    }

    /**
     * Initialize the array for f(m, n, x).
     * Set value to 1 for x=0; otherwise {@link #UNSET}.
     *
     * @param fmn Array.
     */
    private static void initialize(double[] fmn) {
        Arrays.fill(fmn, UNSET);
        // f(m, n, 0) == 1 if m >= 0, n >= 0
        fmn[0] = 1;
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
    private static double fmnk(double[][][] f, int m, int n, int k) {
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
            f[m][n][k] = fmnk = fmnk(f, m - 1, n, k - n) + fmnk(f, m, n - 1, k);
        }
        return fmnk;
    }
}
