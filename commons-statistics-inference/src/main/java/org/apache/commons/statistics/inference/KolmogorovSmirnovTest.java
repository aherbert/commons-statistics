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
import java.util.SplittableRandom;
import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
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
 * default 2-sample test method, {@link #kolmogorovSmirnovTest(double[], double[], boolean)} works as
 * follows:
 *
 * <ul>
 * <li>When both sample sizes are less than 10000, the method presented in [5]
 * is used to compute the exact p-value for the 2-sample test. The {@code boolean}
 * arguments allows the probability used to estimate the p-value to be
 * expressed using strict or non-strict inequality.
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
 * the {@link #estimateP(double[],double[],int,boolean,UniformRandomProvider)}
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
 * </ol>
 *
 * <p>Note that [1] contains an error in computing h, refer to <a
 * href="https://issues.apache.org/jira/browse/MATH-437">MATH-437</a> for details.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test">
 * Kolmogorov-Smirnov (K-S) test (Wikipedia)</a>
 * @since 1.1
 */
public class KolmogorovSmirnovTest {
    /** Name for sample 1. */
    private static final String SAMPLE_1_NAME = "Sample 1";
    /** Name for sample 2. */
    private static final String SAMPLE_2_NAME = "Sample 2";
    /** When the largest sample size exceeds this value, 2-sample K-S test uses asymptotic
     * distribution to compute the p-value. */
    private static final int LARGE_SAMPLE = 10000;

    /**
     * Computes the one-sample Kolmogorov-Smirnov test statistic, \(D_n=\sup_x |F_n(x)-F(x)|\) where
     * \(F\) is the distribution (cdf) function associated with {@code distribution}, \(n\) is the
     * length of {@code data} and \(F_n\) is the empirical distribution that puts mass \(1/n\) at
     * each of the values in {@code data}.
     *
     * <p>The cumulative distribution function should map a real value {@code x} to a probability
     * in [0, 1]. To use a reference distribution the CDF can be passed using a method reference:
     * <pre>
     * UniformContinuousDistribution dist = UniformContinuousDistribution.of(0, 1);
     * UniformRandomProvider rng = RandomSource.KISS.create(123);
     * double[] x = dist.sampler(rng).samples(100);
     * double d = kolmogorovSmirnovStatistic(dist::cumulativeProbability, x);
     * </pre>
     *
     * @param cdf Reference cumulative distribution function.
     * @param data Sample being evaluated.
     * @return Kolmogorov-Smirnov statistic \(D_n\)
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; or contains NaN values.
     */
    public double kolmogorovSmirnovStatistic(DoubleUnaryOperator cdf, double[] data) {
        final int n = checkArrayLength(data);
        final double nd = n;
        final double[] x = sort(data.clone(), "Sample");
        double d = 0;
        // Note: ties in the data do not matter as we compare the empirical CDF
        // immediately before the value (i/n) and at the value (i+1)/n. For ties
        // of length m this would be (i-m+1)/n and (i+1)/n and the result is the same.
        for (int i = 0; i < n; i++) {
            final double yi = cdf.applyAsDouble(x[i]);
            final double currD = Math.max(yi - i / nd, (i + 1) / nd - yi);
            if (currD > d) {
                d = currD;
            }
        }
        return d;
    }

    /**
     * Computes the <i>p-value</i>, or <i>observed significance level</i>, of a one-sample
     * Kolmogorov-Smirnov test
     * evaluating the null hypothesis that {@code data} conforms to {@code distribution}. If
     * {@code exact} is true, the distribution used to compute the p-value is computed using
     * extended precision.
     *
     * @param cdf reference distribution
     * @param data sample being being evaluated
     * @return the p-value associated with the null hypothesis that {@code data} is a sample from
     *         {@code distribution}
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; or contains NaN values.
     * @see #kolmogorovSmirnovStatistic(DoubleUnaryOperator, double[])
     */
    public double kolmogorovSmirnovTest(DoubleUnaryOperator cdf, double[] data) {
        return KolmogorovSmirnovDistribution.Two.sf(kolmogorovSmirnovStatistic(cdf, data), data.length);
    }

    /**
     * Performs a Kolmogorov-Smirnov
     * test evaluating the null hypothesis that {@code data} conforms to {@code distribution}.
     *
     * @param distribution reference distribution
     * @param data sample being being evaluated
     * @param alpha significance level of the test
     * @return true iff the null hypothesis that {@code data} is a sample from {@code distribution}
     *         can be rejected with confidence 1 - {@code alpha}
     * @throws IllegalArgumentException if {@code data} does not have length at least 2; contains NaN values;
     *  or {@code alpha} is not in the range {@code (0, 0.5]}.
     * @see #kolmogorovSmirnovTest(DoubleUnaryOperator, double[])
     */
    public boolean kolmogorovSmirnovTest(DoubleUnaryOperator distribution, double[] data, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return kolmogorovSmirnovTest(distribution, data) < alpha;
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic, \(D_{n,m}=\sup_x |F_n(x)-F_m(x)|\)
     * where \(n\) is the length of {@code x}, \(m\) is the length of {@code y}, \(F_n\) is the
     * empirical distribution that puts mass \(1/n\) at each of the values in {@code x} and \(F_m\)
     * is the empirical distribution of the {@code y} values.
     *
     * @param x first sample
     * @param y second sample
     * @return test statistic \(D_{n,m}\) used to evaluate the null hypothesis that {@code x} and
     *         {@code y} represent samples from the same underlying distribution
     * @throws IllegalArgumentException if either {@code x} or {@code y} does not have length at
     *         least 2; or contain NaN values.
     */
    public double kolmogorovSmirnovStatistic(double[] x, double[] y) {
        final double n = checkArrayLength(x);
        final double m = checkArrayLength(y);
        // Avoid destructive modification of input
        return integralKolmogorovSmirnovStatistic(x.clone(), y.clone()) / (n * m);
    }

    /**
     * Computes the <i>p-value</i>, or <i>observed significance level</i>, of a two-sample
     * Kolmogorov-Smirnov test
     * evaluating the null hypothesis that {@code x} and {@code y} are samples drawn from the same
     * probability distribution. Specifically, what is returned is an estimate of the probability
     * that the {@link #kolmogorovSmirnovStatistic(double[], double[])} associated with a randomly
     * selected partition of the combined sample into subsamples of sizes {@code x.length} and
     * {@code y.length} will strictly exceed (if {@code strict} is {@code true}) or be at least as
     * large as (if {@code strict} is {@code false}) as {@code kolmogorovSmirnovStatistic(x, y)}.
     *
     * @param x first sample dataset.
     * @param y second sample dataset.
     * @param strict whether or not the probability to compute is expressed as
     * a strict inequality (ignored for large samples).
     * @return p-value associated with the null hypothesis that {@code x} and
     * {@code y} represent samples from the same distribution.
     * @throws IllegalArgumentException if either {@code x} or {@code y} does
     * not have length at least 2; or contain NaN values.
     * @see #kolmogorovSmirnovStatistic(double[], double[])
     */
    public double kolmogorovSmirnovTest(double[] x, double[] y, boolean strict) {
        return kolmogorovSmirnovTest(x, y, strict, 0);
    }

    /**
     * Computes the <i>p-value</i>, or <i>observed significance level</i>, of a
     * two-sample Kolmogorov-Smirnov test evaluating the null hypothesis that
     * {@code x} and {@code y} are samples drawn from the same probability
     * distribution. Specifically, what is returned is an estimate of the
     * probability that the {@link #kolmogorovSmirnovStatistic(double[], double[])}
     * associated with a randomly selected partition of the combined sample into
     * subsamples of sizes {@code x.length} and {@code y.length} will strictly
     * exceed (if {@code strict} is {@code true}) or be at least as large as (if
     * {@code strict} is {@code false}) as {@code kolmogorovSmirnovStatistic(x, y)}.
     *
     * @param x first sample dataset.
     * @param y second sample dataset.
     * @param strict whether or not the probability to compute is expressed as a
     * strict inequality (ignored for large samples).
     * @param seed Seed for random tie resolution.
     * @return p-value associated with the null hypothesis that {@code x} and
     * {@code y} represent samples from the same distribution.
     * @throws IllegalArgumentException if either {@code x} or {@code y} does not
     * have length at least 2; or contain NaN values.
     * @see #kolmogorovSmirnovStatistic(double[], double[])
     */
    // XXX - add public API options to allow this to be called with a configurable seed
    // package-private for testing
    static double kolmogorovSmirnovTest(double[] x, double[] y, boolean strict, long seed) {
        final int n = checkArrayLength(x);
        final int m = checkArrayLength(y);
        final boolean exactOption = Math.max(n, m) < LARGE_SAMPLE;
        final boolean[] hasTies = exactOption ? new boolean[1] : null;
        double[] sx = x.clone();
        double[] sy = y.clone();
        // Note: The D value is signed
        long dnm = integralKolmogorovSmirnovStatistic(sx, sy, hasTies);
        // XXX - Better control of p-value computation.
        // If there are ties the exact p-value is invalid.
        // max(n, m) < 10000 -> exact; else approximate
        // if ties then exact cannot be used. Either resort to approximate or
        // randomly resolve ties using an input seed.
        // Return if the sample had ties.
        if (exactOption) {
            // XXX:
            // Ideally the statistic is returned with the matching p-value.
            // Thus you cannot compute the statistic separately and the p-value.
            // Tie resolution and uses of the exact p-value should be a choice.
            if (hasTies[0] && dnm != 0) {
                // Make the random pick robust to reversing the samples by ordering
                // based on the signed D value.
                if (dnm < 0) {
                    final double[] tmp = sx;
                    sx = sy;
                    sy = tmp;
                }
                dnm = integralKolmogorovSmirnovStatisticRandomTies(sx, sy, seed);
            }
            dnm = Math.abs(dnm);
            return twoSampleExactP(dnm, n, m, strict);
        }
        dnm = Math.abs(dnm);
        return twoSampleApproximateP(dnm / ((double) n * m), n, m);
    }

    /**
     * Performs a Kolmogorov-Smirnov
     * test evaluating the null hypothesis that {@code x} and {@code y} are samples drawn from the same
     * probability distribution.
     *
     * @param x first sample dataset.
     * @param y second sample dataset.
     * @param strict whether or not the probability to compute is expressed as
     * a strict inequality (ignored for large samples).
     * @param alpha significance level of the test
     * @return true iff the null hypothesis that {@code data} is a sample from {@code distribution}
     *         can be rejected with confidence 1 - {@code alpha}
     * @throws IllegalArgumentException if either {@code x} or {@code y} does
     * not have length at least 2; contain NaN values; or {@code alpha} is not in
     * the range {@code (0, 0.5]}.
     * @see #kolmogorovSmirnovTest(double[], double[], boolean)
     */
    public boolean kolmogorovSmirnovTest(double[] x, double[] y, boolean strict, double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return kolmogorovSmirnovTest(x, y, strict) < alpha;
    }


    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic, \(D_{n,m}=\sup_x |F_n(x)-F_m(x)|\)
     * where \(n\) is the length of {@code x}, \(m\) is the length of {@code y}, \(F_n\) is the
     * empirical distribution that puts mass \(1/n\) at each of the values in {@code x} and \(F_m\)
     * is the empirical distribution of the {@code y} values. Finally \(n m D_{n,m}\) is returned
     * as long value.
     *
     * <p>This method will destructively modify the input arrays (via a sort).
     *
     * @param x First sample (destructively modified by sort).
     * @param y Second sample  (destructively modified by sort).
     * @return test statistic \(n m D_{n,m}\) used to evaluate the null hypothesis that
     *  {@code x} and {@code y} represent samples from the same underlying distribution
     * @throws IllegalArgumentException if either {@code x} or {@code y} contain NaN values.
     */
    private static long integralKolmogorovSmirnovStatistic(double[] x, double[] y) {
        // Convenience pass through method when not interested in ties, or the sign
        return Math.abs(integralKolmogorovSmirnovStatistic(x, y, null));
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic, \(D_{n,m}=\sup_x |F_n(x)-F_m(x)|\)
     * where \(n\) is the length of {@code x}, \(m\) is the length of {@code y}, \(F_n\) is the
     * empirical distribution that puts mass \(1/n\) at each of the values in {@code x} and \(F_m\)
     * is the empirical distribution of the {@code y} values. Finally \(n m D_{n,m}\) is returned
     * as long value.
     *
     * <p>This method will destructively modify the input arrays (via a sort).
     *
     * <p>This method detects ties in the input data; if the {@code hasTies} array is not null
     * then the presence of ties is recorded in the first position of the {@code hasTies} array.
     *
     * <p><strong>Warning: </strong>Note that the statistic is <em>signed</em>. Effectively it is
     * {@code max(|D+|, |D-|)}. If the statistic is zero the two arrays are identical, or
     * are 'identical' data of a single value (sample sizes may be different).
     *
     * @param x First sample (destructively modified by sort).
     * @param y Second sample  (destructively modified by sort).
     * @param hasTies Flag set to true if the input data has ties (null or non-zero length).
     * @return signed test statistic \(n m D_{n,m}\) used to evaluate the null hypothesis that
     * {@code x} and {@code y} represent samples from the same underlying distribution
     * @throws IllegalArgumentException if either {@code x} or {@code y} contain NaN values.
     */
    private static long integralKolmogorovSmirnovStatistic(double[] x, double[] y, boolean[] hasTies) {
        // Sort the sample arrays
        final double[] sx = sort(x, SAMPLE_1_NAME);
        final double[] sy = sort(y, SAMPLE_2_NAME);
        final int n = sx.length;
        final int m = sy.length;

        // CDFs range from 0 to 1 using increments of 1/n and 1/m for x and y respectively.
        // Scale by n*m to use increments of m and n for x and y.

        int rankX = 0;
        int rankY = 0;
        long curD = 0;
        // Used for tie detection. This code requires no branches within the main loop.
        // c and d are decremented by the amount the rank increases.
        // If both are negative then an AND of their values will be negative.
        int c;
        int d;
        int tieBit = 0;

        // Note: Double.compare will not detect -0.0 and 0.0 as a tie.
        // Use a specialised method that identifies NaNs as greater than all other values
        // but makes zeros equal.

        // Find the max difference between cdf_x and cdf_y
        long plus = 0;
        long minus = 0;
        do {
            // No NaN values so compare using <= (not Double.compare)
            final double z = sx[rankX] <= sy[rankY] ? sx[rankX] : sy[rankY];
            c = rankX;
            d = rankY;
            while (rankX < n && sx[rankX] <= z) {
                rankX++;
                curD += m;
            }
            while (rankY < m && sy[rankY] <= z) {
                rankY++;
                curD -= n;
            }
            if (curD > plus) {
                plus = curD;
            } else if (curD < minus) {
                minus = curD;
            }
            // Tie detection. If both ranks advanced set the tieBit to negative.
            tieBit |= (c - rankX) & (d - rankY);
        } while (rankX < n && rankY < m);
        if (hasTies != null) {
            hasTies[0] = tieBit < 0;
        }
        // This returns the signed max difference
        return -minus > plus ? minus : plus;
    }

    /**
     * Computes the two-sample Kolmogorov-Smirnov test statistic,
     * see {@link #integralKolmogorovSmirnovStatistic(double[], double[])}.
     *
     * <p>This method corrects ties between samples. Any sequence of tied values is randomly
     * shuffled and processed as if the randomly ordered tied values were unique.
     *
     * <p>Note: The input arrays are assumed to be sorted with no NaN values.
     * This is for convenience to allow arrays sorted by the
     * {@link #integralKolmogorovSmirnovStatistic(double[], double[], boolean[])}
     * method to be reused when ties are detected.
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
    private static long integralKolmogorovSmirnovStatisticRandomTies(double[] sx, double[] sy, long seed) {
        final int n = sx.length;
        final int m = sy.length;

        int rankX = 0;
        int rankY = 0;
        long curD = 0;

        // Small state size RNG for fast creation
        final UniformRandomProvider rng = new SplittableRandom(seed)::nextLong;

        long plus = 0;
        long minus = 0;
        // Count length of tied data
        int cx;
        int cy;
        do {
            // No NaN values so compare using <= (not Double.compare)
            final double z = sx[rankX] <= sy[rankY] ? sx[rankX] : sy[rankY];
            cx = rankX;
            cy = rankY;
            while (rankX < n && sx[rankX] <= z) {
                rankX++;
            }
            while (rankY < m && sy[rankY] <= z) {
                rankY++;
            }
            // lengths
            cx = rankX - cx;
            cy = rankY - cy;

            // Only cx or cy should be non-zero (not both), else there are ties between samples.
            // Detect using sign bit xor of adjusted counts. If both are above -1
            // the xor will be positive and ties are present.
            if (((cx - 1) ^ (cy - 1)) >= 0) {
                // Resolve tie randomly.
                do {
                    // A negative value picks x, else y.
                    if (rng.nextInt(-cx, cy) < 0) {
                        cx--;
                        curD += m;
                    } else {
                        cy--;
                        curD -= n;
                    }
                    if (curD > plus) {
                        plus = curD;
                    } else if (curD < minus) {
                        minus = curD;
                    }
                } while (((cx - 1) ^ (cy - 1)) >= 0);
            }

            // Here only 1 of cx or cy are non-zero.
            // Handle ties within the same sample using a multiple of the count.
            curD = cx != 0 ?
                curD + (long) cx * m :
                curD - (long) cy * n;
            if (curD > plus) {
                plus = curD;
            } else if (curD < minus) {
                minus = curD;
            }
        } while (rankX < n && rankY < m);
        // This returns the signed max difference
        return -minus > plus ? minus : plus;
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
     * @param iterations Number of bootstrap resampling iterations.
     * @param strict Whether or not the null hypothesis is expressed as a strict inequality.
     * @param rng Source of randomness for sampling.
     * @return the estimated p-value.
     * @throws IllegalArgumentException if either {@code x} or {@code y} does
     * not have length at least 2; contain NaN values; or the number of iterations is
     * not strictly positive.
     */
    public static double estimateP(double[] x, double[] y,
                                   int iterations, boolean strict,
                                   UniformRandomProvider rng) {
        checkArrayLength(x);
        checkArrayLength(y);
        InferenceUtils.checkNonNaN(x);
        InferenceUtils.checkNonNaN(y);
        InferenceUtils.checkStrictlyPositive(iterations);

        // Sample randomly with replacement from the combined distribution.
        final DoubleSupplier gen = createSampler(x, y, rng);

        // Obtain the original sample statistic. Copy the input to avoid modifying it.
        final double[] sx = x.clone();
        final double[] sy = y.clone();
        final long dnm = integralKolmogorovSmirnovStatistic(sx, sy);
        // Test if the random statistic is greater (strict), or greater or equal to d
        final long d = strict ? dnm : dnm - 1;
        int count = 0;
        for (int i = iterations; i > 0; i--) {
            for (int j = 0; j < sx.length; j++) {
                sx[j] = gen.getAsDouble();
            }
            for (int j = 0; j < sy.length; j++) {
                sy[j] = gen.getAsDouble();
            }
            if (integralKolmogorovSmirnovStatistic(sx, sy) > d) {
                count++;
            }
        }
        return count / (double) iterations;
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
     * d)\), where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic. See
     * {@link #kolmogorovSmirnovStatistic(double[], double[])} for the definition of \(D_{n,m}\).
     *
     * <p>The returned probability is exact, implemented using the stabilized inner method
     * presented in [5] (class javadoc).
     *
     * @param dnm Integral D-statistic value (in [0, n*m])
     * @param n first sample size
     * @param m second sample size
     * @param strict whether or not the probability to compute is expressed as a strict inequality
     * @return probability that a randomly selected m-n partition of m + n generates \(D_{n,m}\)
     *         greater than (resp. greater than or equal to) {@code d}
     */
    static double twoSampleExactP(long dnm, int n, int m, boolean strict) {
        // Edge cases
        if (dnm == 0) {
            // Note: This holds for strict inequality as the distribution is based on
            // ordering without ties. In this case d=0 is not possible as there is always
            // a small difference between the empirical CDFs.
            return 1;
        }
        // Support symmetry by computing the same for each order
        return computeP(dnm, Math.max(n, m), Math.min(n, m), strict);
    }

    /**
     * Computes \(P(D_{n,m} &gt; d)\) if {@code strict} is {@code true}; otherwise \(P(D_{n,m} \ge
     * d)\), where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic. See
     * {@link #kolmogorovSmirnovStatistic(double[], double[])} for the definition of \(D_{n,m}\).
     *
     * <p>The returned probability is exact, implemented using the stabilized inner method
     * presented in [5] (class javadoc).
     *
     * @param dnm Integral D-statistic value (in [0, n*m])
     * @param n first sample size
     * @param m second sample size
     * @param strict whether or not the probability to compute is expressed as a strict inequality
     * @return probability that a randomly selected m-n partition of m + n generates \(D_{n,m}\)
     *         greater than (resp. greater than or equal to) {@code d}
     */
    private static double computeP(long dnm, int n, int m, boolean strict) {
        // Adapted from the python listing in Viehmann (2021).
        // That used d in [0, 1]. This uses dnm in [0, nm] so updates the scaling to
        // compute the ranges.
        // The provided listing is explicit in the values for each j in the range.
        // It can be optimised given the known start and end j for each iteration.
        // Here we extract the i=0 and j=0 parts before the loops which use i>0 and j>0.
        // The loop over j is split as the conditions

        // size = int(2*m*d + 2)
        int size = (int) Math.ceil(2.0 * dnm / n + 2);
        // Only require 1 array as the startJ only ever increases
        // and we update lower indices using higher ones.
        // The maximum value addressed is row[m].
        size = Math.min(size, m + 1);
        final double[] row = new double[size];
        final long d = strict ? dnm : dnm - 1;
        // First iteration with i = 0
        // if j/m > d: row[j]=1, else 0.
        // j/m > d => j*n > dnm
        for (int j = (int) (d / n) + 1; j < size; j++) {
            row[j] = 1;
        }
        int lastStartJ = 0;
        double val;
        int j;
        long im = 0;
        for (int i = 1; i <= n; i++) {
            im += m;
            // startJ = int(m * (i/n + d)) + 1-size
            // => int(mi/n + dnm/n) + 1 - int(2*dnm/n + 2)
            // => int(mi/n + dnm/n) - int(2*dnm/n) - 1
            // If this is too low the loop just does more work.
            int startJ = Math.max((int) ((im - dnm) / n) - 1, 0);
            // j is limited to the range [0, m]
            startJ = Math.min(startJ, m);

            // First iteration with jj = 0. j = startJ.
            // Note: Two conditions can be dropped:
            // 1. max(startJ - lastStartJ) = ceil(m/n)
            // => j - lastStartJ >= size is not possible
            // 2. if startJ > 0 it occurs when
            // (j+1)*n <= im - dnm => im - j*n > dnm
            // if |im - jn| <= dnm then j must be 0 and val=0.
            // Cannot check j==0 as im > d is possible.
            long jn = startJ * (long) n;
            // dist = |i/n - j/m| => |i*m - j*n|
            val = Math.abs(im - jn) > d ? 1 : 0;
            row[0] = val;

            // Remaining loop omits check for i=0 or j=0
            // Only loop over indices that address the row.
            final int max = size + lastStartJ - startJ;
            for (int jj = 1; jj < max; jj++) {
                j = jj + startJ;
                jn += n;
                // dist = |i/n - j/m| => |i*m - j*n|
                if (Math.abs(im - jn) > d) {
                    val = 1;
                // else if j==0:                       [Skip as j > 0]
                //   val = 0
                // else if (j - lastStartJ >= size):   [Skip due to max]
                //   val = (i + val * j) / ((double) i + j)
                } else {
                    val = (row[j - lastStartJ] * i + val * j) / ((double) i + j);
                }
                row[jj] = val;
            }
            // Fill in the remaining values for (j - lastStartJ >= size)
            for (int jj = max; jj < size; jj++) {
                j = jj + startJ;
                jn += n;
                if (Math.abs(im - jn) > d) {
                    // val=1 and the remaining values are all 1
                    Arrays.fill(row, jj, size, 1);
                    break;
                }
                val = (i + val * j) / ((double) i + j);
                row[jj] = val;
            }
            lastStartJ = startJ;
        }
        // Code lists row[m - startJ] as Python scopes the variable inside the loop to the method
        return row[m - lastStartJ];
    }

    /**
     * Uses the Kolmogorov-Smirnov distribution to approximate \(P(D_{n,m} &gt; d)\) where \(D_{n,m}\)
     * is the 2-sample Kolmogorov-Smirnov statistic. See
     * {@link #kolmogorovSmirnovStatistic(double[], double[])} for the definition of \(D_{n,m}\).
     *
     * <p>Specifically, what is returned is \(1 - CDF(d, \sqrt{mn / (m + n)})\) where CDF
     * is the cumulative density function of the two-sided one-sample Kolmogorov-Smirnov
     * distribution.
     *
     * @param d D-statistic value.
     * @param n First sample size.
     * @param m Second sample size.
     * @return approximate probability that a randomly selected m-n partition of m + n generates
     *         \(D_{n,m}\) greater than {@code d}
     */
    static double twoSampleApproximateP(double d, int n, int m) {
        // Smirnov's asymptotic formula:
        // P(sqrt(N) D_n > x) = 2 \sum_{i=1}^\infty (-1)^(i-1) e^{-2 i^2 x^2}
        // x^2 = N * d * d
        // N = m*n/(m+n)
        // Comparison of exactP(d*n*m, n, m, false) with Two.sf or an implementation of
        // the KS sum over a range of N and d where the p-value is a typical alpha threshold
        // of 0.001 to 0.1 shows that the Two.sf has lower RMSD relative error except when N is
        // very small (e.g. 4); here neither approximate p-value is close to the exact P.
        return KolmogorovSmirnovDistribution.Two.sf(d,
            (int) Math.round(((double) m * n) / ((double) m + n)));
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
     * @return a refercen to the input (sorted) array
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
}
