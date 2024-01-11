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
package org.apache.commons.statistics.examples.jmh.descriptive;

import java.util.Arrays;
import java.util.Objects;

/**
 * Provides quantile computation.
 *
 * <p>For values of length {@code n}:
 * <ul>
 * <li>The result is {@code NaN} if {@code n = 0}.
 * <li>The result is {@code values[0]} if {@code n = 1}.
 * <li>Otherwise the result is computed using the {@link EstimationType}.
 * </ul>
 *
 * <p>This implementation respects the ordering imposed by
 * {@link Double#compare(double, double)} for {@code NaN} values. If a {@code NaN} occurs
 * in the selected positions in the fully sorted values then the result is {@code NaN}.
 *
 * @see java.util.Arrays#sort(double[])
 * @since 1.1
 */
public final class Quantile {
    // Implementation note:
    //
    // This class has been adapted from Commons Math Percentile which exposes the
    // index selection API methods. These are now intentionally not public.
    //
    // This class allows multiple quantiles to be provided in a single call.
    // The current implementation performs selection of multiple quantiles using
    // repeated calls for each index to a partitioning algorithm. This can use
    // known pivot points (correctly sorted indices) in repeat calls.
    //
    // A future implementation may use a partition algorithm that accepts all the
    // indices to select as an argument, e.g.
    //   double[] part = partition(double[] array, int[] indices, boolean inPlace)
    // This class would then require identification of all required indices before
    // the call. To allow this change the classes performing the partitioning and
    // the estimation methods for index selection are not public.

    /** Message when no quantiles are provided for the varargs method. */
    private static final String NO_QUANTILES_SPECIFIED = "No quantiles specified";

    /** Default instance.
     * Note: Numpy and R use method 7 as default. Method 8 is recommended by Hyndman and Fan. */
    private static final Quantile DEFAULT = new Quantile(false, new KthSelector(),
        new Partition(), EstimationType.HF8);

    /** Flag to indicate if the data should be overwritten. */
    private final boolean overwrite;
    /** Selector for the K-th element in an array. */
    private final KthSelector kthSelector;
    /** Partition method for partial sort of an array. */
    private final Partition partition;
    /** Estimation type used to determine the value from the quantile. */
    private final EstimationType estimationType;

    /**
     * Partition function. Used to decouple the partition of data around indices
     * and the computation of quantiles from ordered data points.
     */
    private interface PartitionFunction {
        /**
         * Partition the array such that indices {@code k} correspond to their correctly
         * sorted value in the equivalent fully sorted array. For all indices {@code k}
         * and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * <p>Uses a single-pivot partition method.
         *
         * @param data Values.
         * @param k Indices.
         */
        void partition(double[] data, int... k);
    }

    /**
     * @param overwrite Flag to indicate if the data should be overwritten.
     * @param kthSelector Selector for the K-th element in an array.
     * @param partition Partition method for partial sort of an array.
     * @param estimationType Estimation type used to determine the value from the quantile.
     */
    private Quantile(boolean overwrite, KthSelector kthSelector,
            Partition partition, EstimationType estimationType) {
        this.overwrite = overwrite;
        this.kthSelector = kthSelector;
        this.partition = partition;
        this.estimationType = estimationType;
    }

    /**
     * Return a new instance with the default options.
     *
     * <ul>
     * <li>{@linkplain #withOverwrite(boolean) Overwrite = false}
     * <li>{@linkplain #withEstimationType(EstimationType) Estimation type = HF8}
     * </ul>
     *
     * @return the quantile
     */
    public static Quantile withDefaults() {
        return DEFAULT;
    }

    /**
     * Return an instance with the configured overwrite behaviour. If {@code true} then
     * the input array will be modified by the call to
     *
     * @param v Value.
     * @return an instance
     */
    public Quantile withOverwrite(boolean v) {
        return new Quantile(v, kthSelector, partition, estimationType);
    }

    /**
     * Return an instance with the configured {@link KthSelector}.
     *
     * <p>Note: This is not part of the public API. It is exposed for performance testing.
     *
     * @param v Value.
     * @return an instance
     */
    Quantile withKthSelector(KthSelector v) {
        return new Quantile(overwrite, Objects.requireNonNull(v), partition, estimationType);
    }

    /**
     * Return an instance with the configured {@link Partition}.
     *
     * <p>Note: This is not part of the public API. It is exposed for performance testing.
     *
     * @param v Value.
     * @return an instance
     */
    Quantile withPartition(Partition v) {
        return new Quantile(overwrite, kthSelector, Objects.requireNonNull(v), estimationType);
    }

    /**
     * Return an instance with the configured overwrite behaviour. If {@code true} then
     * the input array will be modified by the call to
     *
     * @param v Value.
     * @return an instance
     */
    public Quantile withEstimationType(EstimationType v) {
        return new Quantile(overwrite, kthSelector, partition, Objects.requireNonNull(v));
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot partition method with a heap to cache pivots points.
     * Same as the method in Commons Math without the unnecessary heap for a single pass.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSP(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateSPH(double[] values, double p) {
        // Implicit NPE
        final int n = values.length;
        checkQuantile(p);
        // Special cases
        if (n <= 1) {
            return n == 0 ? Double.NaN : values[0];
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        // No pivot heap required for a single pass
        final int[] pivotHeap = KthSelector.NO_PIVOTS;
        return estimationType.evaluate(x, pivotHeap, kthSelector, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot partition method with a heap to cache pivots points.
     * Same as the method in Commons Math (but without the unnecessary heap for a single pass).
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateSPH(double[] values, double... p) {
        // Implicit NPE
        final int n = values.length;
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Special cases
        final double[] q = new double[p.length];
        if (n <= 1) {
            Arrays.fill(q, n == 0 ? Double.NaN : values[0]);
            return q;
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        // No pivot heap required for a single pass
        final int[] pivotHeap = p.length == 1 ?
            KthSelector.NO_PIVOTS :
            KthSelector.createPivotsHeap(n);
        for (int i = 0; i < p.length; i++) {
            q[i] = estimationType.evaluate(x, pivotHeap, kthSelector, p[i]);
        }
        return q;
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot partition method. Estimation is coupled to the
     * {@link EstimationType} enum and requires a double[] data type.
     * Variant of the method in Commons Math without the unnecessary heap for a single pass.
     * The data coupling is undesirable.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateSPE(double[] values, double... p) {
        // Implicit NPE
        final int n = values.length;
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Special cases
        final double[] q = new double[p.length];
        if (n <= 1) {
            Arrays.fill(q, n == 0 ? Double.NaN : values[0]);
            return q;
        }
        if (p.length == 1) {
            // Faster to do a single partition
            return new double[] {evaluateSP(values, p[0])};
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        // Collect indices (duplicates are ignored by the partition algorithm)
        final int[] indices = new int[p.length * 2];
        int count = 0;
        for (int i = 0; i < p.length; i++) {
            count = estimationType.evaluateIndices(p[i], n, indices, count);
        }
        // Partition
        kthSelector.partitionSP(x, Arrays.copyOf(indices, count));
        // Compute
        for (int i = 0; i < p.length; i++) {
            // XXX: EstimationType coupled to the type double[]
            q[i] = estimationType.evaluatePartitioned(x, p[i]);
        }
        return q;
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSP(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param part Partition function.
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    private double evaluate(PartitionFunction part, double[] values, double p) {
        // Implicit NPE
        final int n = values.length;
        checkQuantile(p);
        // Special cases
        if (n <= 1) {
            return n == 0 ? Double.NaN : values[0];
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();

        final double[] g = {0};
        final int i = estimationType.index(p, n, g);

        // Partition and compute
        if (g[0] != 0) {
            part.partition(x, i, i + 1);
            return DoubleMath.interpolate(x[i], x[i + 1], g[0]);
        }
        part.partition(x, i);
        return x[i];
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * @param part Partition function.
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluate(PartitionFunction part, double[] values, double... p) {
        // Implicit NPE
        final int n = values.length;
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Special cases
        final double[] q = new double[p.length];
        if (n <= 1) {
            Arrays.fill(q, n == 0 ? Double.NaN : values[0]);
            return q;
        }

        // A sort is required
        final double[] x = overwrite ? values : values.clone();

        // Collect interpolation positions. We use the output q to store factors.
        final int[] indices = new int[p.length * 2];
        final double[] g = {0};
        int count = p.length;
        for (int k = 0; k < p.length; k++) {
            final int i = estimationType.index(p[k], n, g);
            indices[k] = i;
            if (g[0] != 0) {
                // Require the next index for interpolation
                indices[count++] = i + 1;
                q[k] = g[0];
            }
        }

        // Partition
        part.partition(x, Arrays.copyOf(indices, count));

        // Compute
        for (int k = 0; k < p.length; k++) {
            int i = indices[k];
            if (q[k] != 0) {
                q[k] = DoubleMath.interpolate(x[i], x[i + 1], q[k]);
            } else {
                q[k] = x[i];
            }
        }
        return q;
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot partition method.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSP(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateSP(double[] values, double p) {
        return evaluate(kthSelector::partitionSP, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot partition method.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateSP(double[] values, double... p) {
        return evaluate(kthSelector::partitionSP, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateBM(double[] values, double p) {
        return evaluate(kthSelector::partitionBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateBM(double[] values, double... p) {
        return evaluate(kthSelector::partitionBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method from Sedgewick.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateSBM(double[] values, double p) {
        return evaluate(kthSelector::partitionSBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method from Sedgewick.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateSBM(double[] values, double... p) {
        return evaluate(kthSelector::partitionSBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateDP(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateDP(double[] values, double p) {
        return evaluate(kthSelector::partitionDP, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateDP(double[] values, double... p) {
        return evaluate(kthSelector::partitionDP, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateDP5(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateDP5(double[] values, double p) {
        return evaluate(kthSelector::partitionDP5, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort method by Vladimir Yaroslavskiy.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateDP5(double[] values, double... p) {
        return evaluate(kthSelector::partitionDP5, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method from Sedgewick.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateSBM2(double[] values, double p) {
        return evaluate(partition::partitionSBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method from Sedgewick.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateSBM2(double[] values, double... p) {
        return evaluate(partition::partitionSBM, values, p);
    }

    /**
     * Check the quantile {@code p} is in the range {@code [0, 1]}.
     *
     * @param p Quantile.
     * @throws IllegalArgumentException if the quantile is not in the range {@code [0, 1]}
     */
    private static void checkQuantile(double p) {
        if (!(p >= 0 && p <= 1)) {
            throw new IllegalArgumentException("Invalid quantile: " + p);
        }
    }

    /**
     * Estimation strategies for a quantile. Provides the nine quantile algorithms
     * defined in Hyndman and Fan (1996)[1] as {@code HF1 - HF9}.
     *
     * <p>Samples quantiles are defined by:
     *
     * <p>\[ Q(p) = (1 - \gamma) x_j + \gamma x_{j+1} \]
     *
     * <p>where \( \frac{j-m}{n} \leq p \le \frac{j-m+1}{n}, \( x_j \) is the \( j \)th
     * order statistic, \( n \) is the sample size, the value of \( \gamma \) is a function
     * of \( j = \lfloor np+m \rfloor \) and \( g = np + m - j \), and \( m \) is a constant
     * determined by the sample quantile type.
     *
     * <p>Note that the real valued position \( np + m \) is a 1-based index and
     * \( j \in [1, n] \). If the real valued position is computed as beyond the lowest or
     * highest values in the sample, this implementation will return the minimum or maximum
     * observation respectively.
     *
     * <p>Types 1, 2, and 3 are discontinuous functions of \( p \); types 4 to 9 are continuous
     * functions of \( p \).
     *
     * <ol>
     * <li>Hyndman and Fan (1996)
     *     <i>Sample Quantiles in Statistical Packages.</i>
     *     The American Statistician, 50, 361-365.
     *     <a href="https://www.jstor.org/stable/2684934">doi.org/10.2307/2684934</a>
     * <li><a href="http://en.wikipedia.org/wiki/Quantile">Quantile (Wikipedia)</a>
     * </ol>
     */
    public enum EstimationType {
        /**
         * Inverse of the empirical distribution function.
         *
         * <p>\( m = 0 \); \( \gamma = 0 \) if \( g = 0 \), and 1 otherwise.
         */
        HF1 {
            @Override
            double position(double p, int n) {
                return n * p;
            }

            @Override
            int toInterpolation(int j, double alpha, double[] g) {
                // ceil(j + alpha) : note j is 1-based
                return alpha == 0 ? j - 1 : j;
            }

            @Override
            int estimateIndices(int j, double g, int n, int[] indices, int count) {
                // ceil(j + g) : note j is 1-based
                final int i = g == 0 ? j - 1 : j;
                indices[count] = i;
                return count + 1;
            }

            @Override
            double estimatePartitioned(double[] values, int j, double g, int n) {
                final int i = g == 0 ? j - 1 : j;
                return values[i];
            }

            @Override
            double estimate(double[] values, int[] pivotsHeap, int j, double g, int n, KthSelector selector) {
                // ceil(j + g) : note j is 1-based
                final int i = g == 0 ? j - 1 : j;
                return selector.selectSPH(values, pivotsHeap, i, null);
            }
        },
        /**
         * Similar to {@link #HF1} with averaging at discontinuities.
         *
         * <p>\( m = 0 \); \( \gamma = 0.5 \) if \( g = 0 \), and 1 otherwise.
         */
        HF2 {
            @Override
            double position(double p, int n) {
                return n * p;
            }

            @Override
            int toInterpolation(int j, double alpha, double[] g) {
                if (alpha == 0) {
                    // Average at discontinuities
                    g[0] = 0.5;
                    return j - 1;
                }
                // As HF1 : ceil(j + alpha)
                return j;
            }

            @Override
            int estimateIndices(int j, double g, int n, int[] indices, int count) {
                if (g == 0) {
                    // Average at discontinuities
                    return super.estimateIndices(j, 0.5, n, indices, count);
                }
                // As HF1 : ceil(j + g)
                indices[count] = j;
                return count + 1;
            }

            @Override
            double estimatePartitioned(double[] values, int j, double g, int n) {
                if (g == 0) {
                    return super.estimatePartitioned(values, j, 0.5, n);
                }
                return values[j];
            }

            @Override
            double estimate(double[] values, int[] pivotsHeap, int j, double g, int n, KthSelector selector) {
                if (g == 0) {
                    // Average at discontinuities
                    return super.estimate(values, pivotsHeap, j, 0.5, n, selector);
                }
                // As HF1 : ceil(j + g)
                return selector.selectSPH(values, pivotsHeap, j, null);
            }
        },
        /**
         * The observation closest to \( np \). Ties are resolved to the nearest even order statistic.
         *
         * <p>\( m = -1/2 \); \( \gamma = 0 \) if \( g = 0 \) and \( j \) is even, and 1 otherwise.
         */
        HF3 {
            @Override
            double position(double p, int n) {
                // Let rint do the work for ties to even
                return Math.rint(n * p);
            }
        },
        /**
         * Linear interpolation of the inverse of the empirical CDF.
         *
         * <p>\( m = 0 \).
         */
        HF4 {
            @Override
            double position(double p, int n) {
                return n * p;
            }
        },
        /**
         * A piecewise linear function where the knots are the values midway through the steps of
         * the empirical CDF. Proposed by Hazen (1914) and popular amongst hydrologists.
         *
         * <p>\( m = 1/2 \).
         */
        HF5 {
            @Override
            double position(double p, int n) {
                return n * p + 0.5;
            }
        },
        /**
         * Linear interpolation of the expectations for the order statistics for the uniform
         * distribution on [0,1]. Proposed by Weibull (1939).
         *
         * <p>\( m = p \).
         *
         * <p>This method computes the quantile as per the Apache Commons Math Percentile
         * legacy implementation.
         */
        HF6 {
            @Override
            double position(double p, int n) {
                return (n + 1) * p;
            }
        },
        /**
         * Linear interpolation of the modes for the order statistics for the uniform
         * distribution on [0,1]. Proposed by Gumbull (1939).
         *
         * <p>\( m = 1 - p \).
         */
        HF7 {
            @Override
            double position(double p, int n) {
                return (n - 1) * p + 1;
            }
        },
        /**
         * Linear interpolation of the approximate medians for order statistics.
         *
         * <p>\( m = (p + 1)/3 \).
         *
         * <p>As per Hyndman and Fan (1996) this approach is most recommended as it provides
         * an approximate median-unbiased estimate regardless of distribution.
         */
        HF8 {
            @Override
            double position(double p, int n) {
                return n * p + (p + 1) / 3;
            }
        },
        /**
         * Quantile estimates are approximately unbiased for the expected order statistics if
         * \( x \) is normally distributed.
         *
         * <p>\( m = p/4 + 3/8 \).
         */
        HF9 {
            @Override
            double position(double p, int n) {
                return (n + 0.25) * p + 3d / 8;
            }
        };

        /**
         * Finds the real valued position for calculation of the quantile.
         *
         * @param p p<sup>th</sup> quantile.
         * @param n Size.
         * @return a real valued position (1-based) into the range {@code [1, n]}
         */
        abstract double position(double p, int n);

        /**
         * Finds the index {@code i} and fractional part {@code g} of a real valued position
         * to interpolate the quantile.
         *
         * <p>Given a real valued position in {@code [1, n]} obtained using the estimation
         * type from the p<sup>th</sup> quantile and size {@code n}:
         *
         * <p>position = j + alpha
         *
         * <p>Return {@code i + g} such that the quantile value from sorted data is:
         *
         * <p>value = data[i] + g * (data[i+1] - data[i])
         *
         * <p>The default implementation returns:
         * <ul>
         * <li>if {@code j < 1}: {@code i = 0; g = 0}
         * <li>if {@code j >= n}: {@code i = n - 1; g = 0}
         * <li>else {@code i = j - 1}: {@code g = alpha}
         * </ul>
         *
         * <p>Note: Interpolation should not use {@code data[i+1]} unless {@code g != 0}.
         *
         * @param p p<sup>th</sup> quantile.
         * @param n Size.
         * @param g Fractional part.
         * @return index
         */
        final int index(double p, int n, double[] g) {
            // Implicit NPE
            g[0] = 0;
            final double pos = position(p, n);
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double alpha = pos - fpos;
            // Bounds check. Note that j is 1-based.
            if (j < 1) {
                return 0;
            }
            if (j >= n) {
                return n - 1;
            }
            return toInterpolation(j, alpha, g);
        }

        /**
         * Convert the real-valued index {@code pos = j + alpha} {@code [1, n]} obtained using
         * {@link #position(double, int)} to an index {@code i} and factor {@code g} to
         * interpolation the p<sup>th</sup> quantile:
         *
         * <p>This method is only called when {@code 1 <= j < n}. The fractional part {@code alpha}
         * may be zero, i.e. {@code 0 <= alpha < 1}. The method is called with {@code g}
         * initialised to zero.
         *
         * <p>The default implementation return {@code j - 1} and {@code g = alphaj}.
         * This may be overridden to compute slightly different estimations.
         *
         * @param j Integer part of position.
         * @param alpha Fractional part of position.
         * @param g Output fractional part.
         * @return output index
         */
        int toInterpolation(int j, double alpha, double[] g) {
            g[0] = alpha;
            return j - 1;
        }

        /**
         * Finds the array indices required for calculation of the quantile
         * using the {@link #evaluatePartitioned(double[], double)} method.
         * Writes the indices to the provided array at the given {@code count}
         * and returns the new {@code count}.
         *
         * <p>This method should compute either one index {@code j}, or a pair of adjacent indices
         * {@code (j, j+1)} for use in linear interpolation of the real valued position index.
         *
         * <p>The default implementation calls {@link #position(double, int)} to obtain the position
         * {@code pos} and records {@code j = floor(pos) - 1}. Adds {@code j+1} if
         * {@code pos - j - 1} is non-zero.
         *
         *
         * @param p p<sup>th</sup> quantile.
         * @param n Size.
         * @param indices Output indices.
         * @param count Current indices count.
         * @return new indices count
         */
        final int evaluateIndices(double p, int n, int[] indices, int count) {
            return estimateIndices(position(p, n), n, indices, count);
        }

        /**
         * Finds the array indices required for calculation of the quantile
         * using the {@link #estimatePartitioned(double[], double, int)} method.
         * Writes the indices to the provided array at the given {@code count}
         * and returns the new {@code count}.
         *
         * <p>This method should compute either one index {@code j}, or a pair of adjacent indices
         * {@code (j, j+1)} for use in linear interpolation of the real valued position index.
         *
         * <p>The default implementation calls {@link #position(double, int)} to obtain the position
         * {@code pos} and records {@code j = floor(pos) - 1}. Adds {@code j+1} if
         * {@code pos - j - 1} is non-zero.
         *
         * @param pos Positional index prior computed from calling {@link #position(double, int)}.
         * @param n Size.
         * @param indices Output indices.
         * @param count Current indices count.
         * @return new indices count
         */
        final int estimateIndices(double pos, int n, int[] indices, int count) {
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double g = pos - fpos;
            int c = count;
            // Bounds check. Note that j is 1-based.
            if (j < 1) {
                indices[c++] = 0;
            } else if (j >= n) {
                indices[c++] = n - 1;
            } else {
                return estimateIndices(j, g, n, indices, count);
            }
            return c;
        }

        /**
         * Finds the array indices required for calculation of the quantile
         * using the {@link #estimatePartitioned(double[], int, double, int)} method.
         * Writes the indices to the provided array at the given {@code count}
         * and returns the new {@code count}.
         *
         * <p>This method should compute either one index {@code j}, or a pair of adjacent indices
         * {@code (j, j+1)} for use in linear interpolation of the real valued position index.
         *
         * <p>Uses the position index from {@link #position(double, int)}:
         * <pre>
         * j = floor(position)
         * g = position - j
         * </pre>
         *
         * <p>This method is only called when {@code 1 <= j < n}. The fractional part {@code g}
         * may be zero, i.e. {@code 0 <= g < 1}.
         *
         * <p>The default implementation records {@code j - 1} and adds {@code j} if
         * {@code g} is non-zero. This may be overridden in specific
         * enums to compute slightly different estimations.
         *
         * @param j Integer part of the position index.
         * @param g Fractional part of the position index.
         * @param n Size.
         * @param indices Output indices.
         * @param count Current indices count.
         * @return new indices count
         */
        int estimateIndices(int j, double g, int n, int[] indices, int count) {
            // Note position index is 1-based:
            int c = count;
            indices[c++] = j - 1;
            if (g != 0) {
                indices[c++] = j;
            }
            return c;
        }

        /**
         * Evaluate method to compute the quantile.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #evaluateIndices(double, int, int[], int)} method.
         *
         * @param values Values.
         * @param p p<sup>th</sup> quantile to be computed
         * @return estimated quantile
         */
        final double evaluatePartitioned(double[] values, double p) {
            final int n = values.length;
            return estimatePartitioned(values, position(p, n), n);
        }

        /**
         * Estimation based on partitioned values.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #estimateIndices(double, int, int[], int)} method.
         *
         * @param values Values.
         * @param pos Positional index prior computed from calling {@link #position(double, int)}.
         * @param n Size.
         * @return estimated quantile
         */
        final double estimatePartitioned(double[] values, double pos, int n) {
            // position = j + g = pn + m
            // j = floor(pn + m)
            // g = pn + m - j
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double g = pos - fpos;
            // Bounds check. Note that j is 1-based.
            if (j < 1) {
                return values[0];
            }
            if (j >= n) {
                return values[n - 1];
            }
            return estimatePartitioned(values, j, g, n);
        }

        /**
         * Estimation based on partitioned values.
         * Uses the position index from {@link #position(double, int)}:
         * <pre>
         * j = floor(position)
         * g = position - j
         * </pre>
         *
         * <p>This method is only called when {@code 1 <= j < n}. The fractional part {@code g}
         * may be zero, i.e. {@code 0 <= g < 1}.
         *
         * <p>The default implementation provides a continuous function using linear interpolation
         * between the value at {@code j} and {@code j+1}. This may be overridden in specific
         * enums to compute slightly different estimations.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #estimateIndices(int, double, int, int[], int)} method.
         *
         * @param values Values.
         * @param j Integer part of the position index.
         * @param g Fractional part of the position index.
         * @param n Size.
         * @return estimated quantile
         */
        double estimatePartitioned(double[] values, int j, double g, int n) {
            // Note position index is 1-based:
            if (g == 0) {
                // No interpolation
                return values[j - 1];
            }
            return DoubleMath.interpolate(values[j - 1], values[j], g);
        }

        /**
         * Evaluate method to compute the quantile.
         *
         * @param values Values.
         * @param pivotsHeap Cache of sort pivot points.
         * @param p p<sup>th</sup> quantile to be computed
         * @param selector {@link KthSelector} used for pivoting during search
         * @return estimated quantile
         */
        final double evaluate(double[] values, int[] pivotsHeap, KthSelector selector, double p) {
            final int n = values.length;
            return estimate(values, pivotsHeap, position(p, n), n, selector);
        }

        /**
         * Estimation based on K<sup>th</sup> selection.
         *
         * @param values Values.
         * @param pos Positional index prior computed from calling {@link #position(double, int)}.
         * @param pivotsHeap Cache of sort pivot points.
         * @param n Size.
         * @param selector {@link KthSelector} used for pivoting during search
         * @return estimated quantile
         */
        final double estimate(double[] values, int[] pivotsHeap, double pos, int n, KthSelector selector) {
            // position = j + g = pn + m
            // j = floor(pn + m)
            // g = pn + m - j
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double g = pos - fpos;
            // Bounds check. Note that j is 1-based.
            if (j < 1) {
                return selector.selectSPH(values, pivotsHeap, 0, null);
            }
            if (j >= n) {
                return selector.selectSPH(values, pivotsHeap, n - 1, null);
            }
            return estimate(values, pivotsHeap, j, g, n, selector);
        }

        /**
         * Estimation based on K<sup>th</sup> selection.
         * Uses the position index from {@link #position(double, int)}:
         * <pre>
         * j = floor(position)
         * g = position - j
         * </pre>
         *
         * <p>This method is only called when {@code 1 <= j < n}. The fractional part {@code g}
         * may be zero, i.e. {@code 0 <= g < 1}.
         *
         * <p>The default implementation provides a continuous function using linear interpolation
         * between the value at {@code j} and {@code j+1}. This may be overridden in specific
         * enums to compute slightly different estimations.
         *
         * @param values Values.
         * @param pivotsHeap Cache of sort pivot points.
         * @param j Integer part of the position index.
         * @param g Fractional part of the position index.
         * @param n Size.
         * @param selector {@link KthSelector} used for pivoting during search
         * @return estimated quantile
         */
        double estimate(double[] values, int[] pivotsHeap, int j, double g, int n, KthSelector selector) {
            // Note position index is 1-based:
            if (g == 0) {
                // No interpolation
                return selector.selectSPH(values, pivotsHeap, j - 1, null);
            }
            final double[] upper = {0};
            final double lower = selector.selectSPH(values, pivotsHeap, j - 1, upper);
            return DoubleMath.interpolate(lower, upper[0], g);
        }
    }
}
