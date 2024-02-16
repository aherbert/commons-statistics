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
import java.util.function.Supplier;

/**
 * Provides quantile computation.
 *
 * <p>For values of length {@code n}:
 * <ul>
 * <li>The result is {@code NaN} if {@code n = 0}.
 * <li>The result is {@code values[0]} if {@code n = 1}.
 * <li>Otherwise the result is computed using the {@link EstimationMethod}.
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
    // This class has been adapted from Commons Math (CM) Percentile which exposes the
    // index selection API methods. These are now intentionally not public.
    // It contains variant implementations of the method in CM, and others developed
    // during testing.
    //
    // This class allows multiple quantiles to be provided in a single call.
    //
    // Some implementations perform selection of multiple quantiles using
    // repeated calls for each index to a partitioning algorithm. This can use
    // known pivot points (correctly sorted indices) in repeat calls.
    // The data at each pivot may be rearranged by later calls if the store of
    // pivots is saturated. This is an issue if you wish to perform partitioning
    // in-place with a large number of pivots, e.g. partition an array using 100
    // linearly spaced quantiles.
    //
    // Alternative implementations use a partition algorithm that accepts all the
    // indices to select as an argument.
    // This class identifies all required indices before the call. The partition
    // algorithm ensures the returned data is correctly partitioned on the indices.
    //
    // To allow changes to the classes performing the partitioning and
    // the estimation, methods for index selection are not public.

    /** Message when no quantiles are provided for the varargs method. */
    private static final String NO_QUANTILES_SPECIFIED = "No quantiles specified";

    /** Default instance.
     * Note: Numpy and R use method 7 as default. Method 8 is recommended by Hyndman and Fan. */
    private static final Quantile DEFAULT = new Quantile(false, NaNPolicy.INCLUDE,
        new KthSelector(), new Partition(), EstimationMethod.HF8);

    /** Flag to indicate if the data should be overwritten. */
    private final boolean overwrite;
    /** NaN policy for floating point data. */
    private final NaNPolicy nanPolicy;
    /** Transformer factory for double data. */
    private final Supplier<DoubleDataTransformer> transformer;
    /** Selector for the K-th element in an array. */
    private final KthSelector kthSelector;
    /** Partition method for partial sort of an array. */
    private final Partition partition;
    /** Estimation type used to determine the value from the quantile. */
    private final EstimationMethod estimationType;

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
         * @param data Values.
         * @param k Indices.
         */
        void partition(double[] data, int... k);
    }

    /**
     * Partition function. Used to decouple the partition of data around indices
     * and the computation of quantiles from ordered data points.
     */
    private interface PartitionFunction2 {
        /**
         * Partition the array such that indices {@code k} correspond to their correctly
         * sorted value in the equivalent fully sorted array. For all indices {@code k}
         * and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * @param data Values.
         * @param k Indices (may be destructively modified).
         * @param n Count of indices.
         */
        void partition(double[] data, int[] k, int n);
    }

    /**
     * Partition function. Used to decouple the partition of data around indices
     * and the computation of quantiles from ordered data points.
     *
     * <p>The function is not required to handle NaN or signed zeros.
     */
    private interface PartitionFunction3 {
        /**
         * Partition the array such that indices {@code k} correspond to their correctly
         * sorted value in the equivalent fully sorted array. For all indices {@code k}
         * and any index {@code i}:
         *
         * <pre>{@code
         * data[i < k] <= data[k] <= data[k < i]
         * }</pre>
         *
         * @param data Values.
         * @param length Length of the data.
         * @param k Indices (may be destructively modified).
         * @param n Count of indices.
         */
        void partition(double[] data, int length, int[] k, int n);
    }

    /**
     * Instantiates a new quantile.
     *
     * @param overwrite Flag to indicate if the data should be overwritten.
     * @param nanPolicy NaN policy.
     * @param kthSelector Selector for the K-th element in an array.
     * @param partition Partition method for partial sort of an array.
     * @param estimationType Estimation type used to determine the value from the quantile.
     */
    private Quantile(boolean overwrite, NaNPolicy nanPolicy, KthSelector kthSelector,
            Partition partition, EstimationMethod estimationType) {
        this.overwrite = overwrite;
        this.nanPolicy = nanPolicy;
        this.kthSelector = kthSelector;
        this.partition = partition;
        this.estimationType = estimationType;
        transformer = DoubleDataTransformers.createFactory(nanPolicy, !overwrite);
    }

    /**
     * Return a new instance with the default options.
     *
     * <ul>
     * <li>{@linkplain #withOverwrite(boolean) Overwrite = false}
     * <li>{@linkplain #with(NaNPolicy) NaN policy = include}
     * <li>{@linkplain #with(EstimationMethod) Estimation method = HF8}
     * </ul>
     *
     * @return the quantile
     */
    public static Quantile withDefaults() {
        return DEFAULT;
    }

    /**
     * Return an instance with the configured overwrite behaviour. If {@code true} then
     * the input array will be modified by the call to evaluate quantiles.
     *
     * @param v Value.
     * @return an instance
     */
    public Quantile withOverwrite(boolean v) {
        return new Quantile(v, nanPolicy, kthSelector, partition, estimationType);
    }

    /**
     * Return an instance with the configured {@link NaNPolicy}.
     *
     * @param v Value.
     * @return an instance
     */
    public Quantile with(NaNPolicy v) {
        return new Quantile(overwrite, Objects.requireNonNull(v), kthSelector, partition, estimationType);
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
        return new Quantile(overwrite, nanPolicy, Objects.requireNonNull(v), partition, estimationType);
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
        return new Quantile(overwrite, nanPolicy, kthSelector, Objects.requireNonNull(v), estimationType);
    }

    /**
     * Return an instance with the configured {@link EstimationMethod}.
     *
     * @param v Value.
     * @return an instance
     */
    public Quantile with(EstimationMethod v) {
        return new Quantile(overwrite, nanPolicy, kthSelector, partition, Objects.requireNonNull(v));
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
     * {@link EstimationMethod} enum and requires a double[] data type.
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
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            final int i = (int) pos;
            indices[count++] = i;
            if (pos > i) {
                // Require the next index for interpolation
                indices[count++] = i + 1;
            }
        }
        // Partition
        kthSelector.partitionSP(x, Arrays.copyOf(indices, count));
        // Compute
        for (int i = 0; i < p.length; i++) {
            // XXX: EstimationMethod coupled to the type double[]
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

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        if (pos > i) {
            part.partition(x, i, i + 1);
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
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
    private double[] evaluate(PartitionFunction part, double[] values, double... p) {
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
        int count = p.length;
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            final int i = (int) pos;
            indices[k] = i;
            if (pos > i) {
                // Require the next index for interpolation
                indices[count++] = i + 1;
                q[k] = pos - i;
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
    private double evaluate2(PartitionFunction2 part, double[] values, double p) {
        checkQuantile(p);
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 1) {
            return n == 0 ? Double.NaN : values[0];
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        if (pos > i) {
            part.partition(x, new int[] {i, i + 1}, 2);
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
        }
        part.partition(x, new int[] {i}, 1);
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
    private double[] evaluate2(PartitionFunction2 part, double[] values, double... p) {
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Implicit NPE
        final int n = values.length;
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
        int count = 0;
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            q[k] = pos;
            final int i = (int) pos;
            indices[count++] = i;
            if (pos > i) {
                // Require the next index for interpolation
                indices[count++] = i + 1;
            }
        }

        // Partition
        part.partition(x, indices, count);

        // Compute
        for (int k = 0; k < p.length; k++) {
            final int i = (int) q[k];
            final double alpha = q[k] - i;
            if (alpha != 0) {
                q[k] = DoubleMath.interpolate(x[i], x[i + 1], alpha);
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
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateSP(double[], double...)} method should be used
     * which provides better performance.
     *
     * <p>The partition function is not required to handle NaN or signed zeros.
     *
     * @param part Partition function.
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    private double evaluate3(PartitionFunction3 part, double[] values, double p) {
        checkQuantile(p);
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 1) {
            t.postProcess(x, null, 0);
            return n == 0 ? Double.NaN : values[0];
        }
        // Length of data to partition
        final int len = t.length();

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        // Do the minimal partition work below the data length.
        if (pos > i) {
            final int[] k = new int[] {i, i + 1};
            if (i < len) {
                final int kn = i <= len ? 2 : 1;
                part.partition(x, len, k, kn);
                t.postProcess(x, k, kn);
            } else {
                t.postProcess(x, null, 0);
            }
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
        }
        if (i < len) {
            final int[] k = new int[] {i};
            part.partition(x, len, k, 1);
            t.postProcess(x, k, 1);
        } else {
            t.postProcess(x, null, 0);
        }
        return x[i];
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>The partition function is not required to handle NaN or signed zeros.
     *
     * @param part Partition function.
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    private double[] evaluate3(PartitionFunction3 part, double[] values, double... p) {
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        final double[] q = new double[p.length];
        if (n <= 1) {
            t.postProcess(x, null, 0);
            Arrays.fill(q, n == 0 ? Double.NaN : values[0]);
            return q;
        }

        // Length of data to partition
        final int len = t.length();

        // Collect interpolation positions. We use the output q to store factors.
        final int[] indices = new int[p.length * 2];
        int count = 0;
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            q[k] = pos;
            final int i = (int) pos;
            // Only have to partition up to length
            if (i < len) {
                indices[count++] = i;
                if (pos > i && i <= len) {
                    // Require the next index for interpolation
                    indices[count++] = i + 1;
                }
            }
        }

        // Partition
        if (count != 0) {
            part.partition(x, len, indices, count);
        }
        t.postProcess(x, indices, count);

        // Compute
        for (int k = 0; k < p.length; k++) {
            final int i = (int) q[k];
            final double alpha = q[k] - i;
            if (alpha != 0) {
                q[k] = DoubleMath.interpolate(x[i], x[i + 1], alpha);
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
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluate(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluate(double[] values, double p) {
        checkQuantile(p);
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 1) {
            t.postProcess(x, null, 0);
            return n == 0 ? Double.NaN : values[0];
        }
        // Length of data to partition
        final int len = t.length();

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        // Do the minimal partition work below the data length.
        if (pos > i) {
            final int[] k = new int[] {i, i + 1};
            if (i < len) {
                final int kn = i <= len ? 2 : 1;
                Partition.select(x, len, k, kn);
                t.postProcess(x, k, kn);
            } else {
                t.postProcess(x, null, 0);
            }
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
        }
        if (i < len) {
            final int[] k = new int[] {i};
            Partition.select(x, len, k, 1);
            t.postProcess(x, k, 1);
        } else {
            t.postProcess(x, null, 0);
        }
        return x[i];
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluate(double[] values, double... p) {
        if (p.length == 0) {
            throw new IllegalArgumentException(NO_QUANTILES_SPECIFIED);
        }
        for (final double pp : p) {
            checkQuantile(pp);
        }
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        final double[] q = new double[p.length];
        if (n <= 1) {
            t.postProcess(x, null, 0);
            Arrays.fill(q, n == 0 ? Double.NaN : values[0]);
            return q;
        }

        // Length of data to partition
        final int len = t.length();

        // Collect interpolation positions. We use the output q to store factors.
        final int[] indices = new int[p.length * 2];
        int count = 0;
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            q[k] = pos;
            final int i = (int) pos;
            // Only have to partition up to length
            if (i < len) {
                indices[count++] = i;
                if (pos > i && i <= len) {
                    // Require the next index for interpolation
                    indices[count++] = i + 1;
                }
            }
        }

        // Partition
        if (count != 0) {
            Partition.select(x, len, indices, count);
        }
        t.postProcess(x, indices, count);

        // Compute
        for (int k = 0; k < p.length; k++) {
            final int i = (int) q[k];
            final double alpha = q[k] - i;
            if (alpha != 0) {
                q[k] = DoubleMath.interpolate(x[i], x[i + 1], alpha);
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
    private double evaluateK1(PartitionFunction2 part, double[] values, double p) {
        // Implicit NPE
        final int n = values.length;
        checkQuantile(p);
        // Special cases
        if (n <= 1) {
            return n == 0 ? Double.NaN : values[0];
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        // This requires a partition function to partition k+1
        // for each input k
        if (pos > i) {
            part.partition(x, new int[] {i}, 1);
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
        }
        part.partition(x, new int[] {i}, 1);
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
    private double[] evaluateK1(PartitionFunction2 part, double[] values, double... p) {
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
        // This requires a partition function to partition k+1 for each input k
        final int[] indices = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            final double pos = estimationType.index(p[i], n);
            q[i] = pos;
            indices[i] = (int) pos;
        }

        // Partition
        part.partition(x, indices, indices.length);

        // Compute
        for (int i = 0; i < p.length; i++) {
            final int index = (int) q[i];
            final double alpha = q[i] - index;
            if (alpha != 0) {
                q[i] = DoubleMath.interpolate(x[index], x[index + 1], alpha);
            } else {
                q[i] = x[index];
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
    private double evaluatePaired(PartitionFunction part, double[] values, double p) {
        // Implicit NPE
        final int n = values.length;
        checkQuantile(p);
        // Special cases
        if (n <= 1) {
            return n == 0 ? Double.NaN : values[0];
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();

        final double pos = estimationType.index(p, n);
        final int i = (int) pos;

        // Partition and compute
        if (pos > i) {
            part.partition(x, i | Integer.MIN_VALUE);
            return DoubleMath.interpolate(x[i], x[i + 1], pos - i);
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
    public double[] evaluatePaired(PartitionFunction part, double[] values, double... p) {
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
        final int[] indices = new int[p.length];
        for (int k = 0; k < p.length; k++) {
            final double pos = estimationType.index(p[k], n);
            q[k] = pos;
            int i = (int) pos;
            if (pos > i) {
                // Require the next index for interpolation
                i |= Integer.MIN_VALUE;
            }
            indices[k] = i;
        }

        // Partition
        part.partition(x, indices);

        // Compute
        for (int k = 0; k < p.length; k++) {
            final int i = (int) q[k];
            final double alpha = q[k] - i;
            if (alpha != 0) {
                q[k] = DoubleMath.interpolate(x[i], x[i + 1], alpha);
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
        return evaluate2(partition::partitionSBM, values, p);
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
        return evaluate2(partition::partitionSBM, values, p);
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
     * within the same values. The {@link #evaluateKSBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateKSBM(double[] values, double p) {
        return evaluate2(partition::partitionKSBM, values, p);
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
    public double[] evaluateKSBM(double[] values, double... p) {
        return evaluate2(partition::partitionKSBM, values, p);
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
     * within the same values. The {@link #evaluateKSBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateK1SBM(double[] values, double p) {
        return evaluateK1(partition::partitionK1SBM, values, p);
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
    public double[] evaluateK1SBM(double[] values, double... p) {
        return evaluateK1(partition::partitionK1SBM, values, p);
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
    public double evaluatePairedSBM(double[] values, double p) {
        return evaluatePaired(partition::partitionPairedSBM, values, p);
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
    public double[] evaluatePairedSBM(double[] values, double... p) {
        return evaluatePaired(partition::partitionPairedSBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a Bentley-McIlroy quickselect partition method
     * handling equal keys by Sedgewick; switching to heapselect if quickselect convergence
     * is slow.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateISBM(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateISBM(double[] values, double p) {
        return evaluate3(partition::partitionISBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a Bentley-McIlroy quickselect partition method
     * handling equal keys by Sedgewick; switching to heapselect if quickselect convergence
     * is slow.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateISBM(double[] values, double... p) {
        return evaluate3(partition::partitionISBM, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantile of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a dual-pivot quickselect partition method;
     * switching to heapselect if quickselect convergence is slow.
     *
     * <p><strong>Performance</strong>
     *
     * <p>It is not recommended to use this method for repeat calls for different quantiles
     * within the same values. The {@link #evaluateIDP(double[], double...)} method should be used
     * which provides better performance.
     *
     * @param values Values.
     * @param p Quantile.
     * @return the quantile
     * @throws IllegalArgumentException if the quantile {@code p} is not in the range {@code [0, 1]}
     * @see #evaluateSP(double[], double...)
     */
    public double evaluateIDP(double[] values, double p) {
        return evaluate3(partition::partitionIDP, values, p);
    }

    /**
     * Evaluate the <code>p</code>th quantiles of the values.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a dual-pivot quickselect partition method;
     * switching to heapselect if quickselect convergence is slow.
     *
     * @param values Values.
     * @param p Quantiles.
     * @return the quantiles
     * @throws IllegalArgumentException if any quantile {@code p} not in the range {@code [0, 1]};
     * or no quantiles are specified.
     */
    public double[] evaluateIDP(double[] values, double... p) {
        return evaluate3(partition::partitionIDP, values, p);
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
     * Estimation methods for a quantile. Provides the nine quantile algorithms
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
    public enum EstimationMethod {
        /**
         * Inverse of the empirical distribution function.
         *
         * <p>\( m = 0 \); \( \gamma = 0 \) if \( g = 0 \), and 1 otherwise.
         */
        HF1 {
            @Override
            double position0(double p, int n) {
                double pos = n * p;
                // ceil(j + alpha) : note j is 1-based
                return Math.ceil(pos) - 1;
            }
        },
        /**
         * Similar to {@link #HF1} with averaging at discontinuities.
         *
         * <p>\( m = 0 \); \( \gamma = 0.5 \) if \( g = 0 \), and 1 otherwise.
         */
        HF2 {
            @Override
            double position0(double p, int n) {
                double pos = n * p;
                // Average at discontinuities
                int j = (int) Math.floor(pos);
                double alpha = pos - j;
                if (alpha == 0) {
                    return j - 0.5;
                }
                // As HF1 : ceil(j + alpha) : note j is 1-based
                // so we can return floor(j + alpha)
                return j;
            }
        },
        /**
         * The observation closest to \( np \). Ties are resolved to the nearest even order statistic.
         *
         * <p>\( m = -1/2 \); \( \gamma = 0 \) if \( g = 0 \) and \( j \) is even, and 1 otherwise.
         */
        HF3 {
            @Override
            double position0(double p, int n) {
                // Let rint do the work for ties to even
                return Math.rint(n * p) - 1;
            }
        },
        /**
         * Linear interpolation of the inverse of the empirical CDF.
         *
         * <p>\( m = 0 \).
         */
        HF4 {
            @Override
            double position0(double p, int n) {
                return n * p - 1;
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
            double position0(double p, int n) {
                return n * p - 0.5;
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
            double position0(double p, int n) {
                return (n + 1) * p - 1;
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
            double position0(double p, int n) {
                return (n - 1) * p;
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
            double position0(double p, int n) {
                return n * p + (p + 1) / 3 - 1;
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
            double position0(double p, int n) {
                return (n + 0.25) * p - 0.625;
            }
        };

        /**
         * Finds the real valued position for calculation of the quantile.
         *
         * <p>Return {@code i + g} such that the quantile value from sorted data is:
         *
         * <p>value = data[i] + g * (data[i+1] - data[i])
         *
         * <p>Warning: Interpolation should not use {@code data[i+1]} unless {@code g != 0}.
         *
         * <p>Note: In contrast to the definition of Hyndman and Fan in the class header
         * which uses a 1-based position, this is a zero based index. This change is for
         * convenience when addressing array positions.
         *
         * @param p p<sup>th</sup> quantile.
         * @param n Size.
         * @return a real valued position (0-based) into the range {@code [0, n)}
         */
        abstract double position0(double p, int n);

        /**
         * Finds the index {@code i} and fractional part {@code g} of a real valued position
         * to interpolate the quantile.
         *
         * <p>Return {@code i + g} such that the quantile value from sorted data is:
         *
         * <p>value = data[i] + g * (data[i+1] - data[i])
         *
         * <p>Note: Interpolation should not use {@code data[i+1]} unless {@code g != 0}.
         *
         * @param p p<sup>th</sup> quantile.
         * @param n Size.
         * @return index
         */
        final double index(double p, int n) {
            final double pos = position0(p, n);
            // Bounds check
            if (pos < 0) {
                return 0;
            }
            if (pos >= n - 1.0) {
                return n - 1.0;
            }
            return pos;
        }

        /**
         * Evaluate method to compute the quantile.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #index(double, int)} method.
         *
         * @param values Values.
         * @param p p<sup>th</sup> quantile to be computed
         * @return estimated quantile
         */
        final double evaluatePartitioned(double[] values, double p) {
            final int n = values.length;
            return estimatePartitioned(values, position0(p, n), n);
        }

        /**
         * Estimation based on partitioned values.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #index(double, int)} method.
         *
         * @param values Values.
         * @param pos Positional index prior computed from calling {@link #position0(double, int)}.
         * @param n Size.
         * @return estimated quantile
         */
        final double estimatePartitioned(double[] values, double pos, int n) {
            // position = i + g = pn + m - 1
            // i = floor(pn + m - 1)
            // g = pn + m - i
            final double fpos = Math.floor(pos);
            final int i = (int) fpos;
            final double g = pos - fpos;
            if (i < 0) {
                return values[0];
            }
            if (i >= n - 1) {
                return values[n - 1];
            }
            return estimatePartitioned(values, i, g, n);
        }

        /**
         * Estimation based on partitioned values.
         * Uses the position index from {@link #position0(double, int)}:
         * <pre>
         * i = floor(position)
         * g = position - i
         * </pre>
         *
         * <p>This method is only called when {@code 1 <= i < n}. The fractional part {@code g}
         * may be zero, i.e. {@code 0 <= g < 1}.
         *
         * <p>The default implementation provides a continuous function using linear interpolation
         * between the value at {@code i} and {@code i+1}. This may be overridden in specific
         * enums to compute slightly different estimations.
         *
         * <p>It is assumed that the values are correctly partitioned by the indices
         * identified by the {@link #index(double, int)} method.
         *
         * @param values Values.
         * @param i Integer part of the position index.
         * @param g Fractional part of the position index.
         * @param n Size.
         * @return estimated quantile
         */
        double estimatePartitioned(double[] values, int i, double g, int n) {
            // Note position index is 1-based:
            if (g == 0) {
                // No interpolation
                return values[i];
            }
            return DoubleMath.interpolate(values[i], values[i + 1], g);
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
            return estimate(values, pivotsHeap, position0(p, n), n, selector);
        }

        /**
         * Estimation based on K<sup>th</sup> selection.
         *
         * @param values Values.
         * @param pos Positional index prior computed from calling {@link #position0(double, int)}.
         * @param pivotsHeap Cache of sort pivot points.
         * @param n Size.
         * @param selector {@link KthSelector} used for pivoting during search
         * @return estimated quantile
         */
        final double estimate(double[] values, int[] pivotsHeap, double pos, int n, KthSelector selector) {
            // position = i + g = pn + m - 1
            // i = floor(pn + m - 1)
            // g = pn + m - i - 1
            final double fpos = Math.floor(pos);
            final int i = (int) fpos;
            final double g = pos - fpos;
            // Bounds check. Note that i is 1-based.
            if (i < 0) {
                return selector.selectSPH(values, pivotsHeap, 0, null);
            }
            if (i >= n - 1) {
                return selector.selectSPH(values, pivotsHeap, n - 1, null);
            }
            return estimate(values, pivotsHeap, i, g, n, selector);
        }

        /**
         * Estimation based on K<sup>th</sup> selection.
         * Uses the position index from {@link #position0(double, int)}:
         * <pre>
         * i = floor(position)
         * g = position - i
         * </pre>
         *
         * <p>This method is only called when {@code 0 <= i < n - 1}. The fractional part {@code g}
         * may be zero, i.e. {@code 0 <= g < 1}.
         *
         * <p>The default implementation provides a continuous function using linear interpolation
         * between the value at {@code i} and {@code i+1}. This may be overridden in specific
         * enums to compute slightly different estimations.
         *
         * @param values Values.
         * @param pivotsHeap Cache of sort pivot points.
         * @param i Integer part of the position index.
         * @param g Fractional part of the position index.
         * @param n Size.
         * @param selector {@link KthSelector} used for pivoting during search
         * @return estimated quantile
         */
        double estimate(double[] values, int[] pivotsHeap, int i, double g, int n, KthSelector selector) {
            // Note position index is 0-based:
            if (g == 0) {
                // No interpolation
                return selector.selectSPH(values, pivotsHeap, i, null);
            }
            final double[] upper = {0};
            final double lower = selector.selectSPH(values, pivotsHeap, i, upper);
            return DoubleMath.interpolate(lower, upper[0], g);
        }
    }
}
