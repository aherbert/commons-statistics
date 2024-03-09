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

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Returns the median of the available values.
 *
 * <p>For values of length {@code n}, let {@code k = n / 2}:
 * <ul>
 * <li>The result is {@code NaN} if {@code n = 0}.
 * <li>The result is {@code values[k]} if {@code n} is odd.
 * <li>The result is {@code (values[k - 1] + values[k]) / 2} if {@code n} is even.
 * </ul>
 *
 * <p>This implementation respects the ordering imposed by
 * {@link Double#compare(double, double)} for {@code NaN} values. If a {@code NaN} occurs
 * in the selected positions in the fully sorted values then the result is {@code NaN}.
 *
 * @see java.util.Arrays#sort(double[])
 * @since 1.1
 */
public final class Median {
    /** Default instance. */
    private static final Median DEFAULT = new Median(false, NaNPolicy.INCLUDE,
        new KthSelector(), new Partition());

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

    /**
     * @param overwrite Flag to indicate if the data should be overwritten.
     * @param nanPolicy NaN policy.
     * @param kthSelector Selector for the K-th element in an array.
     * @param partition Partition method for partial sort of an array.
     */
    private Median(boolean overwrite, NaNPolicy nanPolicy,
            KthSelector kthSelector, Partition partition) {
        this.overwrite = overwrite;
        this.nanPolicy = nanPolicy;
        this.kthSelector = kthSelector;
        this.partition = partition;
        transformer = DoubleDataTransformers.createFactory(nanPolicy, !overwrite);
    }

    /**
     * Return a new instance with the default options.
     *
     * <ul>
     * <li>{@linkplain #withOverwrite(boolean) Overwrite = false}
     * <li>{@linkplain #with(NaNPolicy) NaN policy = include}
     * </ul>
     *
     * @return the median
     */
    public static Median withDefaults() {
        return DEFAULT;
    }

    /**
     * Return an instance with the configured overwrite behaviour. If {@code true} then
     * the input array will be modified by the call to
     *
     * @param v Value.
     * @return an instance
     */
    public Median withOverwrite(boolean v) {
        return new Median(v, nanPolicy, kthSelector, partition);
    }

    /**
     * Return an instance with the configured {@link NaNPolicy}.
     *
     * @param v Value.
     * @return an instance
     */
    public Median with(NaNPolicy v) {
        return new Median(overwrite, Objects.requireNonNull(v), kthSelector, partition);
    }

    /**
     * Return an instance with the configured {@link KthSelector}.
     *
     * <p>Note: This is not part of the public API. It is exposed for performance testing.
     *
     * @param v Value.
     * @return an instance
     */
    Median withKthSelector(KthSelector v) {
        return new Median(overwrite, nanPolicy, Objects.requireNonNull(v), partition);
    }

    /**
     * Return an instance with the configured {@link Partition}.
     *
     * <p>Note: This is not part of the public API. It is exposed for performance testing.
     *
     * @param v Value.
     * @return an instance
     */
    Median withPartition(Partition v) {
        return new Median(overwrite, nanPolicy, kthSelector, Objects.requireNonNull(v));
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot quicksort partition method with equivalent of
     * Double.compare to sort NaN and signed zeros.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateSP(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            return kthSelector.selectSP(x, k, null);
        }
        // Even
        final double[] kp1 = {0};
        final double a = kthSelector.selectSP(x, k - 1, kp1);
        return DoubleMath.mean(a, kp1[0]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a single-pivot quicksort partition method with special NaN/zero handling.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateSPN(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            return kthSelector.selectSPN(x, k, null);
        }
        // Even
        final double[] kp1 = {0};
        final double a = kthSelector.selectSPN(x, k - 1, kp1);
        return DoubleMath.mean(a, kp1[0]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method handling equal keys by Sedgewick.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateSBM(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            kthSelector.partitionSBM(x, k);
            return x[k];
        }
        // Even
        kthSelector.partitionSBM(x, k - 1, k);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition handling equal keys(original).
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateBM(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            kthSelector.partitionBM(x, k);
            return x[k];
        }
        // Even
        kthSelector.partitionBM(x, k - 1, k);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort partition method handling equal keys.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateDP(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            kthSelector.partitionDP(x, k);
            return x[k];
        }
        // Even
        kthSelector.partitionDP(x, k - 1, k);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a dual-pivot quicksort partition method handling equal keys
     * with 5 sorted points to choose pivots.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateDP5(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            kthSelector.partitionDP5(x, k);
            return x[k];
        }
        // Even
        kthSelector.partitionDP5(x, k - 1, k);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method handling equal keys by Sedgewick.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateSBM2(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            partition.partitionSBM(x, new int[] {k}, 1);
            return x[k];
        }
        // Even
        partition.partitionSBM(x, new int[] {k - 1, k}, 2);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method handling equal keys by Sedgewick.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluatePairedSBM(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            partition.partitionPairedSBM(x, k);
            return x[k];
        }
        // Even: require (k-1, k)
        partition.partitionPairedSBM(x, (k - 1) | Integer.MIN_VALUE);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method handling equal keys by Sedgewick.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateKSBM(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            partition.partitionKSBM(x, new int[] {k}, 1);
            return x[k];
        }
        // Even: require (k-1, k)
        partition.partitionKSBM(x, new int[] {k - 1, k}, 2);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses a Bentley-McIlroy quicksort partition method handling equal keys by Sedgewick.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateK1SBM(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return DoubleMath.mean(values[0], values[1]);
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = overwrite ? values : values.clone();
        final int k = n >>> 1;

        // Odd
        if ((n & 0x1) == 0x1) {
            partition.partitionK1SBM(x, new int[] {k}, 1);
            return x[k];
        }
        // Even: require (k-1, k)
        // Here we know the function partitions
        // a pair of indices together so only pass in k - 1.
        partition.partitionK1SBM(x, new int[] {k - 1}, 1);
        return DoubleMath.mean(x[k - 1], x[k]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant. The quickselect is a single-pivot partition method;
     * the fall-back on poor convergence of the quickselect is a heapselect.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateISP(double[] values) {
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 2) {
            t.postProcess(x, null, 0);
            switch (n) {
            case 2:
                return DoubleMath.mean(x[0], x[1]);
            case 1:
                return x[0];
            default:
                return Double.NaN;
            }
        }
        // Median index
        final int m = n >>> 1;
        // Length of data to partition
        final int len = t.length();
        // Odd
        if ((n & 0x1) == 0x1) {
            if (m < len) {
                final int[] k = new int[] {m};
                partition.partitionISP(x, len, k, 1);
                t.postProcess(x, k, 1);
            } else {
                t.postProcess(x, null, 0);
            }
            return x[m];
        }
        // Even: require (m-1, m)
        // Do the minimal partition work
        final int[] k = new int[] {m - 1, m};
        if (m - 1 < len) {
            final int kn = m < len ? 2 : 1;
            partition.partitionISP(x, len, k, kn);
            t.postProcess(x, k, kn);
        } else {
            t.postProcess(x, null, 0);
        }
        return DoubleMath.mean(x[m - 1], x[m]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a Bentley-McIlroy quickselect partition method
     * handling equal keys by Sedgewick; switching to heapselect if quickselect convergence
     * is slow.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateISBM(double[] values) {
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 2) {
            t.postProcess(x, null, 0);
            switch (n) {
            case 2:
                return DoubleMath.mean(x[0], x[1]);
            case 1:
                return x[0];
            default:
                return Double.NaN;
            }
        }
        // Median index
        final int m = n >>> 1;
        // Length of data to partition
        final int len = t.length();
        // Odd
        if ((n & 0x1) == 0x1) {
            if (m < len) {
                final int[] k = new int[] {m};
                partition.partitionISBM(x, len, k, 1);
                t.postProcess(x, k, 1);
            } else {
                t.postProcess(x, null, 0);
            }
            return x[m];
        }
        // Even: require (m-1, m)
        // Do the minimal partition work
        final int[] k = new int[] {m - 1, m};
        if (m - 1 < len) {
            final int kn = m < len ? 2 : 1;
            partition.partitionISBM(x, len, k, kn);
            t.postProcess(x, k, kn);
        } else {
            t.postProcess(x, null, 0);
        }
        return DoubleMath.mean(x[m - 1], x[m]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * <p>Uses an introselect variant with a dual-pivot quickselect partition method;
     * switching to heapselect if quickselect convergence is slow.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluateIDP(double[] values) {
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 2) {
            t.postProcess(x, null, 0);
            switch (n) {
            case 2:
                return DoubleMath.mean(x[0], x[1]);
            case 1:
                return x[0];
            default:
                return Double.NaN;
            }
        }
        // Median index
        final int m = n >>> 1;
        // Length of data to partition
        final int len = t.length();
        // Odd
        if ((n & 0x1) == 0x1) {
            if (m < len) {
                final int[] k = new int[] {m};
                partition.partitionIDP(x, len, k, 1);
                t.postProcess(x, k, 1);
            } else {
                t.postProcess(x, null, 0);
            }
            return x[m];
        }
        // Even: require (m-1, m)
        // Do the minimal partition work
        final int[] k = new int[] {m - 1, m};
        if (m - 1 < len) {
            final int kn = m < len ? 2 : 1;
            partition.partitionIDP(x, len, k, kn);
            t.postProcess(x, k, kn);
        } else {
            t.postProcess(x, null, 0);
        }
        return DoubleMath.mean(x[m - 1], x[m]);
    }

    /**
     * Evaluate the median.
     *
     * <p>Note: This method may partially sort this input values if configured to
     * {@link #withOverwrite(boolean) overwrite} the input data.
     *
     * @param values Values.
     * @return the median
     */
    public double evaluate(double[] values) {
        // Floating-point data handling
        final DoubleDataTransformer t = transformer.get();
        final double[] x = t.preProcess(values);
        final int n = t.size();
        // Special cases
        if (n <= 2) {
            t.postProcess(x, null, 0);
            switch (n) {
            case 2:
                return DoubleMath.mean(x[0], x[1]);
            case 1:
                return x[0];
            default:
                return Double.NaN;
            }
        }
        // Median index
        final int m = n >>> 1;
        // Length of data to partition
        final int len = t.length();
        // Odd
        if ((n & 0x1) == 0x1) {
            if (m < len) {
                final int[] k = new int[] {m};
                Partition.select(x, len, k, 1);
                t.postProcess(x, k, 1);
            } else {
                t.postProcess(x, null, 0);
            }
            return x[m];
        }
        // Even: require (m-1, m)
        // Do the minimal partition work
        final int[] k = new int[] {m - 1, m};
        if (m - 1 < len) {
            final int kn = m < len ? 2 : 1;
            Partition.select(x, len, k, kn);
            t.postProcess(x, k, kn);
        } else {
            t.postProcess(x, null, 0);
        }
        return DoubleMath.mean(x[m - 1], x[m]);
    }
}
