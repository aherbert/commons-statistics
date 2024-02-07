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
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.IntSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.PermutationSampler;
import org.apache.commons.rng.sampling.distribution.ContinuousSampler;
import org.apache.commons.rng.sampling.distribution.DiscreteUniformSampler;
import org.apache.commons.rng.sampling.distribution.ZigguratSampler;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Quantile.EstimationMethod;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Executes a benchmark of the creation of a quantile from {@code double} array data.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {"-server", "-Xms512M", "-Xmx8192M"})
public class QuantilePerformance {
    /** Commons Statistics Quantile implementation with single-pivot partitioning using a heap.
     * This method is copied from Commons Math. */
    private static final String SPH = "SPH";
    /** Commons Statistics Quantile implementation with single-pivot partitioning.
     * This method is adapted from Commons Math.
     * Evaluation couples the double[] data type to the EstimationMethod class. */
    private static final String SPE = "SPE";
    /** Commons Statistics Quantile implementation with single-pivot partitioning. */
    private static final String SP = "SP";
    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String SBM = "SBM";
    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (original). */
    private static final String BM = "BM";
    /** Commons Statistics Quantile implementation with a dual-pivot strategy. */
    private static final String DP = "DP";
    /** Commons Statistics Quantile implementation with a dual-pivot strategy
     * with 5 sorted points to choose pivots. */
    private static final String DP5 = "DP_5";
    /** Commons Math Percentile implementation. */
    private static final String CM = "CM";
    /** Quantile implementation using a sort. */
    private static final String SORT = "Sort";
    /** Partition implementation using a single-pivot strategy with Dutch National Flag partitioning. */
    private static final String DNF = "DNF";
    /** Use the JDK sort function. */
    private static final String JDK = "JDK";

    // Second generation partition functions

    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String SBM2 = "2SBM";
    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String KSBM = "KSBM";
    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String K1SBM = "K1SBM";

    // Paired-key partition functions

    /** Commons Statistics Quantile implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String PSBM = "PSBM";

    // Introselect functions

    /** Commons Statistics Quantile implementation with Sedgewick's BM quickselect, switching
     * to heapselect when progress is poor. */
    private static final String ISBM = "ISBM";

    /** Pattern for the minimum quickselect size. */
    private static final Pattern QS_PATTERN = Pattern.compile("QS(\\d+)");
    /** Pattern for the minimum heapselect size. */
    private static final Pattern HS_PATTERN = Pattern.compile("HS(\\d+)");

    /**
     * Source of {@code double} array data.
     */
    @State(Scope.Benchmark)
    public abstract static class AbstractDataSource {
        /** Seed for data length extension.
         * This is fixed to generate the same lengths across samples
         * for all iterations. JMH may spawn new JVMs so we used a constant value. */
        private static final long RANGE_SEED = 2384628642384839L;

        /** Type of data. */
        @Param({"uniform",
            //"normal", "exponential"
            })
        private String distribution;

        /** Number of samples. */
        @Param({"100"})
        private int samples;

        /** Data. */
        private double[][] data;

        /**
         * @return the data
         */
        public double[][] getData() {
            return data;
        }

        /**
         * Create the data.
         */
        @Setup(Level.Iteration)
        public void setup() {
            Objects.requireNonNull(distribution);
            final int length = getLength();
            // Data will be randomized per iteration
            final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
            data = new double[samples][];
            // Create the sampler
            ContinuousSampler sampler;

            // TODO:
            // Add support for other data, e.g. see Bentley-McIlroy paper on quicksort
            // which has example of other data types.

            if (distribution.startsWith("uni")) {
                final int discrete = getInteger(distribution, -1);
                if (discrete > 0) {
                    // Note: upper bound is inclusive.
                    // This will handle discrete == 1 (although the stream ... toArray() is inefficient).
                    sampler = DiscreteUniformSampler.of(rng, 0, discrete - 1)::sample;
                } else {
                    sampler = rng::nextDouble;
                }
            } else if (distribution.startsWith("norm")) {
                sampler = ZigguratSampler.NormalizedGaussian.of(rng)::sample;
            } else if (distribution.startsWith("exp")) {
                sampler = ZigguratSampler.Exponential.of(rng)::sample;
            } else {
                throw new IllegalStateException("Unknown distribution: " + distribution);
            }
            IntSupplier sampleLength;
            int range = getRange();
            if (range <= 0) {
                sampleLength = () -> length;
            } else {
                // Sample in [length, length + range]
                sampleLength = DiscreteUniformSampler.of(
                    RandomSource.XO_RO_SHI_RO_128_PP.create(RANGE_SEED), length,
                    length + range)::sample;
            }
            for (int i = 0; i < samples; i++) {
                data[i] = sampler.samples(sampleLength.getAsInt()).toArray();
            }
        }

        /**
         * Gets the minimum length of the data.
         * The actual length is randomly sampled from {@code [length, length + range]}.
         *
         * @return the length
         * @see #getRange()
         */
        protected abstract int getLength();

        /**
         * Gets the maximum addition to extend the length of each sample of data.
         *
         * <p>Can be used to create random lengths of data. The same seed is
         * used for the random length so that repeat data generation creates
         * the same lengths for all iterations.
         *
         * <p>The default value is zero.
         *
         * @return the range
         * @see #getLength()
         */
        protected int getRange() {
            return 0;
        }
    }

    /**
     * Source of {@code double} array data.
     */
    @State(Scope.Benchmark)
    public static class DataSource extends AbstractDataSource {
        /** Data length. */
        @Param({//"0", "1",
            "10", "100", "1000", "10000"})
        private int length;

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }
    }

    /**
     * Source of quantiles.
     */
    @State(Scope.Benchmark)
    public static class QuantileSource {
        /** Quantiles.
         * Delimited by ':' to allow use via the JMH command-line parser which
         * uses ',' as the delimiter. */
        @Param({"0.25:0.5:0.75",
                "0.01:0.99",
                "0.0:1.0", // min,max
                "0.25:0.75",
                "0.001:0.005:0.01:0.02:0.05:0.1:0.5",
                "0.01:0.05:0.1:0.5:0.9:0.95:0.99"})
        private String quantiles;

        /** Data. */
        private double[] data;

        /**
         * @return the data
         */
        public double[] getData() {
            return data;
        }

        /**
         * Create the data.
         */
        @Setup
        public void setup() {
            data = Arrays.stream(quantiles.split(":")).mapToDouble(Double::parseDouble).toArray();
        }
    }

    /**
     * Source of quantiles uniformly spaced within a range.
     */
    @State(Scope.Benchmark)
    public static class QuantileRangeSource {
        /** Lower quantile. */
        @Param({"0.01"})
        private double lowerQ;
        /** Upper quantile. */
        @Param({"0.99"})
        private double upperQ;
        /** Number of quantiles. */
        @Param({"100"})
        private int quantiles;

        /** Data. */
        private double[] data;

        /**
         * @return the data
         */
        public double[] getData() {
            return data;
        }

        /**
         * Create the data.
         */
        @Setup
        public void setup() {
            if (quantiles < 2) {
                throw new IllegalStateException("Bad quantile count: " + quantiles);
            }
            if (!(lowerQ >= 0 && upperQ <= 1)) {
                throw new IllegalStateException("Bad quantile range: [" + lowerQ + ", " + upperQ + "]");
            }
            data = new double[quantiles];
            for (int i = 0; i < quantiles; i++) {
                // Create u in [0, 1]
                final double u = i / (quantiles - 1.0);
                data[i] = (1 - u) * lowerQ + u * upperQ;
            }
        }
    }

    /**
     * Source of a {@link BinaryOperator} for a {@code double[]} and quantiles.
     */
    @State(Scope.Benchmark)
    public static class DoubleFunctionSource {
        /** Name of the source. */
        @Param({
            // Slow
            //SORT, SPH, SPE
            CM, SP, BM, SBM, DP, DP5,
            SBM2, KSBM, K1SBM,
            PSBM,
            ISBM})
        private String name;

        /** The action. */
        private BinaryOperator<double[]> function;

        /**
         * @return the function
         */
        public BinaryOperator<double[]> getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            // Note: Functions defensively copy the data by default
            if (SORT.equals(name)) {
                function = QuantilePerformance::sortQuantile;
            } else if (CM.equals(name)) {
                final Percentile s = new Percentile().withNaNStrategy(NaNStrategy.FIXED);
                function = (x, p) -> {
                    final double[] q = new double[p.length];
                    s.setData(x);
                    for (int i = 0; i < p.length; i++) {
                        // Convert quantile to percentile
                        q[i] = s.evaluate(p[i] * 100);
                    }
                    return q;
                };
            // First generation kth-selector functions
            } else if (name.startsWith(SPH)) {
                function = withKthSelector(name)::evaluateSPH;
            } else if (name.startsWith(SPE)) {
                function = withKthSelector(name)::evaluateSPE;
            } else if (name.startsWith(SP)) {
                function = withKthSelector(name)::evaluateSP;
            } else if (name.startsWith(BM)) {
                function = withKthSelector(name)::evaluateBM;
            } else if (name.startsWith(SBM)) {
                function = withKthSelector(name)::evaluateSBM;
            } else if (name.startsWith(DP)) {
                function = withKthSelector(name)::evaluateDP;
            } else if (name.startsWith(DP5)) {
                function = withKthSelector(name)::evaluateDP5;
            // Second generation partition functions
            } else if (name.startsWith(SBM2)) {
                function = withPartition(name)::evaluateSBM2;
            } else if (name.startsWith(KSBM)) {
                function = withPartition(name)::evaluateKSBM;
            } else if (name.startsWith(K1SBM)) {
                function = withPartition(name)::evaluateK1SBM;
            // Paired key implementations
            } else if (name.startsWith(PSBM)) {
                function = withPartition(name)::evaluatePairedSBM;
            // Introselect implementations
            } else if (name.startsWith(ISBM)) {
                function = withPartition(name)::evaluateISBM;
            } else {
                throw new IllegalStateException("Unknown double[] function: " + name);
            }
        }
    }

    /**
     * Source of {@code double} array data to sort.
     */
    @State(Scope.Benchmark)
    public static class SortSource extends AbstractDataSource {
        /** Data length. */
        @Param({"20", "40", "80"})
        private int length;

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }
    }

    /**
     * Source of k-th indices.
     */
    @State(Scope.Benchmark)
    public static class KSource extends SortSource {
        /** Number of indices to select. */
        @Param({"1", "2", "3", "5", "10"})
        private int k;
        /** Number of repeats. */
        @Param({"10"})
        private int repeats;

        /** Indices. */
        private int[][] indices;

        /**
         * @return the indices
         */
        public int[][] getIndices() {
            return indices;
        }

        /**
         * Create the indices.
         */
        @Override
        @Setup(Level.Iteration)
        public void setup() {
            super.setup();
            // Data will be randomized per iteration
            final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
            final PermutationSampler s = new PermutationSampler(rng, getLength(), k);
            indices = s.samples(repeats).toArray(int[][]::new);
        }
    }

    /**
     * Source of a sort function.
     */
    @State(Scope.Benchmark)
    public static class SortFunctionSource {
        /** Name of the source. */
        @Param({JDK, SP, BM, SBM, DP, DP5,
            SBM2,
            //DNF,
            // Not run by default as it is slow on large data
            //"InsertionSort",
            //"BM25"
            ISBM,
            })
        private String name;

        /** The action. */
        private Consumer<double[]> function;

        /**
         * @return the function
         */
        public Consumer<double[]> getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            if (JDK.equals(name)) {
                function = Arrays::sort;
            // First generation kth-selector functions
            } else if (name.startsWith(SP)) {
                function = createKthSelector(name)::sortSP;
            } else if (name.startsWith(SBM)) {
                function = createKthSelector(name)::sortSBM;
            } else if (name.startsWith(BM)) {
                function = createKthSelector(name)::sortBM;
            } else if (name.startsWith(DP)) {
                function = createKthSelector(name)::sortDP;
            } else if (name.startsWith(DP5)) {
                function = createKthSelector(name)::sortDP5;
            } else if (name.startsWith(DNF)) {
                function = createKthSelector(name)::sortDNF;
            // 2nd generation partition functions
            } else if (name.startsWith(SBM2)) {
                function = createPartition(name)::sortSBM;
            // Introsort
            } else if (name.startsWith(ISBM)) {
                function = createPartition(name)::sortISBM;
            } else if ("InsertionSort".equals(name)) {
                function = x -> KthSelector.insertionSort(x, 0, x.length, false);
            } else {
                throw new IllegalStateException("Unknown sort function: " + name);
            }
        }
    }

    /**
     * Source of a k-th selector function.
     */
    @State(Scope.Benchmark)
    public static class KFunctionSource {
        /** Name of the source. */
        @Param({JDK, SPH,
            SP, BM, SBM,
            DP, DP5, DNF,
            SBM2, KSBM, K1SBM,
            PSBM,
            ISBM})
        private String name;

        /** The action. */
        private BiFunction<double[], int[], double[]> function;

        /**
         * @return the function
         */
        public BiFunction<double[], int[], double[]> getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            // Note: For parity in the test, each partition method that accepts the keys as any array
            // receives a clone of the indices.
            if (JDK.equals(name)) {
                function = (data, indices) -> {
                    Arrays.sort(data);
                    return extractIndices(data, indices);
                };
            // First generation kth-selector functions
            } else if (name.startsWith(SPH)) {
                // Ported CM implementation with a heap
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    final int n = indices.length;
                    // Note: Pivots heap is not optimal here but should be enough for most cases.
                    // The size matches that in the Commons Math Percentile class
                    final int[] pivots = n <= 1 ?
                        KthSelector.NO_PIVOTS :
                        new int[1023];
                    final double[] x = new double[indices.length];
                    for (int i = 0; i < indices.length; i++) {
                        x[i] = selector.selectSPH(data, pivots, indices[i], null);
                    }
                    return x;
                };
            } else if (name.startsWith(SP)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionSP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(BM)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(SBM)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionSBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionDP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP5)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionDP5(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DNF)) {
                final KthSelector selector = createKthSelector(name);
                function = (data, indices) -> {
                    selector.partitionDNF(data, indices.clone());
                    return extractIndices(data, indices);
                };
            // Second generation partition functions
            } else if (name.startsWith(SBM2)) {
                final Partition part = createPartition(name);
                function = (data, indices) -> {
                    part.partitionSBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(KSBM)) {
                final Partition part = createPartition(name);
                function = (data, indices) -> {
                    part.partitionKSBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(K1SBM)) {
                final Partition part = createPartition(name);
                function = (data, indices) -> {
                    part.partitionK1SBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            // Paired key implementations.
            // This can be used to show they have no disadvantage for processing single keys.
            } else if (name.startsWith(PSBM)) {
                final Partition part = createPartition(name);
                function = (data, indices) -> {
                    part.partitionPairedSBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            // Introselect implementations
            } else if (name.startsWith(ISBM)) {
                final Partition part = createPartition(name);
                function = (data, indices) -> {
                    part.partitionISBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else {
                throw new IllegalStateException("Unknown selector function: " + name);
            }
        }

        /**
         * Extract the data at the specified indices.
         *
         * @param data Data.
         * @param indices Indices.
         * @return the data
         */
        private static double[] extractIndices(double[] data, int[] indices) {
            final double[] x = new double[indices.length];
            for (int i = 0; i < indices.length; i++) {
                x[i] = data[indices[i]];
            }
            return x;
        }
    }

    /**
     * Creates the {@link Quantile}.
     * Parameters for the {@link KthSelector} are derived from the {@code name}.
     *
     * @param name Name.
     * @return the {@link Quantile} instance
     */
    private static Quantile withKthSelector(String name) {
        // For parity with the CM implementation use HF6
        return Quantile.withDefaults().with(EstimationMethod.HF6)
            .withKthSelector(createKthSelector(name));
    }

    /**
     * Creates the {@link Quantile}.
     * Parameters for the {@link Partition} are derived from the {@code name}.
     *
     * @param name Name.
     * @return the {@link Quantile} instance
     */
    private static Quantile withPartition(String name) {
        // For parity with the CM implementation use HF6
        return Quantile.withDefaults().with(EstimationMethod.HF6)
            .withPartition(createPartition(name));
    }

    /**
     * Creates the {@link KthSelector}. Parameters are derived from the {@code name}.
     *
     * @param name Name.
     * @return the {@link KthSelector} instance
     */
    static KthSelector createKthSelector(String name) {
        final int minQuickSelectSize = getMinQuickSelectSize(name);
        final PivotingStrategy s = getPivotStrategy(name);
        return new KthSelector(s, minQuickSelectSize);
    }

    /**
     * Creates the {@link Partition}. Parameters are derived from the {@code name}.
     *
     * @param name Name.
     * @return the {@link Partition} instance
     */
    static Partition createPartition(String name) {
        final int minQuickSelectSize = getMinQuickSelectSize(name);
        final int minHeapSelectSize = getMinHeapSelectSize(name);
        final PivotingStrategy s = getPivotStrategy(name);
        final KeyStrategy keyStrategy = getKeyStrategy(name);
        return new Partition(s, minQuickSelectSize, minHeapSelectSize, keyStrategy);
    }

    /**
     * Gets an integer number using trailing digits from a string.
     *
     * @param value Value.
     * @param defaultValue Default number.
     * @return the number (or the default)
     */
    static int getInteger(String value, int defaultValue) {
        int i = value.length();
        while (i > 0 && Character.isDigit(value.charAt(i - 1))) {
            i--;
        }
        if (i < value.length()) {
            return Integer.parseInt(value, i, value.length(), 10);
        }
        return defaultValue;
    }

    /**
     * Gets the minimum size for the recursive quickselect partition algorithm.
     * Below this size the algorithm will change strategy for partitioning,
     * e.g. change to a full sort.
     *
     * @param name Algorithm name.
     * @return the minimum quickselect size
     */
    static int getMinQuickSelectSize(String name) {
        final Matcher m = QS_PATTERN.matcher(name);
        if (m.find()) {
            return Integer.parseInt(name, m.start(1), m.end(1), 10);
        }
        return Partition.MIN_QUICKSELECT_SIZE;
    }

    /**
     * Gets the minimum size for the heapselect partition algorithm.
     *
     * @param name Algorithm name.
     * @return the minimum heapselect size
     */
    static int getMinHeapSelectSize(String name) {
        final Matcher m = HS_PATTERN.matcher(name);
        if (m.find()) {
            return Integer.parseInt(name, m.start(1), m.end(1), 10);
        }
        return Partition.MIN_HEAPSELECT_SIZE;
    }

    /**
     * Gets the pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name.
     * @return the pivot strategy
     */
    static PivotingStrategy getPivotStrategy(String name) {
        return getPivotStrategyOrEsle(name, PivotingStrategy.MEDIAN_OF_3);
    }

    /**
     * Gets the pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name.
     * @param defaultValue Default value.
     * @return the pivot strategy
     */
    static PivotingStrategy getPivotStrategyOrEsle(String name, PivotingStrategy defaultValue) {
        for (final PivotingStrategy s : PivotingStrategy.values()) {
            if (name.contains(s.name())) {
                return s;
            }
        }
        return defaultValue;
    }

    /**
     * Gets the sequential strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name.
     * @return the sequential strategy
     */
    static KeyStrategy getKeyStrategy(String name) {
        return getKeyStrategyOrElse(name, KeyStrategy.INDEX_SET);
    }

    /**
     * Gets the sequential strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name.
     * @param defaultValue Default value.
     * @return the sequential strategy
     */
    static KeyStrategy getKeyStrategyOrElse(String name, KeyStrategy defaultValue) {
        for (final KeyStrategy s : KeyStrategy.values()) {
            if (name.contains(s.name())) {
                return s;
            }
        }
        return defaultValue;
    }

    /**
     * Sort the values and compute the median.
     *
     * @param values Values.
     * @param p p-th quantiles to compute.
     * @return the quantiles
     */
    static double[] sortQuantile(double[] values, double[] p) {
        // Implicit NPE
        final int n = values.length;
        if (p.length == 0) {
            throw new IllegalArgumentException("No quantiles specified");
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
        final double[] x = values.clone();
        Arrays.sort(x);
        for (int i = 0; i < p.length; i++) {
            // EstimationMethod.HF6 (as per the Apache Commons Math Percentile
            // legacy implementation)
            final double pos = p[i] * (n + 1);
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double g = pos - fpos;
            if (j < 1) {
                q[i] = x[0];
            } else if (j >= n) {
                q[i] = x[n - 1];
            } else {
                q[i] = DoubleMath.interpolate(x[j - 1], x[j], g);
            }
        }
        return q;
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
     * Create the statistic using an array and given quantiles.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param quantiles Source of the quantiles.
     * @param bh Data sink.
     */
    @Benchmark
    public void quantiles(DoubleFunctionSource function, DataSource source,
            QuantileSource quantiles, Blackhole bh) {
        final double[][] data = source.getData();
        final double[] p = quantiles.getData();
        final BinaryOperator<double[]> fun = function.getFunction();
        for (final double[] x : data) {
            bh.consume(fun.apply(x, p));
        }
    }

    /**
     * Create the statistic using an array and given quantiles.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param quantiles Source of the quantiles.
     * @param bh Data sink.
     */
    @Benchmark
    public void quantileRange(DoubleFunctionSource function, DataSource source,
            QuantileRangeSource quantiles, Blackhole bh) {
        final double[][] data = source.getData();
        final double[] p = quantiles.getData();
        final BinaryOperator<double[]> fun = function.getFunction();
        for (final double[] x : data) {
            bh.consume(fun.apply(x, p));
        }
    }

    /**
     * Benchmark a sort on the data.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void sort(SortFunctionSource function, SortSource source, Blackhole bh) {
        final double[][] data = source.getData();
        final Consumer<double[]> fun = function.getFunction();
        for (final double[] x : data) {
            final double[] y = x.clone();
            fun.accept(y);
            bh.consume(y);
        }
    }

    /**
     * Benchmark partitioning using k partition indices.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void partition(KFunctionSource function, KSource source, Blackhole bh) {
        final double[][] data = source.getData();
        final int[][] indices = source.getIndices();
        final BiFunction<double[], int[], double[]> fun = function.getFunction();
        for (final double[] x : data) {
            for (final int[] i : indices) {
                bh.consume(fun.apply(x.clone(), i));
            }
        }
    }
}
