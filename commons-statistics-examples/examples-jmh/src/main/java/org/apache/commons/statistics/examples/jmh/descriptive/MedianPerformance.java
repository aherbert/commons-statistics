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
import java.util.function.ToDoubleFunction;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
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
 * Executes a benchmark of the creation of a median from {@code double} array data.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {"-server", "-Xms512M", "-Xmx8192M"})
public class MedianPerformance {
    /** Commons Statistics Median implementation using a single-pivot quickselect. */
    private static final String SP = "SP";
    /** Commons Statistics Median implementation with special NaN/zero handling. */
    private static final String SP_NAN = "SP_NaN";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect. */
    private static final String SBM = "SBM";
    /** Commons Statistics Median implementation with Bentley-McIlroy (original) quickselect. */
    private static final String BM = "BM";
    /** Commons Statistics Median implementation with a dual-pivot quickselect. */
    private static final String DP = "DP";
    /** Commons Statistics Median implementation with a dual-pivot quickselect
     * with 5 sorted points to choose pivots. */
    private static final String DP5 = "DP_5";
    /** Commons Math Median implementation. */
    private static final String CM = "CM";
    /** Median implementation using a sort. */
    private static final String SORT = "Sort";

    // Second generation partition functions

    /** Commons Statistics Median implementation with Sedgewick's BM quickselect. */
    private static final String RANGE_SBM = "RangeSBM";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect. */
    private static final String SBM2 = "2SBM";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String KSBM = "KSBM";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String K1SBM = "K1SBM";

    // Paired-key partition functions

    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String PSBM = "PairedSBM";

    // TODO: Use same data class as QuantilePerformance

    /**
     * Source of {@code double} array data.
     */
    @State(Scope.Benchmark)
    public static class DataSource {
        /** Fraction of data that is unique. */
        @Param({"0.9"})
        private double unique;

        /** Data length. */
        @Param({
            //"0", "1",
            //"10", "11",
            "10.5",
            //"100", "101",
            "100.5",
            //"1000", "1001",
            "1000.5"})
        private double length;

        /** Number of samples. */
        @Param({"100"})
        private int samples;

        /** Data. */
        private double[][] data;

        /** Data. */
        private int[][] intData;

        /**
         * @return the data
         */
        public double[][] getData() {
            return data;
        }

        /**
         * @return the int data
         */
        public int[][] getIntData() {
            return intData;
        }

        /**
         * Create the data.
         */
        @Setup(Level.Iteration)
        public void setup() {
            // Data will be randomized per iteration
            final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
            intData = new int[samples][];
            data = new double[samples][];
            final int n = (int) Math.floor(length);
            final boolean split = n != length;
            if (split && (n & 0x1) != 0) {
                throw new IllegalStateException("Split length requires floor(length) to be even");
            }
            for (int i = 0; i < samples; i++) {
                // For non-integer length, alternate the size N or N+1
                final int n1 = split ? n + (i & 0x1) : n;
                if (unique >= 1.0) {
                    // Effectively unique
                    intData[i] = rng.ints(n1).toArray();
                } else {
                    // Set a bound on random integer data to create duplicates
                    final int bound = (int) Math.ceil(unique * length);
                    if (bound < 1) {
                        intData[i] = new int[n1];
                        Arrays.fill(intData[i], rng.nextInt());
                    } else {
                        intData[i] = rng.ints(n1, 0, bound).toArray();
                    }
                }
                data[i] = Arrays.stream(intData[i]).asDoubleStream().toArray();
            }
        }
    }

    /**
     * Source of a {@link ToDoubleFunction} for a {@code double[]}.
     */
    @State(Scope.Benchmark)
    public static class DoubleFunctionSource {
        /** Name of the source. */
        @Param({SORT, CM, SP, SP_NAN, SBM, BM, DP, DP5,
            SBM2,
            // With many iterations (of data) this is slower then the default median-of-3 strategy
            //MEDIAN_CENTRAL,
            // Not obviously better
            //MEDIAN_DYNAMIC
        })
        private String name;

        /** The action. */
        private ToDoubleFunction<double[]> function;

        /**
         * @return the function
         */
        public ToDoubleFunction<double[]> getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            // Note: Functions defensively copy the data by default
            // Note: KeyStratgey does not matter for single / paired keys but
            // we set it anyway for completeness.
            Objects.requireNonNull(name);
            if (SORT.equals(name)) {
                function = MedianPerformance::sortMedian;
            } else if (CM.equals(name)) {
                final org.apache.commons.math3.stat.descriptive.rank.Median m =
                    new org.apache.commons.math3.stat.descriptive.rank.Median();
                function = m::evaluate;
            } else if (name.startsWith(SP)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateSP;
            } else if (name.startsWith(SP_NAN)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateSPN;
            } else if (name.startsWith(SBM)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateSBM;
            } else if (name.startsWith(BM)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateBM;
            } else if (name.startsWith(DP)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateDP;
            } else if (name.startsWith(DP5)) {
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                function = Median.withDefaults().withKthSelector(new KthSelector(s))::evaluateDP5;
            // Second generation partition functions (allow configuration)
            } else if (name.startsWith(RANGE_SBM)) {
                final int minSelectSize = QuantilePerformance.getMinQuickSelectSize(name);
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                final KeyStrategy keyStrategy = QuantilePerformance.getKeyStrategy(name);
                function = Median.withDefaults().withPartition(new Partition(s, minSelectSize, keyStrategy))
                    ::evaluateRangeSBM;
            } else if (name.startsWith(SBM2)) {
                final int minSelectSize = QuantilePerformance.getMinQuickSelectSize(name);
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                final KeyStrategy keyStrategy = QuantilePerformance.getKeyStrategy(name);
                function = Median.withDefaults().withPartition(new Partition(s, minSelectSize, keyStrategy))
                    ::evaluateSBM2;
            } else if (name.startsWith(KSBM)) {
                final int minSelectSize = QuantilePerformance.getMinQuickSelectSize(name);
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                final KeyStrategy keyStrategy = QuantilePerformance.getKeyStrategy(name);
                function = Median.withDefaults().withPartition(new Partition(s, minSelectSize, keyStrategy))
                    ::evaluateKSBM;
            } else if (name.startsWith(K1SBM)) {
                final int minSelectSize = QuantilePerformance.getMinQuickSelectSize(name);
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                final KeyStrategy keyStrategy = QuantilePerformance.getKeyStrategy(name);
                function = Median.withDefaults().withPartition(new Partition(s, minSelectSize, keyStrategy))
                    ::evaluateK1SBM;
            // Paired key implementations
            } else if (name.startsWith(PSBM)) {
                final int minSelectSize = QuantilePerformance.getMinQuickSelectSize(name);
                final PivotingStrategy s = QuantilePerformance.getPivotStrategy(name);
                final KeyStrategy keyStrategy = QuantilePerformance.getKeyStrategy(name);
                function = Median.withDefaults().withPartition(new Partition(s, minSelectSize, keyStrategy))
                    ::evaluatePairedSBM;
            } else {
                throw new IllegalStateException("Unknown double[] function: " + name);
            }
        }
    }

    /**
     * Sort the values and compute the median.
     *
     * @param values Values.
     * @return the median
     */
    static double sortMedian(double[] values) {
        // Implicit NPE
        final int n = values.length;
        // Special cases
        if (n <= 2) {
            switch (n) {
            case 2:
                return (values[0] + values[1]) * 0.5;
            case 1:
                return values[0];
            default:
                return Double.NaN;
            }
        }
        // A sort is required
        final double[] x = values.clone();
        Arrays.sort(x);
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            return x[k];
        }
        // Even
        return (x[k - 1] + x[k]) * 0.5;
    }

    /**
     * Create the statistic using an array.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void arrayDoubleStatistic(DoubleFunctionSource function, DataSource source, Blackhole bh) {
        final double[][] data = source.getData();
        final ToDoubleFunction<double[]> fun = function.getFunction();
        for (final double[] x : data) {
            bh.consume(fun.applyAsDouble(x));
        }
    }
}
