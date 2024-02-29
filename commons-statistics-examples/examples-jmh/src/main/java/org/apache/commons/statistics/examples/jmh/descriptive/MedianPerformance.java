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
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
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
    private static final String SP_NAN = "NSP";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect. */
    private static final String SBM = "SBM";
    /** Commons Statistics Median implementation with Bentley-McIlroy (original) quickselect. */
    private static final String BM = "BM";
    /** Commons Statistics Median implementation with a dual-pivot quickselect. */
    private static final String DP = "DP";
    /** Commons Statistics Median implementation with a dual-pivot quickselect
     * with 5 sorted points to choose pivots. */
    private static final String DP5 = "5DP";
    /** Commons Math Median implementation. */
    private static final String CM = "CM";
    /** Median implementation using the JDK sort function. */
    private static final String JDK = "JDK";

    // Second generation partition functions

    /** Commons Statistics Median implementation with Sedgewick's BM quickselect. */
    private static final String SBM2 = "2SBM";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String KSBM = "KSBM";
    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String K1SBM = "K1SBM";

    // Paired-key partition functions

    /** Commons Statistics Median implementation with Sedgewick's BM quickselect (paired-index variant). */
    private static final String PSBM = "PairedSBM";

    // Introselect functions

    /** Commons Statistics Median introselect implementation with Sedgewick's Bentley-McIlroy
     * partitioning, switching to heapselect when progress is poor. */
    private static final String ISBM = "ISBM";
    /** Commons Statistics Median introselect implementation with dual-pivot
     * partitioning, switching to heapselect when progress is poor. */
    private static final String IDP = "IDP";
    /** Commons Statistics Median implementation. This method is built using the best performing
     * select function across a range of input data. Current implementation uses
     * an introselect variant with a dual-pivot quickselect; switching to heapselect when
     * progress is poor. This algorithm currently cannot be configured. */
    private static final String SELECT = "SELECT";

    /**
     * Source of {@code double} array data.
     *
     * <p>This uses the same data class as {@link QuantilePerformance}.
     * This enables reuse of the various data distributions provided.
     */
    @State(Scope.Benchmark)
    public static class DataSource extends AbstractDataSource {
        /** Data length. */
        @Param({
            "10",
            "100",
            "1000"})
        private int length;

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }
    }

    /**
     * Source of a {@link ToDoubleFunction} for a {@code double[]}.
     */
    @State(Scope.Benchmark)
    public static class DoubleFunctionSource {
        /** Name of the source. */
        @Param({JDK, CM, SP, SP_NAN, SBM, BM, DP, DP5,
            SBM2,
            ISBM, IDP, SELECT
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
            if (JDK.equals(name)) {
                function = MedianPerformance::sortMedian;
            } else if (CM.equals(name)) {
                final org.apache.commons.math3.stat.descriptive.rank.Median m =
                    new org.apache.commons.math3.stat.descriptive.rank.Median();
                function = m::evaluate;
            // First generation kth-selector functions
            } else if (name.startsWith(SP)) {
                function = withKthSelector(name, SP)::evaluateSP;
            } else if (name.startsWith(SP_NAN)) {
                function = withKthSelector(name, SP_NAN)::evaluateSPN;
            } else if (name.startsWith(SBM)) {
                function = withKthSelector(name, SBM)::evaluateSBM;
            } else if (name.startsWith(BM)) {
                function = withKthSelector(name, BM)::evaluateBM;
            } else if (name.startsWith(DP)) {
                function = withKthSelector(name, DP)::evaluateDP;
            } else if (name.startsWith(DP5)) {
                function = withKthSelector(name, DP5)::evaluateDP5;
            // Second generation partition functions
            } else if (name.startsWith(SBM2)) {
                function = withPartition(name, SBM2)::evaluateSBM2;
            } else if (name.startsWith(KSBM)) {
                function = withPartition(name, KSBM)::evaluateKSBM;
            } else if (name.startsWith(K1SBM)) {
                function = withPartition(name, K1SBM)::evaluateK1SBM;
            // Paired key implementations
            } else if (name.startsWith(PSBM)) {
                function = withPartition(name, PSBM)::evaluatePairedSBM;
            // Introselect implementations
            } else if (name.startsWith(ISBM)) {
                function = withPartition(name, ISBM)::evaluateISBM;
            } else if (name.startsWith(IDP)) {
                function = withPartition(name, IDP)::evaluateIDP;
            } else if (name.startsWith(SELECT)) {
                function = withPartition(name, SELECT)::evaluate;
            } else {
                throw new IllegalStateException("Unknown double[] function: " + name);
            }
        }

        /**
         * Creates the {@link Median}.
         * Parameters for the {@link KthSelector} are derived from the {@code name}.
         *
         * @param name Name.
         * @param prefix Method prefix.
         * @return the {@link Median} instance
         */
        private static Median withKthSelector(String name, String prefix) {
            return Median.withDefaults()
                .withOverwrite(true)
                .withKthSelector(QuantilePerformance.createKthSelector(name, prefix));
        }

        /**
         * Creates the {@link Median}.
         * Parameters for the {@link Partition} are derived from the {@code name}.
         *
         * @param name Name.
         * @param prefix Method prefix.
         * @return the {@link Median} instance
         */
        private static Median withPartition(String name, String prefix) {
            return Median.withDefaults()
                .withOverwrite(true)
                .withPartition(QuantilePerformance.createPartition(name, prefix));
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
        Arrays.sort(values);
        final int k = n >>> 1;
        // Odd
        if ((n & 0x1) == 0x1) {
            return values[k];
        }
        // Even
        return (values[k - 1] + values[k]) * 0.5;
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
        final int size = source.size();
        final ToDoubleFunction<double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            bh.consume(fun.applyAsDouble(source.getData(j)));
        }
    }
}
