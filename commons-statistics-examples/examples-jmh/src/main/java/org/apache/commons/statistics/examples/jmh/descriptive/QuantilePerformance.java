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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.PermutationSampler;
import org.apache.commons.rng.sampling.distribution.DiscreteUniformSampler;
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler;
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
    private static final String DP5 = "5DP";
    /** Commons Math Percentile implementation. */
    private static final String CM = "CM";
    /** Partition implementation using a single-pivot strategy with Dutch National Flag partitioning. */
    private static final String DNF = "DNF";
    /** Use the JDK sort function. */
    private static final String JDK = "JDK";
    /** Use a sort function. */
    private static final String SORT = "Sort";
    /** Baseline for the benchmark. */
    private static final String BASELINE = "Baseline";

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

    /** Commons Statistics Quantile introselect implementation with Sedgewick's Bentley-McIlroy
     * partitioning, switching to heapselect when progress is poor. */
    private static final String ISBM = "ISBM";
    /** Commons Statistics Quantile introselect implementation with Dutch National Flag partitioning
     * partitioning, switching to heapselect when progress is poor. The DNF algorithm is appened
     * as a suffix. */
    private static final String IDNF = "IDNF";
    /** Commons Statistics Quantile introselect implementation with dual-pivot
     * partitioning, switching to heapselect when progress is poor. */
    private static final String IDP = "IDP";
    /** Commons Statistics Quantile implementation. This method is built using the best performing
     * select function across a range of input data. Current implementation uses
     * an introselect variant with a dual-pivot quickselect; switching to heapselect when
     * progress is poor. This algorithm currently cannot be configured. */
    private static final String SELECT = "SELECT";

    /** Pattern for the minimum quickselect size. */
    private static final Pattern QS_PATTERN = Pattern.compile("QS(\\d+)");
    /** Pattern for the heapselect shift. */
    private static final Pattern HS_PATTERN = Pattern.compile("HS(\\d+)");
    /** Pattern for the heapselect constant. */
    private static final Pattern HC_PATTERN = Pattern.compile("HC(\\d+)");
    /** Pattern for the heapselect mask shift. */
    private static final Pattern MS_PATTERN = Pattern.compile("MS(\\d+)");
    /** Pattern for the recursion multiple (simple float format). */
    private static final Pattern RM_PATTERN = Pattern.compile("RM(\\d+\\.?\\d*)");
    /** Pattern for the recursion constant. */
    private static final Pattern RC_PATTERN = Pattern.compile("RC(\\d+)");
    /** Pattern for the compression level. */
    private static final Pattern CL_PATTERN = Pattern.compile("CL(\\d+)");

    /** Random source. */
    private static final RandomSource RANDOM_SOURCE = RandomSource.XO_RO_SHI_RO_128_PP;

    /**
     * Source of {@code double} array data.
     *
     * <p>This uses the adverse input test suite from figure 1 in Bentley and McIlroy
     * (1993) Engineering a sort function, SOFTWARE—PRACTICE AND EXPERIENCE, VOL.23(11),
     * 1249–1265.
     *
     * <p>Note
     *
     * <p>This class has setter methods to allow re-use in unit testing without requiring
     * use of reflection to set fields. Parameters set by JMH are initialized to their
     * defaults for convenience. Re-use requires:
     *
     * <ol> <li>Creating an instance of the abstract class that provides the data length
     * <li>Calling {@link #setup()} to create the data <li>Iterating over the data </ol>
     *
     * <pre>
     * AbstractDataSource s = new AbstractDataSource() {
     *     protected int getLength() {
     *         return 123;
     *     }
     * };
     * s.setDistribution(Distribution.SAWTOOTH, Distribution.SHUFFLE);
     * s.setModification(Modification.REVERSE_FRONT);
     * s.setRange(2);
     * s.setup();
     * for (int i = 0; i &lt; s.size(); i++) {
     *     s.getData(i);
     * }
     * </pre>
     *
     * <p>Random distribution mode
     *
     * <p>The default configuration includes random samples generated as a family of
     * single samples created from ranges that are powers of two [0, 2^i). This small set
     * of samples is only a small representation of randomness. For small lengths this may
     * only be a few random samples.
     *
     * <p>The data source can be changed to generate a fixed number of random samples
     * using a uniform distribution [0, n]. For this purpose the distribution must be set
     * to {@link Distribution#RANDOM} and the {@link #setSamples(int) samples} set above
     * zero. The length range is ignored. The inclusive upper bound {@code n} is set using
     * the {@link #setSeed(int) seed}. If this is zero then the default is
     * {@link Integer#MAX_VALUE}.
     */
    @State(Scope.Benchmark)
    public abstract static class AbstractDataSource {
        /** All distributions / modifications. */
        private static final String ALL = "all";
        /** Fixed seed for the random source. */
        private static final byte[] SEED = RANDOM_SOURCE.createSeed();

        /**
         * The type of distribution.
         */
        enum Distribution {
            /** sawtooth distribution. */
            SAWTOOTH,
            /** random distribution. */
            RANDOM,
            /** stagger distribution. */
            STAGGER,
            /** plateau distribution. */
            PLATEAU,
            /** shuffle distribution. */
            SHUFFLE;
        }

        /**
         * The type of data modification.
         */
        enum Modification {
            /** copy modification. */
            COPY,
            /** reverse modification. */
            REVERSE,
            /** reverse front-half modification. */
            REVERSE_FRONT,
            /** reverse back-half modification. */
            REVERSE_BACK,
            /** sort modification. */
            SORT,
            /** dither modification. */
            DITHER;
        }

        /** Type of data. Multiple types can be specified in the same string using
         * lower/upper case, delimited using ':'. */
        @Param({ALL})
        private String distribution = ALL;

        /** Type of data modification. Multiple types can be specified in the same string using
         * lower/upper case, delimited using ':'. */
        @Param({ALL})
        private String modification = ALL;

        /** Extra range to add to the data length.
         * E.g. Use 1 to force use of odd and even length samples for the median. */
        @Param({"1"})
        private int range = 1;

        /** Sample 'seed'. This is {@code m} in Bentley and McIlroy's test suite.
         * If set to zero the default is to use powers of 2 based on sample size. */
        @Param({"0"})
        private int seed;

        /** Number of samples. Applies only to the random distribution. */
        @Param({"0"})
        private int samples;

        /** Data. This is stored as integer data which saves memory. Note that when ranking
         * data it is not necessary to have the full range of the double data type; the same
         * number of unique values can be recorded in an array using an integer type.
         * Returning a double[] forces a copy to be generated for destructive sorting /
         * partitioning methods. */
        private int[][] data;

        /**
         * Gets the sample for the given {@code index}.
         *
         * @param index Index.
         * @return the data sample
         */
        public double[] getData(int index) {
            final int[] a = data[index];
            final double[] x = new double[a.length];
            for (int i = -1; ++i < a.length;) {
                x[i] = a[i];
            }
            return x;
        }

        /**
         * Get the number of data samples.
         *
         * @return the number of samples
         */
        public int size() {
            return data.length;
        }

        /**
         * Create the data.
         */
        @Setup(Level.Iteration)
        public void setup() {
            Objects.requireNonNull(distribution);
            Objects.requireNonNull(modification);

            // Set-up using parameters (may throw)
            final EnumSet<Distribution> dist = getDistributions();
            final int length = getLength();
            if (length < 1) {
                throw new IllegalStateException("Unsupported length: " + length);
            }

            // Special case for random distribution mode
            if (dist.contains(Distribution.RANDOM) && dist.size() == 1 && samples > 0) {
                final UniformRandomProvider rng = RANDOM_SOURCE.create();
                data = new int[samples][length];
                final int upper = seed > 0 ? seed : Integer.MAX_VALUE;
                final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, upper);
                for (int i = 0; i < data.length; i++) {
                    final int[] a = data[i];
                    for (int j = a.length; --j >= 0;) {
                        a[j] = s.sample();
                    }
                }
                return;
            }

            // Only run per iteration for random distribution mode
            if (data != null) {
                return;
            }

            final EnumSet<Modification> mod = getModifications();
            // Note: Bentley-McIlroy use n in {100, 1023, 1024, 1025}.
            // Here we only support a continuous range. The range is important
            // for the median as it will require one or two points to partition
            // if the length is odd or even.
            final int r = range > 0 ? range : 0;
            if (length + (long) r > Integer.MAX_VALUE) {
                throw new IllegalStateException("Unsupported upper length: " + length);
            }
            final int length2 = length + r;

            // Data using the RNG will be randomized only once.
            // Here we use the same seed for parity across methods.
            // Note that most distributions do not use the source of randomness.
            final UniformRandomProvider rng = RANDOM_SOURCE.create(SEED);
            final ArrayList<int[]> sampleData = new ArrayList<>();
            for (int n = length; n <= length2; n++) {
                // Note: Large lengths may wish to limit the range of m to limit
                // the memory required to store the samples. Currently a single
                // m is supported via the seed parameter.
                // Default seed will create ceil(log2(2*n)) * 5 dist * 6 mods samples:
                // MAX  = 32 * 5 * 6 * (2^31-1) * 4 bytes == 7679 GiB
                // HUGE = 31 * 5 * 6 * 2^30 * 4 bytes == 3720 GiB
                // BIG  = 21 * 5 * 6 * 2^20 * 4 bytes == 2520 MiB  <-- within configured JVM -Xmx
                // MED  = 11 * 5 * 6 * 2^10 * 4 bytes == 1320 KiB
                // It is possible to create lengths above 2^30 using a single distribution,
                // modification, and seed:
                // MAX1 = 1 * 1 * 1 * (2^31-1) * 4 bytes == 8191 MiB
                // However this is then used to create double[] data thus requiring an extra
                // ~16GiB memory for the sample to partition.
                for (final int m : createSeeds(seed, n)) {
                    for (final int[] x : createDistributions(dist, rng, n, m)) {
                        if (mod.contains(Modification.COPY)) {
                            // Don't copy! All other methods generate copies
                            // so we can use this in-place.
                            sampleData.add(x);
                        }
                        if (mod.contains(Modification.REVERSE)) {
                            sampleData.add(reverse(x, 0, n));
                        }
                        if (mod.contains(Modification.REVERSE_FRONT)) {
                            sampleData.add(reverse(x, 0, n / 2));
                        }
                        if (mod.contains(Modification.REVERSE_BACK)) {
                            sampleData.add(reverse(x, n / 2, n));
                        }
                        if (mod.contains(Modification.SORT)) {
                            sampleData.add(sort(x));
                        }
                        if (mod.contains(Modification.DITHER)) {
                            sampleData.add(dither(x));
                        }
                    }
                }
            }
            data = sampleData.toArray(int[][]::new);
        }

        /**
         * @return the distributions
         */
        private EnumSet<Distribution> getDistributions() {
            return getEnumFromParam(Distribution.class, distribution);
        }

        /**
         * @return the modifications
         */
        private EnumSet<Modification> getModifications() {
            return getEnumFromParam(Modification.class, modification);
        }

        /**
         * Gets all the enum values of the given class from the parameters.
         *
         * @param <E> Enum type.
         * @param cls Class of the enum.
         * @param parameters Parameters (multiple values delimited by ':')
         * @return the enum values
         */
        static <E extends Enum<E>> EnumSet<E> getEnumFromParam(Class<E> cls, String parameters) {
            if (ALL.equals(parameters)) {
                return EnumSet.allOf(cls);
            }
            final EnumSet<E> set = EnumSet.noneOf(cls);
            final String s = parameters.toUpperCase(Locale.ROOT);
            for (final E e : cls.getEnumConstants()) {
                // Scan for the name
                for (int i = s.indexOf(e.name(), 0); i >= 0; i = s.indexOf(e.name(), i)) {
                    // Ensure a full match to the name:
                    // either at the end of the string, or followed by the delimiter
                    i += e.name().length();
                    if (i == s.length() || s.charAt(i) == ':') {
                        set.add(e);
                        break;
                    }
                }
            }
            if (set.isEmpty()) {
                throw new IllegalStateException("Unknown parameters: " + parameters);
            }
            return set;
        }

        /**
         * Creates the seeds.
         *
         * <p>This can be pasted into a JShell terminal to verify it works for any size
         * {@code 1 <= n < 2^31}. With the default behaviour all seeds {@code m} are
         * strictly positive powers of 2 and the highest seed should be below {@code 2*n}.
         *
         * @param seed Seed (use 0 for default; or provide a strictly positive {@code 1 <= m <= 2^31}).
         * @param n Sample size.
         * @return the seeds
         */
        private static int[] createSeeds(int seed, int n) {
            // Allow [1, 2^31] (note 2^31 is negative but handled as a power of 2)
            if (seed - 1 >= 0) {
                return new int[] {seed};
            }
            // Bentley-McIlroy use:
            // for: m = 1; m < 2 * n; m *= 2
            // This has been modified here to handle n up to MAX_VALUE
            // by knowing the count of m to generate.

            // ceil(log2(n)) + 1 == ceil(log2(2*n)) but handles MAX_VALUE
            int c = 33 - Integer.numberOfLeadingZeros(n - 1);
            final int[] seeds = new int[c];
            c = 0;
            for (int m = 1; c != seeds.length; m *= 2) {
                seeds[c++] = m;
            }
            return seeds;
        }

        /**
         * Creates the distribution samples. Handles {@code m = 2^31} using {@link Integer#MIN_VALUE}.
         *
         * @param dist Distributions.
         * @param rng Source of randomness.
         * @param n Length of the sample.
         * @param m Sample seed (in [1, 2^31])
         * @return the samples
         */
        private List<int[]> createDistributions(EnumSet<Distribution> dist, UniformRandomProvider rng, int n, int m) {
            final ArrayList<int[]> distData = new ArrayList<>(5);
            int[] x;
            if (dist.contains(Distribution.SAWTOOTH)) {
                distData.add(x = new int[n]);
                // i % m
                // Typical case m is a power of 2 so we can use a mask
                final int mask = m - 1;
                if ((m & mask) == 0) {
                    for (int i = -1; ++i < n;) {
                        x[i] = i & mask;
                    }
                } else {
                    // User input seed
                    for (int i = -1; ++i < n;) {
                        x[i] = i % m;
                    }
                }
            }
            if (dist.contains(Distribution.RANDOM)) {
                distData.add(x = new int[n]);
                // rand() % m
                // A sampler is faster than rng.nextInt(m); the sampler has an inclusive upper.
                final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, m - 1);
                for (int i = -1; ++i < n;) {
                    x[i] = s.sample();
                }
            }
            if (dist.contains(Distribution.STAGGER)) {
                distData.add(x = new int[n]);
                // Overflow safe: (i * m + i) % n
                final long nn = n;
                for (int i = -1; ++i < n;) {
                    x[i] = (int) (Integer.toUnsignedLong(i * m + i) % nn);
                }
            }
            if (dist.contains(Distribution.PLATEAU)) {
                distData.add(x = new int[n]);
                // min(i, m)
                for (int i = Math.min(n, m); --i >= 0;) {
                    x[i] = i;
                }
                for (int i = m - 1; ++i < n;) {
                    x[i] = m;
                }
            }
            if (dist.contains(Distribution.SHUFFLE)) {
                distData.add(x = new int[n]);
                // rand() % m ? (j += 2) : (k += 2)
                final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, m - 1);
                for (int i = -1, j = 0, k = 1; ++i < n;) {
                    x[i] = s.sample() != 0 ? (j += 2) : (k += 2);
                }
            }
            return distData;
        }

        /**
         * Return a (part) reversed copy of the data.
         *
         * @param x Data.
         * @param from Start index to reverse (inclusive).
         * @param to End index to reverse (exclusive).
         * @return the copy
         */
        private static int[] reverse(int[] x, int from, int to) {
            final int[] a = x.clone();
            for (int i = from - 1, j = to; ++i < --j;) {
                final int v = a[i];
                a[i] = a[j];
                a[j] = v;
            }
            return a;
        }

        /**
         * Return a sorted copy of the data.
         *
         * @param x Data.
         * @return the copy
         */
        private static int[] sort(int[] x) {
            final int[] a = x.clone();
            Arrays.sort(a);
            return a;
        }

        /**
         * Return a dithered copy of the data.
         *
         * @param x Data.
         * @return the copy
         */
        private static int[] dither(int[] x) {
            final int[] a = x.clone();
            for (int i = a.length; --i >= 0;) {
                // Bentley-McIlroy use i % 5.
                // It is important this is not a power of 2 so it will not coincide
                // with patterns created in the data using the default m powers-of-2.
                a[i] += i % 5;
            }
            return a;
        }

        /**
         * Gets the minimum length of the data.
         * The actual length is enumerated in {@code [length, length + range]}.
         *
         * @return the length
         */
        protected abstract int getLength();

        /**
         * Sets the distribution(s) of the data.
         * If the input is an empty array or the first enum value is null,
         * then all distributions are used.
         *
         * @param v Values.
         */
        void setDistribution(Distribution... v) {
            if (v.length == 0 || v[0] == null) {
                distribution = ALL;
            } else {
                final EnumSet<Distribution> s = EnumSet.of(v[0], v);
                distribution = s.stream().map(Enum::name).collect(Collectors.joining(":"));
            }
        }

        /**
         * Sets the modification of the data.
         * If the input is an empty array or the first enum value is null,
         * then all distributions are used.
         *
         * @param v Value.
         */
        void setModification(Modification... v) {
            if (v.length == 0 || v[0] == null) {
                modification = ALL;
            } else {
                final EnumSet<Modification> s = EnumSet.of(v[0], v);
                modification = s.stream().map(Enum::name).collect(Collectors.joining(":"));
            }
        }

        /**
         * Sets the maximum addition to extend the length of each sample of data.
         * The actual length is enumerated in {@code [length, length + range]}.
         *
         * <p>Supports positive values and the edge case of {@link Integer#MIN_VALUE}
         * which is treated as an unisgned power of 2.
         *
         * @param v Value.
         */
        void setRange(int v) {
            range = v;
        }

        /**
         * Sets the sample 'seed' used to generate distributions.
         * If set to zero the default is to use powers of 2 based on sample size.
         *
         * <p>Supports positive values and the edge case of {@link Integer#MIN_VALUE}
         * which is treated as an unsigned power of 2.
         *
         * @param v Value (ignored if not within {@code [1, 2^31]}).
         */
        void setSeed(int v) {
            seed = v;
        }

        /**
         * Sets the number of samples to use for the random distribution mode.
         * See {@link AbstractDataSource} for details.
         *
         * @param v Value.
         */
        void setSamples(int v) {
            seed = v;
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
            ISBM, IDP, SELECT})
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
            // Note: Functions should not defensively copy the data
            // as a clone is passed in from the data source.
            if (JDK.equals(name)) {
                function = QuantilePerformance::sortQuantile;
            } else if (CM.equals(name)) {
                // No way to avoid a data copy here. CM does
                // defensive copying for most array input. This
                // method is copied in SPH which overwrites the data.
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
                function = withKthSelector(name, SPH)::evaluateSPH;
            } else if (name.startsWith(SPE)) {
                function = withKthSelector(name, SPE)::evaluateSPE;
            } else if (name.startsWith(SP)) {
                function = withKthSelector(name, SP)::evaluateSP;
            } else if (name.startsWith(BM)) {
                function = withKthSelector(name, BM)::evaluateBM;
            } else if (name.startsWith(SBM)) {
                function = withKthSelector(name, SBM)::evaluateSBM;
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
            final UniformRandomProvider rng = RANDOM_SOURCE.create();
            final PermutationSampler s = new PermutationSampler(rng, getLength(), k);
            indices = s.samples(repeats).toArray(int[][]::new);
        }
    }

    /**
     * Source of k-th indices to be searched by an {@link IndexInterval}.
     */
    @State(Scope.Benchmark)
    public static class IndexSource {
        /** Upper bound (exclusive) on the indices. */
        @Param({"1000", "1000000"})
        private int length;
        /** Number of indices to select. */
        @Param({"5", "10", "20", "40"})
        private int k;
        /** Number of repeats. */
        @Param({"10"})
        private int repeats;

        /** Indices. */
        private int[][] indices;
        /** Search points. */
        private int[][] points;

        /**
         * @return the indices
         */
        public int[][] getIndices() {
            return indices;
        }

        /**
         * @return the search points
         */
        public int[][] getPoints() {
            return points;
        }

        /**
         * Create the indices and search points.
         */
        @Setup(Level.Iteration)
        public void setup() {
            if (k < 2) {
                throw new IllegalStateException("Require multiple indices");
            }
            // Data will be randomized per iteration
            final UniformRandomProvider rng = RANDOM_SOURCE.create();
            final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, length - 1);

            indices = new int[repeats][];
            points = new int[repeats][];

            for (int i = repeats; --i >= 0;) {
                // Indices with possible repeats
                final int[] x = new int[k];
                for (int j = k; --j >= 0;) {
                    x[j] = s.sample();
                }

                // Get the sorted unique indices
                final int[] y = x.clone();
                final int unique = Sorting.sortIndices(y, k);

                // Create the cut points between each unique index
                final int[] p = new int[unique - 1];
                for (int j = 0; j < p.length; j++) {
                    p[j] = (y[j] + y[j + 1]) >>> 1;
                }
                shuffle(rng, p);

                indices[i] = x;
                points[i] = p;
            }
        }
        /**
         * Shuffles the entries of the given array.
         *
         * @param rng Source of randomness.
         * @param array Array whose entries will be shuffled (in-place).
         */
        private static void shuffle(UniformRandomProvider rng, int[] array) {
            for (int i = array.length; i > 1; i--) {
                swap(array, i - 1, rng.nextInt(i));
            }
        }

        /**
         * Swaps the two specified elements in the array.
         *
         * @param array Array.
         * @param i First index.
         * @param j Second index.
         */
        private static void swap(int[] array, int i, int j) {
            final int tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }
    /**
     * Source of an {@link IndexInterval}.
     */
    @State(Scope.Benchmark)
    public static class IndexIntervalSource {
        /** Name of the source. */
        @Param({"ScanningKeyIndexInterval",
            "BinarySearchKeyIndexInterval",
            "IndexSet",
            "CompressedIndexSet"})
        private String name;

        /** The factory. */
        private Function<int[], IndexInterval> factory;

        /**
         * @param indices Indices.
         * @return {@link IndexInterval}
         */
        public IndexInterval create(int[] indices) {
            return factory.apply(indices);
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            if ("ScanningKeyIndexInterval".equals(name)) {
                factory = k -> {
                    k = k.clone();
                    final int unique = Sorting.sortIndices(k, k.length);
                    return ScanningKeyIndexInterval.of(k, unique);
                };
            } else if ("BinarySearchKeyIndexInterval".equals(name)) {
                factory = k -> {
                    k = k.clone();
                    final int unique = Sorting.sortIndices(k, k.length);
                    return BinarySearchKeyIndexInterval.of(k, unique);
                };
            } else if ("IndexSet".equals(name)) {
                factory = k -> {
                    return IndexSet.of(k);
                };
            } else if (name.startsWith("CompressedIndexSet")) {
                final int c = getCompression(name);
                factory = k -> {
                    return CompressedIndexSet.of(c, k);
                };
            } else {
                throw new IllegalStateException("Unknown IndexInterval: " + name);
            }
        }

        /**
         * Gets the compression from the last character of the name.
         *
         * @param name Name.
         * @return the compression
         */
        private static int getCompression(String name) {
            final char c = name.charAt(name.length() - 1);
            if (Character.isDigit(c)) {
                return Character.digit(c, 10);
            }
            return 1;
        }
    }

    /**
     * Source of a range of positions to partition. These are positioned away from the edge
     * using a power of 2 shift.
     *
     * <p>This is a specialised class to allow benchmarking the switch from using
     * quickselect partitioning to using heapselect.
     */
    @State(Scope.Benchmark)
    public static class EdgeSource extends AbstractDataSource {
        /** Data length. */
        @Param({"1023"})
        private int length;
        /** Shift applied to the length to find k. */
        @Param({"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        private int shift;
        /** Target indices. */
        private IndexInterval[] indices;

        /**
         * @return the target indices
         */
        public IndexInterval[] getIndices() {
            return indices;
        }

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }

        /**
         * Create the data and check the indices are not at the end.
         */
        @Override
        @Setup
        public void setup() {
            super.setup();
            // Error for a bad configuration
            final int k = length >>> shift;
            if (k == 0) {
                throw new IllegalStateException(length + " >>> " + shift + " == 0");
            }
            // Create a single index at both ends
            // TODO - support specifying a range: [ka, kb]
            final int k1 = length - 1 - k;
            indices = new IndexInterval[] {
                ScanningKeyIndexInterval.of(new int[] {k}, 1),
                ScanningKeyIndexInterval.of(new int[] {k1}, 1),
            };
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
            //"InsertionSortIF", "InsertionSortIT", "InsertionSort", "InsertionSortB"
            //"BM25"
            ISBM, IDP,
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
                function = createKthSelector(name, SP)::sortSP;
            } else if (name.startsWith(SBM)) {
                function = createKthSelector(name, SBM)::sortSBM;
            } else if (name.startsWith(BM)) {
                function = createKthSelector(name, BM)::sortBM;
            } else if (name.startsWith(DP)) {
                function = createKthSelector(name, DP)::sortDP;
            } else if (name.startsWith(DP5)) {
                function = createKthSelector(name, DP5)::sortDP5;
            } else if (name.startsWith(DNF)) {
                function = createKthSelector(name, DNF)::sortDNF;
            // 2nd generation partition functions
            } else if (name.startsWith(SBM2)) {
                function = createPartition(name, SBM2)::sortSBM;
            // Introsort
            } else if (name.startsWith(ISBM)) {
                function = createPartition(name, ISBM)::sortISBM;
            } else if (name.startsWith(IDNF)) {
                // 3 variants
                if (name.startsWith(IDNF + "3")) {
                    function = createPartition(name, IDNF + "3")::sortIDNF3;
                } else if (name.startsWith(IDNF + "2")) {
                    function = createPartition(name, IDNF + "2")::sortIDNF2;
                } else if (name.startsWith(IDNF + "1")) {
                    function = createPartition(name, IDNF + "1")::sortIDNF1;
                }
            } else if (name.startsWith(IDP)) {
                function = createPartition(name, IDP)::sortIDP;
            // Insertion sort variations.
            // For parity with the internal version these all use the same (shorter) data
            } else if ("InsertionSortIF".equals(name)) {
                function = x -> {
                    // Ignored sentinal
                    x[0] = Double.NEGATIVE_INFINITY;
                    Sorting.sort(x, 1, x.length - 1, false);
                };
            } else if ("InsertionSortIT".equals(name)) {
                // Internal version
                function = x -> {
                    // Add a sentinal
                    x[0] = Double.NEGATIVE_INFINITY;
                    Sorting.sort(x, 1, x.length - 1, true);
                };
            } else if ("InsertionSort".equals(name)) {
                function = x -> {
                    x[0] = Double.NEGATIVE_INFINITY;
                    Sorting.sort(x, 1, x.length - 1);
                };
            } else if ("InsertionSortB".equals(name)) {
                function = x -> {
                    x[0] = Double.NEGATIVE_INFINITY;
                    Sorting.sortb(x, 1, x.length - 1);
                };
            }
            if (function == null) {
                throw new IllegalStateException("Unknown sort function: " + name);
            }
        }
    }


    /**
     * Source of a sort function.
     */
    @State(Scope.Benchmark)
    public static class Sort5FunctionSource {
        /** Name of the source. */
        @Param({"sort5", "sort5b",
            //"sort", "sort5head"
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
            // Note: We do not run this on input of length 5. We can run it on input of
            // any length above 5. So we choose indices using a spacing of 1/4 of the range.
            // Since we do this for all methods it is a fixed overhead. This allows use
            // of a variety of data sizes.
            if ("sort5".equals(name)) {
                function = x -> {
                    final int s = x.length >> 2;
                    Sorting.sort5(x, 0, s, s << 1, x.length - 1 - s, x.length - 1);
                };
            } else if ("sort5b".equals(name)) {
                function = x -> {
                    final int s = x.length >> 2;
                    Sorting.sort5b(x, 0, s, s << 1, x.length - 1 - s, x.length - 1);
                };
            } else if ("sort".equals(name)) {
                function = x -> Sorting.sort(x, 0, 4);
            } else if ("sort5head".equals(name)) {
                function = x -> Sorting.sort5(x, 0, 1, 2, 3, 4);
            } else {
                throw new IllegalStateException("Unknown sort5 function: " + name);
            }
        }
    }

    /**
     * Source of a k-th selector function.
     */
    @State(Scope.Benchmark)
    public static class KFunctionSource {
        /** Name of the source. */
        @Param({SORT + JDK, SPH,
            SP, BM, SBM,
            DP, DP5, DNF,
            SBM2, KSBM, K1SBM,
            PSBM,
            ISBM, IDP, SELECT})
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
            if (name.equals(BASELINE)) {
                function = (data, indices) -> extractIndices(data, indices.clone());
            } else  if (name.startsWith(SORT)) {
                // Sort variants (do not clone the keys)
                if (name.contains(ISBM)) {
                    final Partition part = createPartition(name.substring(SORT.length()), ISBM);
                    function = (data, indices) -> {
                        part.sortISBM(data);
                        return extractIndices(data, indices);
                    };
                } else if (name.contains(IDP)) {
                    final Partition part = createPartition(name.substring(SORT.length()), IDP);
                    function = (data, indices) -> {
                        part.sortIDP(data);
                        return extractIndices(data, indices);
                    };
                } else if (name.contains(JDK)) {
                    function = (data, indices) -> {
                        Arrays.sort(data);
                        return extractIndices(data, indices);
                    };
                }
            // First generation kth-selector functions
            } else if (name.startsWith(SPH)) {
                // Ported CM implementation with a heap
                final KthSelector selector = createKthSelector(name, SPH);
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
            // The following methods clone the indices to avoid destructive modification
            } else if (name.startsWith(SP)) {
                final KthSelector selector = createKthSelector(name, SP);
                function = (data, indices) -> {
                    selector.partitionSP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(BM)) {
                final KthSelector selector = createKthSelector(name, BM);
                function = (data, indices) -> {
                    selector.partitionBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(SBM)) {
                final KthSelector selector = createKthSelector(name, SBM);
                function = (data, indices) -> {
                    selector.partitionSBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP)) {
                final KthSelector selector = createKthSelector(name, DP);
                function = (data, indices) -> {
                    selector.partitionDP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP5)) {
                final KthSelector selector = createKthSelector(name, DP5);
                function = (data, indices) -> {
                    selector.partitionDP5(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DNF)) {
                final KthSelector selector = createKthSelector(name, DNF);
                function = (data, indices) -> {
                    selector.partitionDNF(data, indices.clone());
                    return extractIndices(data, indices);
                };
            // Second generation partition functions
            } else if (name.startsWith(SBM2)) {
                final Partition part = createPartition(name, SBM2);
                function = (data, indices) -> {
                    part.partitionSBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(KSBM)) {
                final Partition part = createPartition(name, KSBM);
                function = (data, indices) -> {
                    part.partitionKSBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(K1SBM)) {
                final Partition part = createPartition(name, K1SBM);
                function = (data, indices) -> {
                    part.partitionK1SBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            // Paired key implementations.
            // This can be used to show they have no disadvantage for processing single keys.
            } else if (name.startsWith(PSBM)) {
                final Partition part = createPartition(name, PSBM);
                function = (data, indices) -> {
                    part.partitionPairedSBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            // Introselect implementations
            } else if (name.startsWith(ISBM)) {
                final Partition part = createPartition(name, ISBM);
                function = (data, indices) -> {
                    part.partitionISBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IDP)) {
                final Partition part = createPartition(name, IDP);
                function = (data, indices) -> {
                    part.partitionIDP(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(SELECT)) {
                // Not configurable
                function = (data, indices) -> {
                    Partition.select(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            }
            if (function == null) {
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
     * Source of a edge selector function. This is a function that selects indices
     * that are clustered close to the edge of the data.
     *
     * <p>This is a specialised class to allow benchmarking the switch from using
     * quickselect partitioning to using heapselect.
     */
    @State(Scope.Benchmark)
    public static class EdgeFunctionSource {
        /** Name of the source.
         * For introselect methods this should effectively turn-off heapselect. */
        @Param({"HeapSelect", ISBM + "_HS20", IDP + "_HS20"})
        private String name;

        /** The action. */
        private BiFunction<double[], IndexInterval, double[]> function;

        /**
         * @return the function
         */
        public BiFunction<double[], IndexInterval, double[]> getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            // Direct use of heapselect
            if ("HeapSelect".equals(name)) {
                function = (data, indices) -> {
                    Partition.heapSelectRange(data, 0, data.length - 1, indices.left(), indices.right());
                    return extractIndices(data, indices);
                };
            // introselect methods - these should be configured to not use heapselect
            } else if (name.startsWith(ISBM)) {
                final Partition part = createPartition(name, ISBM);
                function = (data, indices) -> {
                    part.introselect(Partition::partitionSBM, data,
                        0, data.length - 1, indices, indices.left(), indices.right(), 10000);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IDP)) {
                final Partition part = createPartition(name, IDP);
                function = (data, indices) -> {
                    part.introselect(Partition::partitionDP, data,
                        0, data.length - 1, indices, indices.left(), indices.right(), 10000);
                    return extractIndices(data, indices);
                };
            } else {
                throw new IllegalStateException("Unknown edge selector function: " + name);
            }
        }

        /**
         * Extract the data at the specified indices.
         *
         * @param data Data.
         * @param indices Indices.
         * @return the data
         */
        private static double[] extractIndices(double[] data, IndexInterval indices) {
            final int l = indices.left();
            final int r = indices.right();
            final double[] x = new double[r - l + 1];
            for (int i = l; i <= r; i++) {
                x[i - l] = data[i];
            }
            return x;
        }
    }

    /**
     * Creates the {@link Quantile}.
     * Parameters for the {@link KthSelector} are derived from the {@code name}.
     * The instance is configured to overwrite (process in-place) the input partition data.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link Quantile} instance
     */
    private static Quantile withKthSelector(String name, String prefix) {
        // For parity with the CM implementation use HF6
        return Quantile.withDefaults()
            .with(EstimationMethod.HF6)
            .withOverwrite(true)
            .withKthSelector(createKthSelector(name, prefix));
    }

    /**
     * Creates the {@link Quantile}.
     * Parameters for the {@link Partition} are derived from the {@code name}.
     * The instance is configured to overwrite (process in-place) the input partition data.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link Quantile} instance
     */
    private static Quantile withPartition(String name, String prefix) {
        // For parity with the CM implementation use HF6
        return Quantile.withDefaults()
            .with(EstimationMethod.HF6)
            .withOverwrite(true)
            .withPartition(createPartition(name, prefix));
    }

    /**
     * Creates the {@link KthSelector}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link KthSelector} instance
     */
    static KthSelector createKthSelector(String name, String prefix) {
        final String[] s = {name};
        final int minQuickSelectSize = getMinQuickSelectSize(s);
        final PivotingStrategy sp = getPivotStrategy(s);
        // Check for unharvested parameters
        for (int i = prefix.length(); i < s[0].length(); i++) {
            if (s[0].charAt(i) != '_') {
                throw new IllegalStateException(
                    String.format("Unharvested KthSelector parameters: %s -> %s", name, s[0]));
            }
        }
        return new KthSelector(sp, minQuickSelectSize);
    }

    /**
     * Creates the {@link Partition}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link Partition} instance
     */
    static Partition createPartition(String name, String prefix) {
        final String[] s = {name};
        final PivotingStrategy sp = getPivotStrategy(s);
        final DualPivotingStrategy dp = getDualPivotStrategy(s);
        final int minQuickSelectSize = getMinQuickSelectSize(s);
        final int heapSelectShift = getHeapSelectShift(s);
        final int heapSelectConstant = getHeapSelectConstant(s);
        final int heapSelectMaskShift = getHeapSelectMaskShift(s);
        final KeyStrategy keyStartegy = getKeyStrategy(s);
        final double recursionMultiple = getRecursionMultiple(s);
        final int recursionConstant = getRecursionConstant(s);
        final int compressionLevel = getCompressionLevel(s);
        // Check for unharvested parameters
        for (int i = prefix.length(); i < s[0].length(); i++) {
            if (s[0].charAt(i) != '_') {
                throw new IllegalStateException(
                    String.format("Unharvested Partition parameters: %s -> %s", name, s[0]));
            }
        }
        final Partition p = new Partition(sp, dp, minQuickSelectSize,
            heapSelectShift, heapSelectConstant, heapSelectMaskShift);
        // Some values do not have to be final as they are not used within optimised
        // partitioning code.
        p.setKeyStrategy(keyStartegy);
        p.setRecursionMultiple(recursionMultiple);
        p.setRecursionConstant(recursionConstant);
        p.setCompression(compressionLevel);
        return p;
    }

    /**
     * Gets the minimum size for the recursive quickselect partition algorithm.
     * Below this size the algorithm will change strategy for partitioning,
     * e.g. change to a full sort.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the minimum quickselect size
     */
    static int getMinQuickSelectSize(String[] name) {
        final Matcher m = QS_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.MIN_QUICKSELECT_SIZE;
    }

    /**
     * Gets the length shift for the heapselect distance-from-end computation.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the heapselect shift
     */
    static int getHeapSelectShift(String[] name) {
        final Matcher m = HS_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.HEAPSELECT_SHIFT;
    }

    /**
     * Gets the constant for the heapselect distance-from-end computation.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the heapselect constant
     */
    static int getHeapSelectConstant(String[] name) {
        final Matcher m = HC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.HEAPSELECT_CONSTANT;
    }

    /**
     * Gets the length shift for the mask applied to the dynamic heapselect
     * distance-from-end computation.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the heapselect mask shift
     */
    static int getHeapSelectMaskShift(String[] name) {
        final Matcher m = MS_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.HEAPSELECT_MASK_SHIFT;
    }

    /**
     * Gets the recursion multiplication factor.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the recursion multiple
     */
    static double getRecursionMultiple(String[] name) {
        final Matcher m = RM_PATTERN.matcher(name[0]);
        if (m.find()) {
            final double d = Double.parseDouble(m.group(1));
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return d;
        }
        return Partition.RECURSION_MULTIPLE;
    }

    /**
     * Gets the recursion constant.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the recursion constant
     */
    static int getRecursionConstant(String[] name) {
        final Matcher m = RC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.RECURSION_CONSTANT;
    }

    /**
     * Gets the compression level for {@link CompressedIndexSet}.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the compression
     */
    static int getCompressionLevel(String[] name) {
        final Matcher m = CL_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.COMPRESSION;
    }

    /**
     * Gets the pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the pivot strategy
     */
    static PivotingStrategy getPivotStrategy(String[] name) {
        return getPivotStrategyOrElse(name, Partition.PIVOTING_STRATEGY);
    }

    /**
     * Gets the pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the pivot strategy
     */
    static PivotingStrategy getPivotStrategyOrElse(String[] name, PivotingStrategy defaultValue) {
        // Names can have partial matches. Match the longest name
        int len = 0;
        PivotingStrategy result = defaultValue;
        for (final PivotingStrategy s : PivotingStrategy.values()) {
            if (name[0].contains(s.name()) && s.name().length() > len) {
                result = s;
                len = s.name().length();
            }
        }
        name[0] = name[0].replace(result.toString(), "");
        return result;
    }

    /**
     * Gets the dual pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the dual pivot strategy
     */
    static DualPivotingStrategy getDualPivotStrategy(String[] name) {
        return getDualPivotStrategyOrElse(name, Partition.DUAL_PIVOTING_STRATEGY);
    }

    /**
     * Gets the dual pivot strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the dual pivot strategy
     */
    static DualPivotingStrategy getDualPivotStrategyOrElse(String[] name, DualPivotingStrategy defaultValue) {
        // Names can have partial matches. Match the longest name
        int len = 0;
        DualPivotingStrategy result = defaultValue;
        for (final DualPivotingStrategy s : DualPivotingStrategy.values()) {
            if (name[0].contains(s.name()) && s.name().length() > len) {
                result = s;
                len = s.name().length();
            }
        }
        name[0] = name[0].replace(result.toString(), "");
        return result;
    }

    /**
     * Gets the sequential strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the sequential strategy
     */
    static KeyStrategy getKeyStrategy(String[] name) {
        return getKeyStrategyOrElse(name, Partition.KEY_STRATEGY);
    }

    /**
     * Gets the sequential strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the sequential strategy
     */
    static KeyStrategy getKeyStrategyOrElse(String[] name, KeyStrategy defaultValue) {
        int len = 0;
        KeyStrategy result = defaultValue;
        for (final KeyStrategy s : KeyStrategy.values()) {
            if (name[0].contains(s.name()) && s.name().length() > len) {
                result = s;
                len = s.name().length();
            }
        }
        name[0] = name[0].replace(result.toString(), "");
        return result;
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
        Arrays.sort(values);
        for (int i = 0; i < p.length; i++) {
            // EstimationMethod.HF6 (as per the Apache Commons Math Percentile
            // legacy implementation)
            final double pos = p[i] * (n + 1);
            final double fpos = Math.floor(pos);
            final int j = (int) fpos;
            final double g = pos - fpos;
            if (j < 1) {
                q[i] = values[0];
            } else if (j >= n) {
                q[i] = values[n - 1];
            } else {
                q[i] = DoubleMath.interpolate(values[j - 1], values[j], g);
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
        final int size = source.size();
        final double[] p = quantiles.getData();
        final BinaryOperator<double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            bh.consume(fun.apply(source.getData(j), p));
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
        final int size = source.size();
        final double[] p = quantiles.getData();
        final BinaryOperator<double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            bh.consume(fun.apply(source.getData(j), p));
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
        final int size = source.size();
        final Consumer<double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            final double[] y = source.getData(j);
            fun.accept(y);
            bh.consume(y);
        }
    }

    /**
     * Benchmark a sort of 5 data values.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void fiveSort(Sort5FunctionSource function, SortSource source, Blackhole bh) {
        final int size = source.size();
        final Consumer<double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            final double[] y = source.getData(j);
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
        final int size = source.size();
        final int[][] indices = source.getIndices();
        final BiFunction<double[], int[], double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            for (final int[] i : indices) {
                // Note: This uses the indices without cloning. This is because some
                // functions do not destructively modify the data.
                bh.consume(fun.apply(source.getData(j), i));
            }
        }
    }

    /**
     * Benchmark partitioning of an interval of indices a set distance from the edge.
     * This is used to benchmark the switch from quickselect partitioning to heapselect.
     *
     * @param function Source of the function.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void edgeSelect(EdgeFunctionSource function, EdgeSource source, Blackhole bh) {
        final int size = source.size();
        final IndexInterval[] indices = source.getIndices();
        final BiFunction<double[], IndexInterval, double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            for (final IndexInterval i : indices) {
                bh.consume(fun.apply(source.getData(j), i));
            }
        }
    }

    /**
     * Benchmark the tracking of an interval of indices during a partition algorithm.
     *
     * <p>The {@link IndexInterval} is created for the indices of interest. These are then
     * cut at all points in the interval between indices to simulate a partition algorithm
     * dividing the data and requiring a new interval to use in each part:
     * <pre>{@code
     *            cut
     *             |
     * -------k--------k---------k------k---------k--------
     *          <-- scan previous
     *    scan next -->
     * }</pre>
     *
     * @param function Source of the interval.
     * @param source Source of the data.
     * @return value to consume
     */
    @Benchmark
    public long indexInterval(IndexIntervalSource function, IndexSource source) {
        final int[][] indices = source.getIndices();
        final int[][] points = source.getPoints();
        // Ensure we have something to consume during the benchmark
        long sum = 0;
        for (int i = 0; i < indices.length; i++) {
            final int[] x = indices[i];
            final int[] p = points[i];
            final IndexInterval interval = function.create(x);
            for (final int k : p) {
                sum += interval.nextIndex(k);
                sum += interval.previousIndex(k);
            }
        }
        return sum;
    }
}
