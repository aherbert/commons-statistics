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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Logger;
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
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.PairedKeyStrategy;
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
    /** Single-pivot partitioning using a heap.
     * This method is copied from Commons Math. */
    private static final String SPH = "SPH";
    /** Single-pivot partitioning.
     * This method is adapted from Commons Math.
     * Evaluation couples the double[] data type to the EstimationMethod class. */
    private static final String SPE = "SPE";
    /** Single-pivot partitioning. */
    private static final String SP = "SP";
    /** Bentley-McIlroy partitioning (Sedgewick). */
    private static final String SBM = "SBM";
    /** Bentley-McIlroy partitioning (original). */
    private static final String BM = "BM";
    /** Dual-pivot partitioning. */
    private static final String DP = "DP";
    /** Floyd-Rivest partitioning. */
    private static final String FR = "FR";
    /** Floyd-Rivest partitioning (Kiwiel). */
    private static final String KFR = "KFR";
    /** Dual-pivot partitioning with 5 sorted points to choose pivots. */
    private static final String DP5 = "5DP";
    /** Commons Math Percentile implementation. */
    private static final String CM = "CM";
    /** Dutch National Flag partitioning. */
    private static final String DNF = "DNF";
    /** Use the JDK sort function. */
    private static final String JDK = "JDK";
    /** Use a sort function. */
    private static final String SORT = "Sort";
    /** Baseline for the benchmark. */
    private static final String BASELINE = "Baseline";
    /** Selection method using a heap. */
    private static final String HEAP_SELECT = "HeapSelect";
    /** Selection method using a sort. */
    private static final String SORT_SELECT = "SortSelect";

    // Second generation partition functions

    /** Bentley-McIlroy partitioning (Sedgewick). */
    private static final String SBM2 = "2SBM";

    // Introselect functions - switch to heapselect when progress is poor

    /** Introselect implementation with single pivot partitioning. */
    private static final String ISP = "ISP";
    /** Introselect implementation with Bentley-McIlroy partitioning (original). */
    private static final String IBM = "IBM";
    /** Introselect implementation with Bentley-McIlroy partitioning (Sedgewick). */
    private static final String ISBM = "ISBM";
    /** Introselect implementation with Bentley-McIlroy partitioning (Kiwiel). */
    private static final String IKBM = "IKBM";
    /** Introselect implementation with Dutch National Flag partitioning. */
    private static final String IDNF = "IDNF";
    /** Introselect implementation with dual-pivot partitioning. */
    private static final String IDP = "IDP";
    /** Commons Statistics Quantile implementation. This method is built using the best performing
     * select function across a range of input data. This algorithm currently cannot be configured. */
    private static final String SELECT = "SELECT";

    /** Pattern for the minimum quickselect size. */
    private static final Pattern QS_PATTERN = Pattern.compile("QS(\\d+)");
    /** Pattern for the heapselect shift. */
    private static final Pattern HS_PATTERN = Pattern.compile("HS(\\d+)");
    /** Pattern for the heapselect constant. */
    private static final Pattern HC_PATTERN = Pattern.compile("HC(\\d+)");
    /** Pattern for the heapselect mask shift. */
    private static final Pattern MS_PATTERN = Pattern.compile("MS(\\d+)");
    /** Pattern for the sortselect constant. */
    private static final Pattern SC_PATTERN = Pattern.compile("SC(\\d+)");
    /** Pattern for the sub-sampling size. */
    private static final Pattern SU_PATTERN = Pattern.compile("SU(\\d+)");
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
     * <ol>
     * <li>Creating an instance of the abstract class that provides the data length
     * <li>Calling {@link #setup()} to create the data
     * <li>Iterating over the data
     * </ol>
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
        /** Flag to determine if the data size should be logged. This is useful to be
         * able to determine the execution time per sample when the number of samples
         * is dynamically created based on the data length, range and seed. */
        private static final AtomicBoolean LOG_SIZE = new AtomicBoolean();

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
            SHUFFLE,
            /** sharktooth distribution. This is an addition to the original suite of B & M
             * and is not included in the test suite by default and must be specified.
             *
             * <p>An ascending then descending sequence is also known as organpipe in
             * Valois (2000),
             * Introspective sorting and selection revisited,
             * Software–Practice and Experience 30, 617–638.
             * This version allows multiple ascending/descending runs in the same length. */
            SHARKTOOTH;
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
            /** descending modification (this is an addition to the original suite of B & M).
             * It is useful for testing worst case performance, e.g. insertion sort performs
             * poorly on descending data. Heapselect using a max heap would perform poorly
             * if data is processed in the forward direction as all elements must be inserted.
             *
             * <p>This is not included in the test suite by default and must be specified.
             * Note that the Shuffle distribution with a very large seed 'm' is effectively an
             * ascending sequence and will be reversed to descending as part of the original
             * suite of data. */
            DESCENDING,
            /** dither modification. */
            DITHER;
        }

        /** Order. This is randomized to ensure that successive calls do not partition
         * similar distributions. Randomized per invocation to avoid the JVM 'learning'
         * branch decisions to take in small data sets. */
        protected int[] order;
        /** Cached source of randomness. */
        protected UniformRandomProvider rng;

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

        /** Sample offset. This is used to shift each distribution to create different data.
         * It is advanced on each invocation of {@link #setup()}. */
        @Param({"0"})
        private int offset;

        /** Number of samples. Applies only to the random distribution. In this case
         * the length of the data is randomly chosen in {@code [length, length + range)}. */
        @Param({"0"})
        private int samples;

        /** RNG seed. Created using ThreadLocalRandom.current().nextLong(). This is advanced
         * for the random distribution mode per iteration. Each benchmark executed by
         * JMH will use the same random data, even across JVMs.
         *
         * <p>If this is zero then a random seed is chosen. */
        @Param({"-7450238124206088695"})
        private long rngSeed = -7450238124206088695L;

        /** Data. This is stored as integer data which saves memory. Note that when ranking
         * data it is not necessary to have the full range of the double data type; the same
         * number of unique values can be recorded in an array using an integer type.
         * Returning a double[] forces a copy to be generated for destructive sorting /
         * partitioning methods. */
        private int[][] data;

        /**
         * Gets the sample for the given {@code index}.
         *
         * <p>This is returned in a randomized order per iteration.
         *
         * @param index Index.
         * @return the data sample
         */
        public double[] getData(int index) {
            return getDataSample(order[index]);
        }

        /**
         * Gets the sample for the given {@code index}.
         *
         * @param index Index.
         * @return the data sample
         */
        protected double[] getDataSample(int index) {
            final int[] a = data[index];
            final double[] x = new double[a.length];
            for (int i = -1; ++i < a.length;) {
                x[i] = a[i];
            }
            return x;
        }

        /**
         * Gets the sample size for the given {@code index}.
         *
         * @param index Index.
         * @return the data sample size
         */
        public int getDataSize(int index) {
            return data[index].length;
        }

        /**
         * Get the number of data samples.
         *
         * <p>Note: This data source will create a permutation order per invocation based on
         * this size. Per-invocation control in JMH is recommended for methods that take
         * more than 1 millisecond to execute. For very small data and/or fast methods
         * this may not be achievable. Child classes may override this value to create
         * a large number of repeats of the same data per invocation. Any class performing
         * this should also override {@link #getData(int)} to prevent index out of bound errors.
         * This can be done by mapping the index to the original index using the number of repeats
         * e.g. {@code original index = index / repeats}.
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
            // Note: Bentley-McIlroy use n in {100, 1023, 1024, 1025}.
            // Here we only support a continuous range. The range is important
            // for the median as it will require one or two points to partition
            // if the length is odd or even.
            final int r = range > 0 ? range : 0;
            if (length + (long) r > Integer.MAX_VALUE) {
                throw new IllegalStateException("Unsupported upper length: " + length);
            }
            final int length2 = length + r;

            // Allow pseudorandom seeding
            if (rngSeed == 0) {
                rngSeed = RandomSource.createLong();
            }
            if (rng == null) {
                // First call, create objects
                rng = RANDOM_SOURCE.create(rngSeed);
            }

            // Special case for random distribution mode
            if (dist.contains(Distribution.RANDOM) && dist.size() == 1 && samples > 0) {
                data = new int[samples][];
                final int upper = seed > 0 ? seed : Integer.MAX_VALUE;
                final SharedStateDiscreteSampler s1 = DiscreteUniformSampler.of(rng, 0, upper);
                final SharedStateDiscreteSampler s2 = DiscreteUniformSampler.of(rng, length, length2);
                for (int i = 0; i < data.length; i++) {
                    final int[] a = new int[s2.sample()];
                    for (int j = a.length; --j >= 0;) {
                        a[j] = s1.sample();
                    }
                    data[i] = a;
                }
                return;
            }

            // New data per iteration
            data = null;
            final int o = offset;
            offset = rng.nextInt();

            final EnumSet<Modification> mod = getModifications();

            // Data using the RNG will be randomized only once.
            // Here we use the same seed for parity across methods.
            // Note that most distributions do not use the source of randomness.
            final ArrayList<int[]> sampleData = new ArrayList<>();
            for (int n = length; n <= length2; n++) {
                // Note: Large lengths may wish to limit the range of m to limit
                // the memory required to store the samples. Currently a single
                // m is supported via the seed parameter.
                // Default seed will create ceil(log2(2*n)) * 5 dist * 6 mods samples:
                // MAX  = 32 * 5 * 7 * (2^31-1) * 4 bytes == 7679 GiB
                // HUGE = 31 * 5 * 7 * 2^30 * 4 bytes == 3719 GiB
                // BIG  = 21 * 5 * 7 * 2^20 * 4 bytes == 2519 MiB  <-- within configured JVM -Xmx
                // MED  = 11 * 5 * 7 * 2^10 * 4 bytes == 1318 KiB
                // (This excludes the descending modification.)
                // It is possible to create lengths above 2^30 using a single distribution,
                // modification, and seed:
                // MAX1 = 1 * 1 * 1 * (2^31-1) * 4 bytes == 8191 MiB
                // However this is then used to create double[] data thus requiring an extra
                // ~16GiB memory for the sample to partition.
                for (final int m : createSeeds(seed, n)) {
                    for (final int[] x : createDistributions(dist, rng, n, m, o)) {
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
                        // Only sort once
                        if (mod.contains(Modification.SORT) ||
                            mod.contains(Modification.DESCENDING)) {
                            final int[] y = x.clone();
                            Arrays.sort(y);
                            if (mod.contains(Modification.DESCENDING)) {
                                sampleData.add(reverse(y, 0, n));
                            }
                            if (mod.contains(Modification.SORT)) {
                                sampleData.add(y);
                            }
                        }
                        if (mod.contains(Modification.DITHER)) {
                            sampleData.add(dither(x));
                        }
                    }
                }
            }
            data = sampleData.toArray(int[][]::new);
            if (LOG_SIZE.compareAndSet(false, true)) {
                Logger.getLogger(getClass().getName()).info(
                    () -> String.format("Data length: [%d, %d] n=%d", length, length2, data.length));
            }
        }

        /**
         * Create the order to process the indices.
         *
         * <p>JMH recommends that invocations should take at
         * least 1 millisecond for timings to be usable. In practice there should be
         * enough data that processing takes much longer than a few milliseconds.
         */
        @Setup(Level.Invocation)
        public void createOrder() {
            if (order == null) {
                // First call, create objects
                order = PermutationSampler.natural(size());
            }
            PermutationSampler.shuffle(rng, order);
        }

        /**
         * @return the distributions
         */
        private EnumSet<Distribution> getDistributions() {
            final EnumSet<Distribution> mod = getEnumFromParam(Distribution.class, distribution);
            // Require the sharktooth distribution to be explicitly requested.
            if (ALL.equals(distribution)) {
                mod.remove(Distribution.SHARKTOOTH);
            }
            return mod;
        }

        /**
         * @return the modifications
         */
        private EnumSet<Modification> getModifications() {
            final EnumSet<Modification> mod = getEnumFromParam(Modification.class, modification);
            // Require the descending modification to be explicitly requested.
            if (ALL.equals(modification)) {
                mod.remove(Modification.DESCENDING);
            }
            return mod;
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
         * <p>The offset is used to adjust each distribution to generate a different output.
         * Only applies to distributions that do not use the source of randomness.
         *
         * <p>Distributions that are a constant value at {@code m == 1} are not generated.
         * This case is handled by the plateau distribution which will be a constant value
         * except one occurrence of zero.
         *
         * @param dist Distributions.
         * @param rng Source of randomness.
         * @param n Length of the sample.
         * @param m Sample seed (in [1, 2^31])
         * @param o Offset.
         * @return the samples
         */
        private static List<int[]> createDistributions(EnumSet<Distribution> dist,
                UniformRandomProvider rng, int n, int m, int o) {
            final ArrayList<int[]> distData = new ArrayList<>(5);
            int[] x;
            if (dist.contains(Distribution.SAWTOOTH) && m != 1) {
                distData.add(x = new int[n]);
                // i % m
                // Typical case m is a power of 2 so we can use a mask
                // Use the offset.
                final int mask = m - 1;
                if ((m & mask) == 0) {
                    for (int i = -1; ++i < n;) {
                        x[i] = (i + o) & mask;
                    }
                } else {
                    // User input seed. Start at the offset.
                    int j = o & Integer.MAX_VALUE;
                    for (int i = -1; ++i < n;) {
                        j = j % m;
                        x[i] = j++;
                    }
                }
            }
            if (dist.contains(Distribution.RANDOM) && m != 1) {
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
                    final int j = i + o;
                    x[i] = (int) (Integer.toUnsignedLong(j * m + j) % nn);
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
                // Rotate
                final int n1 = (o & Integer.MAX_VALUE) % n;
                if (n1 != 0) {
                    final int[] a = x.clone();
                    final int n2 = n - n1;
                    System.arraycopy(a, 0, x, n1, n2);
                    System.arraycopy(a, n2, x, 0, n1);
                }
            }
            if (dist.contains(Distribution.SHUFFLE) && m != 1) {
                distData.add(x = new int[n]);
                // rand() % m ? (j += 2) : (k += 2)
                final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, m - 1);
                for (int i = -1, j = 0, k = 1; ++i < n;) {
                    x[i] = s.sample() != 0 ? (j += 2) : (k += 2);
                }
            }
            if (dist.contains(Distribution.SHARKTOOTH) && m != 1) {
                distData.add(x = new int[n]);
                // ascending-descending runs
                int i = -1;
                int j = (o & Integer.MAX_VALUE) % m - 1;
                OUTER:
                for (;;) {
                    while (++j < m) {
                        if (++i == n) {
                            break OUTER;
                        }
                        x[i] = j;
                    }
                    while (--j >= 0) {
                        if (++i == n) {
                            break OUTER;
                        }
                        x[i] = j;
                    }
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
         * Gets the range.
         *
         * @return the range
         */
        final int getRange() {
            return range;
        }

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
         * Sets the sample 'offset' used to generate distributions. Advanced to a new
         * random integer on each invocation of {@link #setup()}.
         *
         * @param v Value.
         */
        void setOffset(int v) {
            offset = v;
        }

        /**
         * Sets the number of samples to use for the random distribution mode.
         * See {@link AbstractDataSource} for details.
         *
         * @param v Value.
         */
        void setSamples(int v) {
            samples = v;
        }

        /**
         * Sets the seed for the random number gnerator.
         *
         * @param v Value.
         */
        void setRngSeed(long v) {
            this.rngSeed = v;
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
    public static class QuantileFunctionSource {
        /** Name of the source. */
        @Param({
            // Slow
            //SORT, SPH, SPE
            CM, SP, BM, SBM, DP, DP5,
            SBM2,
            ISP, IBM, ISBM, IKBM, IDP, SELECT})
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
            // Introselect implementations
            } else if (name.startsWith(ISP)) {
                function = withPartition(name, ISP)::evaluateISP;
            } else if (name.startsWith(IBM)) {
                function = withPartition(name, IBM)::evaluateIBM;
            } else if (name.startsWith(ISBM)) {
                function = withPartition(name, ISBM)::evaluateISBM;
            } else if (name.startsWith(IKBM)) {
                function = withPartition(name, IKBM)::evaluateIKBM;
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
        @Param({"1023"})
        private int length;
        /** Number of repeats. This is used to control the number of times the data is processed
         * per invocation. Note that each invocation randomises the order. For very small data
         * and/or fast methods there may not be enough data to achieve the target of 1
         * millisecond per invocation. Use this value to increase the length of each invocation.
         * For example the insertion sort on tiny data, or the sort5 methods, may require this
         * to be 1,000,000 or higher. */
        @Param({"1"})
        private int repeats;

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }

        /** {@inheritDoc} */
        @Override
        public int size() {
            return super.size() * repeats;
        }

        /** {@inheritDoc} */
        @Override
        public double[] getData(int index) {
            // order = (data index) * repeats + repeat
            // data index = order / repeats
            return super.getDataSample(order[index] / repeats);
        }
    }

    /**
     * Source of k-th indices to partition.
     *
     * <p>This class provides both data to partition and the indices to partition.
     * The indices and data are created per iteration. The order to process them
     * is created per invocation.
     */
    @State(Scope.Benchmark)
    public static class KSource extends AbstractDataSource {
        /** Data length. */
        @Param({"1023"})
        private int length;
        /** Number of indices to select. */
        @Param({"1", "2", "3", "5", "10"})
        private int k;
        /** Number of repeats. */
        @Param({"10"})
        private int repeats;
        /** Distribution mode. K indices can be distributed randomly or uniformly.
         * Index mode uses k as the target index. */
        @Param({"random"})
        private String mode;
        /** Separation. K can be single indices (s=0) or paired (s!=0). Paired indices are
         * separated using the specified separation. When running in paired mode the
         * number of k is doubled and duplicates may occur. This method is used for
         * testing sparse or uniform distributions of paired indices that may occur when
         * interpolating quantiles. Since the separation is allowed to be above 1 it also
         * allows testing configurations for close indices. */
        @Param({"0"})
        private int s;

        /** Indices. */
        private int[][] indices;
        /** Cache permutation samplers. */
        private PermutationSampler[] samplers;

        /** {@inheritDoc} */
        @Override
        protected int getLength() {
            return length;
        }

        /** {@inheritDoc} */
        @Override
        public int size() {
            return super.size() * repeats;
        }

        /** {@inheritDoc} */
        @Override
        public double[] getData(int index) {
            // order = (data index) * repeats + repeat
            // data index = order / repeats
            return super.getDataSample(order[index] / repeats);
        }

        /**
         * Gets the indices for the given {@code index}.
         *
         * @param index Index.
         * @return the data indices
         */
        public int[] getIndices(int index) {
            // order = (data index) * repeats + repeat
            // Directly look-up the indices for this repeat.
            return indices[order[index]];
        }

        /**
         * Create the indices.
         */
        @Override
        @Setup(Level.Iteration)
        public void setup() {
            if (s < 0 || s >= getLength()) {
                throw new IllegalStateException("Invalid separation: " + s);
            }
            super.setup();
            // Data will be randomized per iteration
            if (indices == null) {
                // First call, create objects
                indices = new int[size()][];
                // Cache samplers. These hold an array which is randomized
                // per call to obtain a permutation.
                if (k > 1) {
                    samplers = new PermutationSampler[getRange() + 1];
                }
            }

            // Create indices in the data sample length.
            // If a separation is provided then the length is reduced by the separation
            // to make space for a second index.

            int index = 0;
            final int noOfSamples = super.size();
            if ("random".equals(mode)) {
                // random mode creates a permutation of k indices in the length
                if (k > 1) {
                    final int baseLength = getLength();
                    for (int i = 0; i < noOfSamples; i++) {
                        final int len = getDataSize(i);
                        // Create permutation sampler for the length
                        PermutationSampler sampler = samplers[len - baseLength];
                        if (sampler == null) {
                            // Reduce length by the separation
                            final int n = len - s;
                            samplers[len - baseLength] = sampler = new PermutationSampler(rng, n, k);
                        }
                        for (int j = repeats; --j >= 0;) {
                            indices[index++] = sampler.sample();
                        }
                    }
                } else {
                    // k=1: No requirement for a permutation
                    for (int i = 0; i < noOfSamples; i++) {
                        // Reduce length by the separation
                        final int n = getDataSize(i) - s;
                        for (int j = repeats; --j >= 0;) {
                            indices[index++] = new int[] {rng.nextInt(n)};
                        }
                    }
                }
            } else if ("uniform".equals(mode)) {
                // uniform indices with a random start
                for (int i = 0; i < noOfSamples; i++) {
                    // Reduce length by the separation
                    final int n = getDataSize(i) - s;
                    final int step = Math.max(1, (int) Math.round((double) n / k));
                    for (int j = repeats; --j >= 0;) {
                        final int[] k1 = new int[k];
                        int p = rng.nextInt(n);
                        for (int m = 0; m < k; m++) {
                            p = (p + step) % n;
                            k1[m] = p;
                        }
                        indices[index++] = k1;
                    }
                }
            } else if ("index".equals(mode)) {
                // Same single or paired indices for all samples.
                // Check the index is valid.
                for (int i = 0; i < noOfSamples; i++) {
                    // Reduce length by the separation
                    final int n = getDataSize(i) - s;
                    if (k >= n) {
                        throw new IllegalStateException("Invalid k: " + k + " >= " + n);
                    }
                }
                final int[] kk = s > 0 ? new int[] {k, k + s} : new int[] {k};
                Arrays.fill(indices, kk);
                return;
            } else {
                throw new IllegalStateException("Unknown index mode: " + mode);
            }
            // Add paired indices
            if (s > 0) {
                for (int i = 0; i < indices.length; i++) {
                    final int[] k1 = indices[i];
                    final int[] k2 = new int[k1.length << 1];
                    for (int j = 0; j < k1.length; j++) {
                        k2[j << 1] = k1[j];
                        k2[(j << 1) + 1] = k1[j] + s;
                    }
                    indices[i] = k2;
                }
            }
        }
    }

    /**
     * Source of k-th indices.
     */
    @State(Scope.Benchmark)
    public static class IndexSource {
        /** Indices. */
        protected int[][] indices;
        /** Upper bound (exclusive) on the indices. */
        @Param({"1000", "1000000", "1000000000"})
        private int length;
        /** Number of indices to select. */
        @Param({"10", "20", "40", "80", "160"})
        private int k;
        /** Number of repeats. */
        @Param({"1000"})
        private int repeats;
        /** RNG seed. Created using ThreadLocalRandom.current().nextLong(). Each benchmark
         * executed by JMH will use the same random data, even across JVMs.
         *
         * <p>If this is zero then a random seed is chosen. */
        @Param({"-7450238124206088695"})
        private long rngSeed;
        /** Ordered keys. */
        @Param({"false"})
        private boolean ordered;
        /** Minimum separation between keys. */
        @Param({"32"})
        private int separation;

        /**
         * @return the indices
         */
        public int[][] getIndices() {
            return indices;
        }

        /**
         * Gets the minimum separation between keys. This is used by benchmarks
         * to ignore splitting/search keys below a threshold.
         *
         * @return the minimum separation
         */
        public int getMinSeparation() {
            return separation;
        }

        /**
         * Create the indices and search points.
         */
        @Setup(Level.Iteration)
        public void setup() {
            if (k < 2) {
                throw new IllegalStateException("Require multiple indices");
            }
            // Data will be randomized per iteration. It is the same sequence across
            // benchmarks and JVM instances and allows benchmarking across JVM platforms
            // with the same data.
            // Allow pseudorandom seeding
            if (rngSeed == 0) {
                rngSeed = RandomSource.createLong();
            }
            final UniformRandomProvider rng = RANDOM_SOURCE.create(rngSeed);
            // Advance the seed for the next iteration.
            rngSeed = rng.nextLong();

            final SharedStateDiscreteSampler s = DiscreteUniformSampler.of(rng, 0, length - 1);

            indices = new int[repeats][];

            for (int i = repeats; --i >= 0;) {
                // Indices with possible repeats
                final int[] x = new int[k];
                for (int j = k; --j >= 0;) {
                    x[j] = s.sample();
                }
                indices[i] = x;
                if (ordered) {
                    Sorting.sortIndices(x, x.length);
                }
            }
        }

        /**
         * @return the RNG seed
         */
        long getRngSeed() {
            return rngSeed;
        }
    }

    /**
     * Source of k-th indices to be searched/split.
     * Can be used to split the same indices multiple times, or split a set of indices
     * a single time.
     */
    @State(Scope.Benchmark)
    public static class SplitIndexSource extends IndexSource {
        /** Division mode. */
        @Param({"RANDOM", "BINARY"})
        private DivisionMode mode;

        /** Search points. */
        private int[][] points;
        /** The index+point samples. */
        private long[] samples;

        /** Options for the division mode. */
        public enum DivisionMode {
            /** Randomly divide. */
            RANDOM,
            /** Divide using binary division with recursion left then right. */
            BINARY;
        }

        /**
         * @return the search points
         */
        public int[][] getPoints() {
            return points;
        }

        /**
         * @return the sample size
         */
        int samples() {
            return samples.length;
        }

        /**
         * Gets the indices for the sample.
         *
         * @param index the index
         * @return the indices
         */
        int[] getIndices(int index) {
            return indices[(int) (samples[index] >>> Integer.SIZE)];
        }

        /**
         * Gets the search point for the sample.
         *
         * @param index the index
         * @return the search point
         */
        int getPoint(int index) {
            return (int) samples[index];
        }

        /**
         * Create the indices and search points.
         */
        @Override
        @Setup(Level.Iteration)
        public void setup() {
            super.setup();

            final UniformRandomProvider rng = RANDOM_SOURCE.create(getRngSeed());

            final int[][] indices = getIndices();
            points = new int[indices.length][];

            final int s = getMinSeparation();

            // Set the division mode
            final boolean random = Objects.requireNonNull(mode) == DivisionMode.RANDOM;

            int size = 0;

            for (int i = points.length; --i >= 0;) {
                // Get the sorted unique indices
                final int[] y = indices[i].clone();
                final int unique = Sorting.sortIndices(y, y.length);

                // Create the cut points between each unique index
                int[] p = new int[unique - 1];
                if (random) {
                    int c = 0;
                    for (int j = 0; j < p.length; j++) {
                        // Ignore dense keys
                        if (y[j] + s < y[j + 1]) {
                            p[c++] = (y[j] + y[j + 1]) >>> 1;
                        }
                    }
                    p = Arrays.copyOf(p, c);
                    shuffle(rng, p);
                    points[i] = p;
                } else {
                    // binary division
                    final int c = divide(y, 0, unique - 1, p, 0, s);
                    points[i] = Arrays.copyOf(p, c);
                }
                size += points[i].length;
            }

            // Create the samples: pack indices index+point into a long
            samples = new long[size];
            for (int i = points.length; --i >= 0;) {
                final long l = ((long) i) << Integer.SIZE;
                for (final int p : points[i]) {
                    samples[--size] = l | p;
                }
            }
            shuffle(rng, samples);
        }

        /**
         * Divide the indices using binary division with recursion left then right.
         * If a division is possible store the division point and update the count.
         *
         * @param indices Indices to divide
         * @param lo Lower index in indices (inclusive).
         * @param hi Upper index in indices (inclusive).
         * @param p Division points.
         * @param c Count of division points.
         * @param s Minimum separation between indices.
         * @return the updated count of division points.
         */
        private static int divide(int[] indices, int lo, int hi, int[] p, int c, int s) {
            if (lo < hi) {
                // Divide the interval in half
                final int m = (lo + hi) >>> 1;
                // Create a division point at approximately the midpoint
                final int m1 = m + 1;
                // Ignore dense keys
                if (indices[m] + s < indices[m1]) {
                    final int k = (indices[m] + indices[m1]) >>> 1;
                    p[c++] = k;
                }
                // Recurse left then right.
                // Does nothing if lo + 1 == hi as m == lo and m1 == hi.
                c = divide(indices, lo, m, p, c, s);
                c = divide(indices, m1, hi, p, c, s);
            }
            return c;
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

        /**
         * Shuffles the entries of the given array.
         *
         * @param rng Source of randomness.
         * @param array Array whose entries will be shuffled (in-place).
         */
        private static void shuffle(UniformRandomProvider rng, long[] array) {
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
        private static void swap(long[] array, int i, int j) {
            final long tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }

    /**
     * Source of an {@link SearchableInterval}.
     */
    @State(Scope.Benchmark)
    public static class SearchableIntervalSource {
        /** Name of the source. */
        @Param({"ScanningKeyInterval",
            "BinarySearchKeyInterval",
            "IndexSetInterval",
            "CompressedIndexSet",
            // Same speed as the CompressedIndexSet
            //"CompressedIndexSet2",
            })
        private String name;

        /** The factory. */
        private Function<int[], SearchableInterval> factory;

        /**
         * @param indices Indices.
         * @return {@link SearchableInterval}
         */
        public SearchableInterval create(int[] indices) {
            return factory.apply(indices);
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            if ("ScanningKeyInterval".equals(name)) {
                factory = k -> {
                    k = k.clone();
                    final int unique = Sorting.sortIndices(k, k.length);
                    return ScanningKeyInterval.of(k, unique);
                };
            } else if ("BinarySearchKeyInterval".equals(name)) {
                factory = k -> {
                    k = k.clone();
                    final int unique = Sorting.sortIndices(k, k.length);
                    return BinarySearchKeyInterval.of(k, unique);
                };
            } else if ("IndexSetInterval".equals(name)) {
                factory = IndexSet::of;
            } else if (name.equals("CompressedIndexSet2")) {
                factory = CompressedIndexSet2::of;
            } else if (name.startsWith("CompressedIndexSet")) {
                // To use compression 2 requires CompressedIndexSet_2 otherwise
                // a fixed compression set will be returned
                final int c = getCompression(name);
                factory = k -> CompressedIndexSet.of(c, k);
            } else {
                throw new IllegalStateException("Unknown SearchableInterval: " + name);
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
     * Source of an {@link UpdatingInterval}.
     */
    @State(Scope.Benchmark)
    public static class UpdatingIntervalSource {
        /** Name of the source. */
        @Param({"KeyUpdatingInterval",
            // Same speed as BitIndexUpdatingInterval
            //"IndexSet",
            "BitIndexUpdatingInterval",
            })
        private String name;

        /** The factory. */
        private Function<int[], UpdatingInterval> factory;

        /**
         * @param indices Indices.
         * @return {@link UpdatingInterval}
         */
        public UpdatingInterval create(int[] indices) {
            return factory.apply(indices);
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            if ("KeyUpdatingInterval".equals(name)) {
                factory = k -> {
                    k = k.clone();
                    final int unique = Sorting.sortIndices(k, k.length);
                    return KeyUpdatingInterval.of(k, unique);
                };
            } else if ("IndexSet".equals(name)) {
                factory = k -> IndexSet.of(k).interval();
            } else if (name.equals("BitIndexUpdatingInterval")) {
                factory = k -> BitIndexUpdatingInterval.of(k, k.length);
            } else {
                throw new IllegalStateException("Unknown UpdatingInterval: " + name);
            }
        }
    }

    /**
     * Source of a range of positions to partition. These are positioned away from the edge
     * using a power of 2 shift.
     *
     * <p>This is a specialised class to allow benchmarking the switch from using
     * quickselect partitioning to using heapselect.
     *
     * <p>This class provides both data to partition and the indices to partition.
     * The indices and data are created per iteration. The order to process them
     * is created per invocation.
     */
    @State(Scope.Benchmark)
    public static class EdgeSource extends AbstractDataSource {
        /** Data length. */
        @Param({"1023"})
        private int length;
        /** Mode. */
        @Param({"SHIFT"})
        private Mode mode;
        /** Parameter to find k. Configured for 'shift' of the length. */
        @Param({"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        private int p;
        /** Target indices (as pairs of {@code [ka, kb]} defining a range to select). */
        private int[][] indices;

        /** Define the method used to generated the edge k. */
        public enum Mode {
            /** Create {@code k} using a right-shift {@code >>>} applied to the length. */
            SHIFT,
            /** Use the parameter {@code p} as an index. */
            INDEX;
        }

        /** {@inheritDoc} */
        @Override
        public int size() {
            return super.size() * 2;
        }

        /** {@inheritDoc} */
        @Override
        public double[] getData(int index) {
            // order = (data index) * repeats + repeat
            // data index = order / repeats; repeats=2 divide by using a shift
            return super.getDataSample(order[index] >> 1);
        }

        /**
         * Gets the sample indices for the given {@code index}.
         * Returns a range to partition {@code [k1, kn]}.
         *
         * @param index Index.
         * @return the target indices
         */
        public int[] getIndices(int index) {
            // order = (data index) * repeats + repeat
            // Directly look-up the indices for this repeat.
            return indices[order[index]];
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
        @Setup(Level.Iteration)
        public void setup() {
            // Data will be randomized per iteration
            super.setup();
            // Error for a bad configuration. Allow k=0 but not smaller.
            // Uses the lower bound on the length.
            int k;
            if (mode == Mode.SHIFT) {
                k = length >>> p;
                if (k == 0 && length >>> (p - 1) == 0) {
                    throw new IllegalStateException(length + " >>> (" + p + " - 1) == 0");
                }
            } else if (mode == Mode.INDEX) {
                k = p;
                if (k < 0 || k >= length) {
                    throw new IllegalStateException("Invalid index [0, " + length + "): " + p);
                }
            } else {
                throw new IllegalStateException("Unknown mode: " + mode);
            }

            if (indices == null) {
                // First call, create objects
                indices = new int[size()][];
            }

            // Create a single index at both ends.
            // Note: Data has variable length so we have to compute the upper end for each sample.
            // Re-use the constant lower but we do not bother to cache repeats of the upper.
            final int[] lower = {k, k};
            final int noOfSamples = super.size();
            for (int i = 0; i < noOfSamples; i++) {
                final int len = getDataSize(i);
                final int k1 = len - 1 - k;
                indices[i << 1] = lower;
                indices[(i << 1) + 1] = new int[] {k1, k1};
            }
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
            ISBM, IKBM, IDP,
            })
        private String name;

        /** Override of minimum quickselect size. */
        @Param({"0"})
        private int qs;

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
                function = createKthSelector(name, SP, qs)::sortSP;
            } else if (name.startsWith(SBM)) {
                function = createKthSelector(name, SBM, qs)::sortSBM;
            } else if (name.startsWith(BM)) {
                function = createKthSelector(name, BM, qs)::sortBM;
            } else if (name.startsWith(DP)) {
                function = createKthSelector(name, DP, qs)::sortDP;
            } else if (name.startsWith(DP5)) {
                function = createKthSelector(name, DP5, qs)::sortDP5;
            } else if (name.startsWith(DNF)) {
                function = createKthSelector(name, DNF, qs)::sortDNF;
            // 2nd generation partition functions
            } else if (name.startsWith(SBM2)) {
                function = createPartition(name, SBM2, qs, 0, 0)::sortSBM;
            // Introsort
            } else if (name.startsWith(ISP)) {
                function = createPartition(name, ISP, qs, 0, 0)::sortISP;
            } else if (name.startsWith(IBM)) {
                function = createPartition(name, IBM, qs, 0, 0)::sortIBM;
            } else if (name.startsWith(ISBM)) {
                function = createPartition(name, ISBM, qs, 0, 0)::sortISBM;
            } else if (name.startsWith(IKBM)) {
                function = createPartition(name, IKBM, qs, 0, 0)::sortIKBM;
            } else if (name.startsWith(IDNF)) {
                // 3 variants
                if (name.startsWith(IDNF + "3")) {
                    function = createPartition(name, IDNF + "3", qs, 0, 0)::sortIDNF3;
                } else if (name.startsWith(IDNF + "2")) {
                    function = createPartition(name, IDNF + "2", qs, 0, 0)::sortIDNF2;
                } else if (name.startsWith(IDNF + "1")) {
                    function = createPartition(name, IDNF + "1", qs, 0, 0)::sortIDNF1;
                }
            } else if (name.startsWith(IDP)) {
                function = createPartition(name, IDP, qs, 0, 0)::sortIDP;
            } else if (name.startsWith(SELECT)) {
                // Sort by selection of the entire range.
                final Supplier<DoubleDataTransformer> transformerFactory =
                    DoubleDataTransformers.createFactory(NaNPolicy.INCLUDE, false);
                function = a -> {
                    // Handle NaN / signed zeros
                    final DoubleDataTransformer t = transformerFactory.get();
                    // Assume this is in-place
                    t.preProcess(a);
                    final int end = t.length();
                    if (end <= 1) {
                        // Nothing to sort
                        return;
                    }
                    Partition.select(a, 0, end - 1,
                        IndexIntervals.interval(0, end - 1),
                        Partition.dualPivotFlags(0, end, 0, end - 1, 100));
                    // Restore signed zeros
                    t.postProcess(a);
                };
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
            } else if (name.startsWith("PairedInsertionSort")) {
                if (name.endsWith("1")) {
                    function = x -> {
                        x[0] = Double.NEGATIVE_INFINITY;
                        Sorting.sortPairedInternal1(x, 1, x.length - 1);
                    };
                } else if (name.endsWith("2")) {
                    function = x -> {
                        x[0] = Double.NEGATIVE_INFINITY;
                        Sorting.sortPairedInternal2(x, 1, x.length - 1);
                    };
                } else if (name.endsWith("3")) {
                    function = x -> {
                        x[0] = Double.NEGATIVE_INFINITY;
                        Sorting.sortPairedInternal3(x, 1, x.length - 1);
                    };
                } else if (name.endsWith("4")) {
                    function = x -> {
                        x[0] = Double.NEGATIVE_INFINITY;
                        Sorting.sortPairedInternal4(x, 1, x.length - 1);
                    };
                }
            } else if ("InsertionSortB".equals(name)) {
                function = x -> {
                    x[0] = Double.NEGATIVE_INFINITY;
                    Sorting.sortb(x, 1, x.length - 1);
                };
            // Not actually a sort. This is used to benchmark the speed of heapselect
            // against a full sort of small data.
            } else if (name.startsWith(HEAP_SELECT)) {
                final char c = name.charAt(name.length() - 1);
                // This offsets the start by 1 for comparison with insertion sort
                final int k = Character.isDigit(c) ? Character.digit(c, 10) + 1 : 1;
                function = x -> Partition.heapSelectLeft(x, 1, x.length - 1, k, 0);
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
        @Param({"sort5", "sort5a", "sort5b",
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
            } else if ("sort5a".equals(name)) {
                function = x -> {
                    final int s = x.length >> 2;
                    Sorting.sort5a(x, 0, s, s << 1, x.length - 1 - s, x.length - 1);
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
            SBM2,
            ISP, IBM, ISBM, IKBM, IDNF, IDP, SELECT})
        private String name;

        /** Override of minimum quickselect size. */
        @Param({"0"})
        private int qs;

        /** Override of minimum heapselect constant. */
        @Param({"0"})
        private int hc;

        /** Override of minimum sortselect constant. */
        @Param({"0"})
        private int sc;

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
                    final Partition part = createPartition(name.substring(SORT.length()), ISBM, qs, hc, sc);
                    function = (data, indices) -> {
                        part.sortISBM(data);
                        return extractIndices(data, indices);
                    };
                } else if (name.contains(IKBM)) {
                    final Partition part = createPartition(name.substring(SORT.length()), IKBM, qs, hc, sc);
                    function = (data, indices) -> {
                        part.sortIKBM(data);
                        return extractIndices(data, indices);
                    };
                } else if (name.contains(IDP)) {
                    final Partition part = createPartition(name.substring(SORT.length()), IDP, qs, hc, sc);
                    function = (data, indices) -> {
                        part.sortIDP(data);
                        return extractIndices(data, indices);
                    };
                } else if (name.contains(IDNF + "3")) {
                    // Only support IDNF3
                    final Partition part = createPartition(name.substring(SORT.length()), IDNF + "3", qs, hc, sc);
                    function = (data, indices) -> {
                        part.sortIDNF3(data);
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
                final KthSelector selector = createKthSelector(name, SPH, qs);
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
                final KthSelector selector = createKthSelector(name, SP, qs);
                function = (data, indices) -> {
                    selector.partitionSP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(BM)) {
                final KthSelector selector = createKthSelector(name, BM, qs);
                function = (data, indices) -> {
                    selector.partitionBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(SBM)) {
                final KthSelector selector = createKthSelector(name, SBM, qs);
                function = (data, indices) -> {
                    selector.partitionSBM(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP)) {
                final KthSelector selector = createKthSelector(name, DP, qs);
                function = (data, indices) -> {
                    selector.partitionDP(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DP5)) {
                final KthSelector selector = createKthSelector(name, DP5, qs);
                function = (data, indices) -> {
                    selector.partitionDP5(data, indices.clone());
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(DNF)) {
                final KthSelector selector = createKthSelector(name, DNF, qs);
                function = (data, indices) -> {
                    selector.partitionDNF(data, indices.clone());
                    return extractIndices(data, indices);
                };
            // Second generation partition functions
            } else if (name.startsWith(SBM2)) {
                final Partition part = createPartition(name, SBM2, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionSBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(FR)) {
                // Behaviour defined by control flags appended to FR
                String prefix = FR;
                int v = 0;
                if (name.length() > 2 && Character.isDigit(name.charAt(2))) {
                    prefix = name.substring(0, 3);
                    v = Character.digit(name.charAt(2), 10);
                }
                final Partition part = createPartition(name, prefix, qs, hc, sc).setControlFlags(v);
                function = (data, indices) -> {
                    part.partitionFR(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(KFR)) {
                final Partition part = createPartition(name, KFR, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionKFR(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            // Introselect implementations
            } else if (name.startsWith(ISP)) {
                final Partition part = createPartition(name, ISP, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionISP(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IBM)) {
                final Partition part = createPartition(name, IBM, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionIBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(ISBM)) {
                final Partition part = createPartition(name, ISBM, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionISBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IKBM)) {
                final Partition part = createPartition(name, IKBM, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionIKBM(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IDNF)) {
                final Partition part = createPartition(name, IDNF, qs, hc, sc);
                function = (data, indices) -> {
                    part.partitionIDNF(data, indices.clone(), indices.length);
                    return extractIndices(data, indices);
                };
            } else if (name.startsWith(IDP)) {
                final Partition part = createPartition(name, IDP, qs, hc, sc);
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
        @Param({HEAP_SELECT, IKBM + "_HC0", IDP + "_HC0",
            // Only use for small length as sort insertion is Order(k)
            // vs Order(log(k)) for the heap.
            //SORT_SELECT
            })
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
            // Direct use of heapselect. This has variations which use different
            // optimisations for small heaps.
            // Note: Optimisation for small heap size (n=1,2) is not observable on large data.
            // It requires the use of small data (e.g. len=[16, 32)) to observe differences.
            // The main overhead is the test for insertion against the current top of the
            // heap which grows increasingly unlikely as the range is scanned.
            // Optimisation for n=1 is negligible; for n=2 it is up to 10%. However using only
            // heapSelectRange2 is not as fast as the non-optimised heapSelectRange0
            // when the heap is size 1. For n=1 the heap insertion branch prediction
            // can learn the heap has no children and skip descending the heap, whereas
            // heap size n=2 can descend 1 level if the child is smaller/bigger. This is not
            // as fast as dedicated code for the single child case.
            // This benchmark requires repeating with variable heap size to avoid branch
            // prediction learning what to do.
            if (HEAP_SELECT.equals(name)) {
                function = (data, indices) -> {
                    heapSelectRange0(data, 0, data.length - 1, indices[0], indices[1]);
                    return extractIndices(data, indices[0], indices[1]);
                };
            } else if ((HEAP_SELECT + "1").equals(name)) {
                function = (data, indices) -> {
                    heapSelectRange1(data, 0, data.length - 1, indices[0], indices[1]);
                    return extractIndices(data, indices[0], indices[1]);
                };
            } else if ((HEAP_SELECT + "2").equals(name)) {
                function = (data, indices) -> {
                    heapSelectRange2(data, 0, data.length - 1, indices[0], indices[1]);
                    return extractIndices(data, indices[0], indices[1]);
                };
            } else if ((HEAP_SELECT + "12").equals(name)) {
                function = (data, indices) -> {
                    heapSelectRange12(data, 0, data.length - 1, indices[0], indices[1]);
                    return extractIndices(data, indices[0], indices[1]);
                };
            // Only use on small edge as insertion is Order(k)
            } else if (SORT_SELECT.equals(name)) {
                function = (data, indices) -> {
                    Partition.sortSelectRange(data, 0, data.length - 1, indices[0], indices[1]);
                    return extractIndices(data, indices[0], indices[1]);
                };
            // introselect methods - these should be configured to not use heapselect
            } else if (name.startsWith(IKBM)) {
                final Partition part = createPartition(name, IKBM, 0, 0, 0);
                function = (data, indices) -> {
                    part.introselect(Partition::partitionSBM, data,
                        0, data.length - 1, IndexIntervals.interval(indices[0], indices[1]), 10000);
                    return extractIndices(data, indices[0], indices[1]);
                };
            } else if (name.startsWith(IDP)) {
                final Partition part = createPartition(name, IDP, 0, 0, 0);
                function = (data, indices) -> {
                    part.introselect(Partition::partitionDP, data,
                        0, data.length - 1, IndexIntervals.interval(indices[0], indices[1]), 10000);
                    return extractIndices(data, indices[0], indices[1]);
                };
            } else {
                throw new IllegalStateException("Unknown edge selector function: " + name);
            }
        }

        /**
         * Partition the elements between {@code ka} and {@code kb} using a heap select
         * algorithm. It is assumed {@code left <= ka <= kb <= right}.
         *
         * <p>Note:
         *
         * <p>This is a copy of {@link Partition#heapSelectRange(double[], int, int, int, int)}.
         * It uses no optimised versions for small heaps.
         *
         * @param a Data array to use to find out the K<sup>th</sup> value.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower index to select.
         * @param kb Upper index to select.
         */
        static void heapSelectRange0(double[] a, int left, int right, int ka, int kb) {
            if (right - left < Partition.MIN_HEAPSELECT_SIZE) {
                Sorting.sort(a, left, right);
                return;
            }
            if (kb - left < right - ka) {
                Partition.heapSelectLeft(a, left, right, kb, kb - ka);
            } else {
                Partition.heapSelectRight(a, left, right, ka, kb - ka);
            }
        }

        /**
         * Partition the elements between {@code ka} and {@code kb} using a heap select
         * algorithm. It is assumed {@code left <= ka <= kb <= right}.
         *
         * <p>Note:
         *
         * <p>This is a copy of {@link Partition#heapSelectRange(double[], int, int, int, int)}.
         * It uses no optimised versions for small heap of size 1.
         *
         * @param a Data array to use to find out the K<sup>th</sup> value.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower index to select.
         * @param kb Upper index to select.
         */
        static void heapSelectRange1(double[] a, int left, int right, int ka, int kb) {
            if (right - left < Partition.MIN_HEAPSELECT_SIZE) {
                Sorting.sort(a, left, right);
                return;
            }
            if (kb - left < right - ka) {
                // Optimise
                if (kb == left) {
                    Partition.selectMinIgnoreZeros(a, left, right);
                } else {
                    Partition.heapSelectLeft(a, left, right, kb, kb - ka);
                }
            } else {
                // Optimise
                if (ka == right) {
                    Partition.selectMaxIgnoreZeros(a, left, right);
                } else {
                    Partition.heapSelectRight(a, left, right, ka, kb - ka);
                }
            }
        }

        /**
         * Partition the elements between {@code ka} and {@code kb} using a heap select
         * algorithm. It is assumed {@code left <= ka <= kb <= right}.
         *
         * <p>Note:
         *
         * <p>This is a copy of {@link Partition#heapSelectRange(double[], int, int, int, int)}.
         * It uses optimised versions for small heap of size 2.
         *
         * @param a Data array to use to find out the K<sup>th</sup> value.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower index to select.
         * @param kb Upper index to select.
         */
        static void heapSelectRange2(double[] a, int left, int right, int ka, int kb) {
            if (right - left < Partition.MIN_HEAPSELECT_SIZE) {
                Sorting.sort(a, left, right);
                return;
            }
            if (kb - left < right - ka) {
                // Optimise
                if (kb - 1 <= left) {
                    Partition.selectMin2IgnoreZeros(a, left, right);
                } else {
                    Partition.heapSelectLeft(a, left, right, kb, kb - ka);
                }
            } else {
                // Optimise
                if (ka + 1 >= right) {
                    Partition.selectMax2IgnoreZeros(a, left, right);
                } else {
                    Partition.heapSelectRight(a, left, right, ka, kb - ka);
                }
            }
        }

        /**
         * Partition the elements between {@code ka} and {@code kb} using a heap select
         * algorithm. It is assumed {@code left <= ka <= kb <= right}.
         *
         * <p>Note:
         *
         * <p>This is a copy of {@link Partition#heapSelectRange(double[], int, int, int, int)}.
         * It uses optimised versions for small heap of size 1 and 2.
         *
         * @param a Data array to use to find out the K<sup>th</sup> value.
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         * @param ka Lower index to select.
         * @param kb Upper index to select.
         */
        static void heapSelectRange12(double[] a, int left, int right, int ka, int kb) {
            if (right - left < Partition.MIN_HEAPSELECT_SIZE) {
                Sorting.sort(a, left, right);
                return;
            }
            if (kb - left < right - ka) {
                // Optimise
                if (kb - 1 <= left) {
                    if (kb == left) {
                        Partition.selectMinIgnoreZeros(a, left, right);
                    } else {
                        Partition.selectMin2IgnoreZeros(a, left, right);
                    }
                } else {
                    Partition.heapSelectLeft(a, left, right, kb, kb - ka);
                }
            } else {
                // Optimise
                if (ka + 1 >= right) {
                    if (ka == right) {
                        Partition.selectMaxIgnoreZeros(a, left, right);
                    } else {
                        Partition.selectMax2IgnoreZeros(a, left, right);
                    }
                } else {
                    Partition.heapSelectRight(a, left, right, ka, kb - ka);
                }
            }
        }

        /**
         * Extract the data at the specified indices.
         *
         * @param data Data.
         * @param l Lower bound (inclusive).
         * @param r Upper bound (inclusive).
         * @return the data
         */
        private static double[] extractIndices(double[] data, int l, int r) {
            final double[] x = new double[r - l + 1];
            for (int i = l; i <= r; i++) {
                x[i - l] = data[i];
            }
            return x;
        }
    }

    /**
     * Source of an search function. This is a function that find an index
     * in a sorted list of indices.
     */
    @State(Scope.Benchmark)
    public static class IndexSearchFunctionSource {
        /** Name of the source. */
        @Param({"Binary",
            //"binarySearch",
            "Scan"})
        private String name;

        /** The action. */
        private SearchFunction function;

        /**
         * Define a search function.
         */
        public interface SearchFunction {
            /**
             * Find the index of the element {@code k}, or the closest index
             * to the element (implementation definitions may vary).
             *
             * @param a Data.
             * @param k Element.
             * @return the index
             */
            int find(int[] a, int k);
        }

        /**
         * @return the function
         */
        public SearchFunction getFunction() {
            return function;
        }

        /**
         * Create the function.
         */
        @Setup
        public void setup() {
            Objects.requireNonNull(name);
            if ("Binary".equals(name)) {
                function = (keys, k) -> Partition.searchLessOrEqual(keys, 0, keys.length - 1, k);
            } else if ("binarySearch".equals(name)) {
                function = (keys, k) -> Arrays.binarySearch(keys, 0, keys.length, k);
            } else if ("Scan".equals(name)) {
                function = (keys, k) -> {
                    // Assume that k >= keys[0]
                    int i = keys.length;
                    do {
                        --i;
                    } while (keys[i] > k);
                    return i;
                };
            } else {
                throw new IllegalStateException("Unknown index search function: " + name);
            }
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
            .withKthSelector(createKthSelector(name, prefix, 0));
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
            .withPartition(createPartition(name, prefix, 0, 0, 0));
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
     * @param qs Minimum quickselect size (if non-zero).
     * @return the {@link KthSelector} instance
     */
    static KthSelector createKthSelector(String name, String prefix, int qs) {
        final String[] s = {name};
        final int minQuickSelectSize = qs != 0 ? qs : getMinQuickSelectSize(s);
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
     * @param qs Minimum quickselect size (if non-zero).
     * @param hc Minimum heapselect constant (if non-zero).
     * @param sc Minimum sortselect constant (if non-zero).
     * @return the {@link Partition} instance
     */
    static Partition createPartition(String name, String prefix, int qs, int hc, int sc) {
        final String[] s = {name};
        final PivotingStrategy sp = getPivotStrategy(s);
        final DualPivotingStrategy dp = getDualPivotStrategy(s);
        final int minQuickSelectSize = qs != 0 ? qs : getMinQuickSelectSize(s);
        final int heapSelectShift = getHeapSelectShift(s);
        final int heapSelectConstant = hc != 0 ? hc : getHeapSelectConstant(s);
        final int heapSelectMaskShift = getHeapSelectMaskShift(s);
        final int sortSelectConstant = sc != 0 ? sc : getSortSelectConstant(s);
        final int subSamplingSize = getSubSamplingSize(s);
        final KeyStrategy keyStartegy = getKeyStrategy(s);
        final PairedKeyStrategy pairedKeyStartegy = getPairedKeyStrategy(s);
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
            heapSelectShift, heapSelectConstant, heapSelectMaskShift,
            sortSelectConstant, subSamplingSize);
        // Some values do not have to be final as they are not used within optimised
        // partitioning code.
        p.setKeyStrategy(keyStartegy);
        p.setPairedKeyStrategy(pairedKeyStartegy);
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
     * Gets the constant for the sortselect distance-from-end computation.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the sortselect constant
     */
    static int getSortSelectConstant(String[] name) {
        final Matcher m = SC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.SORTSELECT_CONSTANT;
    }

    /**
     * Gets the minimum size for single-pivot sub-sampling.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the sub-sampling size
     */
    static int getSubSamplingSize(String[] name) {
        final Matcher m = SU_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.SUBSAMPLING_SIZE;
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
        return Partition.COMPRESSION_LEVEL;
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
     * Gets the multiple key strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the key strategy
     */
    static KeyStrategy getKeyStrategy(String[] name) {
        return getKeyStrategyOrElse(name, Partition.KEY_STRATEGY);
    }

    /**
     * Gets the multiple key strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the key strategy
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
     * Gets the 1 or 2 key strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the paired key strategy
     */
    static PairedKeyStrategy getPairedKeyStrategy(String[] name) {
        return getPairedKeyStrategyOrElse(name, Partition.PAIRED_KEY_STRATEGY);
    }

    /**
     * Gets the 1 or 2 key strategy for the recursive partition algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the paired key strategy
     */
    static PairedKeyStrategy getPairedKeyStrategyOrElse(String[] name, PairedKeyStrategy defaultValue) {
        int len = 0;
        PairedKeyStrategy result = defaultValue;
        for (final PairedKeyStrategy s : PairedKeyStrategy.values()) {
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
    public void quantiles(QuantileFunctionSource function, DataSource source,
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
    public void quantileRange(QuantileFunctionSource function, DataSource source,
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
        final BiFunction<double[], int[], double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            // Note: This uses the indices without cloning. This is because some
            // functions do not destructively modify the data.
            bh.consume(fun.apply(source.getData(j), source.getIndices(j)));
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
        final BiFunction<double[], int[], double[]> fun = function.getFunction();
        for (int j = -1; ++j < size;) {
            bh.consume(fun.apply(source.getData(j), source.getIndices(j)));
        }
    }

    // TODO
    // Benchmark for EdgeFunctionSource and KSource to be used with k=1 and length
    // 6 to test the various versions of heapselect with a random index. This prevents
    // branch prediction learning the heap size (as is possible with the EdgeSource).

    /**
     * Benchmark the search of an ordered set of indices.
     *
     * @param function Source of the search.
     * @param source Source of the data.
     * @return value to consume
     */
    @Benchmark
    public long indexSearch(IndexSearchFunctionSource function, SplitIndexSource source) {
        final IndexSearchFunctionSource.SearchFunction fun = function.getFunction();
        // Ensure we have something to consume during the benchmark
        long sum = 0;
        for (int i = source.samples(); --i >= 0;) {
            // Single point in the range
            sum += fun.find(source.getIndices(i), source.getPoint(i));
        }
        return sum;
    }

    /**
     * Benchmark the tracking of an interval of indices during a partition algorithm.
     *
     * <p>The {@link SearchableInterval} is created for the indices of interest. These are then
     * cut at all points in the interval between indices to simulate a partition algorithm
     * dividing the data and requiring a new interval to use in each part:
     * <pre>{@code
     *            cut
     *             |
     * -------k1--------k2---------k3---- ... ---------kn--------
     *          <-- scan previous
     *    scan next -->
     * }</pre>
     *
     * <p>Note: If a cut is made in the interval then the smallest region of data
     * that was most recently partitioned was the length between the two flanking k.
     * This involves a full scan (and partitioning) over the data of length (k2 - k1).
     * A BitSet-type structure will require a scan over 1/64 of this length of data
     * to find the next and previous index from a cut point. In practice
     * the interval may be partitioned over a much larger length, e.g. (kn - k1).
     * Thus the length of time for the partition algorithm is expected to be at least
     * 64x the length of time for the BitSet-type scan. The disadvantage of the
     * BitSet-type structure is memory consumption. For a small number of keys the
     * structures that search the entire set of keys are fast enough. At very high
     * density the BitSet-type structures are preferred.
     *
     * @param function Source of the interval.
     * @param source Source of the data.
     * @return value to consume
     */
    @Benchmark
    public long searchableIntervalNextPrevious(SearchableIntervalSource function, SplitIndexSource source) {
        final int[][] indices = source.getIndices();
        final int[][] points = source.getPoints();
        // Ensure we have something to consume during the benchmark
        long sum = 0;
        for (int i = 0; i < indices.length; i++) {
            final int[] x = indices[i];
            final int[] p = points[i];
            final SearchableInterval interval = function.create(x);
            for (final int k : p) {
                sum += interval.nextIndex(k);
                sum += interval.previousIndex(k);
            }
        }
        return sum;
    }

    /**
     * Benchmark the tracking of an interval of indices during a partition algorithm.
     *
     * <p>This is similar to
     * {@link #searchableIntervalNextPrevious(SearchableIntervalSource, SplitIndexSource)}. It uses the
     * {@link SearchableInterval#split(int, int, int[])} method. This requires {@code k} to be
     * in an open interval. Some modes of the {@link IndexSource} do not ensure that
     * {@code left < k < right} for all split points so we have to check this before
     * calling the split method (it is a fixed overhead for the benchmark).
     *
     * @param function Source of the interval.
     * @param source Source of the data.
     * @return value to consume
     */
    @Benchmark
    public long searchableIntervalSplit(SearchableIntervalSource function, SplitIndexSource source) {
        final int[][] indices = source.getIndices();
        final int[][] points = source.getPoints();
        // Ensure we have something to consume during the benchmark
        long sum = 0;
        final int[] bound = {0};
        for (int i = 0; i < indices.length; i++) {
            final int[] x = indices[i];
            final int[] p = points[i];
            // Note: A partition algorithm would only call split if there are indices
            // above and below the split point.
            final SearchableInterval interval = function.create(x);
            final int left = interval.left();
            final int right = interval.right();
            for (final int k : p) {
                // Check k is in the open interval (left, right)
                if (left < k && k < right) {
                    sum += interval.split(k, k, bound);
                    sum += bound[0];
                }
            }
        }
        return sum;
    }

    /**
     * Benchmark the creation of an interval of indices for controlling a partition algorithm.
     *
     * <p>This baselines the {@link #searchableIntervalNextPrevious(SearchableIntervalSource, SplitIndexSource)} benchmark.
     * For the BitSet-type structures a large overhead is the memory allocation to create
     * the {@link SearchableInterval}. Note that this will be at most 1/64 the size of the array
     * that is being partitioned and in practice this overhead is not significant.
     *
     * @param function Source of the interval.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void createSearchableInterval(SearchableIntervalSource function, IndexSource source, Blackhole bh) {
        final int[][] indices = source.getIndices();
        for (final int[] x : indices) {
            bh.consume(function.create(x));
        }
    }

    /**
     * Benchmark the splitting of an interval of indices during a partition algorithm.
     *
     * <p>This is similar to
     * {@link #searchableIntervalSplit(SearchableIntervalSource, SplitIndexSource)}. It uses the
     * {@link UpdatingInterval#splitLeft(int, int)} method by recursive division of the indices.
     *
     * @param function Source of the interval.
     * @param source Source of the data.
     * @param bh Data sink.
     */
    @Benchmark
    public void updatingIntervalSplit(UpdatingIntervalSource function, IndexSource source, Blackhole bh) {
        final int[][] indices = source.getIndices();
        final int s = source.getMinSeparation();
        for (int i = 0; i < indices.length; i++) {
            split(function.create(indices[i]), s, bh);
        }
    }

    /**
     * Recursively split the interval until the length is below the provided separation.
     * Consume the interval when no more divides can occur.
     * Simulates a single-pivot partition algorithm.
     *
     * @param interval Interval.
     * @param s Minimum separation between left and right.
     * @param bh Data sink.
     */
    private static void split(UpdatingInterval interval, int s, Blackhole bh) {
        int l = interval.left();
        final int r = interval.right();
        // Note: A partition algorithm would only call split if there are indices
        // above and below the split point.
        if (r - l > s) {
            final int middle = (l + r) >>> 1;
            // recurse left
            split(interval.splitLeft(middle, middle), s, bh);
            // continue on right side
            l = interval.left();
        }
        bh.consume(interval);
    }
}
