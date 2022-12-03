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

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import org.apache.commons.statistics.inference.KolmogorovSmirnovDistribution.One.ScaledPower;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvFileSource;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Test cases for {@link KolmogorovSmirnovDistribution}.
 */
class KolmogorovSmirnovDistributionTest {
    /** Minimum relative epsilon between double values. */
    private static final double EPS = Math.ulp(1.0);
    /** Proxy to trigger the default power function for the double-double One.sf computation. */
    private static final ScaledPower DEFAULT_POWER = null;
    /** Proxy to trigger the default MathContext for the BigDecimal One.sf computation. */
    private static final MathContext DEFAULT_MC = null;

    // Unless otherwise stated the test data is from scipy (1.9.3):
    // from scipy.stats import kstwo, ksone
    // import numpy as np
    // np.set_printoptions(precision=20)
    // kstwo.sf(x, n)
    // ksone.sf(x, n)
    //
    // Can be timed in seconds using e.g:
    // from time import time
    // t1 = time(); ksone.sf(1e-2, 1000000); time() - t1
    // 1.3686652207740602e-87
    // 1.1550960540771484
    // Here the scipy implementation takes approximately 1 second at the n used
    // for the asymptotic limit.

    /**
     * Test cases of the two-sided survival function where there is an exact representation.
     */
    @ParameterizedTest
    @CsvSource({
        // Lower limit
        "1, 10, 0",
        "1, 100, 0",
        "1.23, 10, 0", // invalid x
        // nxx >= 370
        "0.5, 1480, 0",
        "0.1, 37000, 0",
        "1e-3, 370000000, 0",
        // Upper limit
        "0, 10, 1",
        "0, 100, 1",
        "-1, 10, 1", // invalid x
        // n = 1
        "0.0125, 1, 1",
        "0.75, 1, 0.5",
        "0.789, 1, 0.42199999999999993",
        "0.99999999999, 1, 2.000000165480742e-11",
        // x <= 1/(2n)
        "0.05, 10, 1",
        "0.025, 20, 1",
        "0.0125, 40, 1",
        // 1/(2n) < x <= 1/n
        "0.075, 10, 0.999999645625",
        "0.045, 20, 0.9999999997324996",
        "0.025, 40, 0.9999999999999999",
        // 1 - 1/n <= x < 1
        "0.99, 10, 2.0000000000000176e-20",
        "0.95, 10, 1.9531250000000172e-13",
        "0.999, 100, 2.0000000000001778e-300",
        "0.995, 100, 1.5777218104421636e-230",
    })
    void testTwoSFExact(double x, int n, double p) {
        final double p2 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        TestUtils.assertProbability(p, p2, EPS, "sf");
    }

    /**
     * Test cases of the two-sided survival function where it uses the Durbin MTW algorithm.
     * This has been verified to invoke the correct function by throwing an exception from
     * within the method.
     */
    @ParameterizedTest
    @CsvSource({
        // n <= 140. Expected precision is 10-digits
        // 1/n < n*x < 1 - 1/n; n*x*x < 0.754693
        "0.101, 10, 0.9995670338682896, 2e-16",
        "0.15, 10, 0.95396527, 2e-16",
        "0.22, 10, 0.6425444017073398, 4e-16",
        "0.274, 10, 0.3715203845434957, 3e-16",
        "0.02, 100, 0.999999999978262, 2e-16",
        "0.04, 100, 0.995307510717411, 2e-16",
        "0.06, 100, 0.8428847956274123, 3e-16",
        "0.0868, 100, 0.4148814057122582, 2e-15",
        "0.018, 140, 0.9999999997083645, 2e-16",
        "0.04, 140, 0.9719726587618661, 2e-16",
        "0.06, 140, 0.6719092248862638, 5e-16",
        "0.0734, 140, 0.4177062319533004, 3e-15",
        // 140 < n <= 100000 && n * Math.pow(x, 1.5) < 1.4. Expected precision is 5 digits.
        "0.008, 1000, 0.9999999133431808, 2e-16",
        "0.009, 1000, 0.9999964462512184, 2e-16",
        "0.01, 1000, 0.9999496745370611, 2e-16",
        "0.0125, 1000, 0.9971528020597908, 2e-16",
        "0.002, 10000, 0.9999999999991761, 2e-16",
        "0.0024, 10000, 0.9999999930856749, 2e-16",
        "0.00269, 10000, 0.9999995514082088, 2e-16",
        "0.000922, 50000, 0.999999999996305, 2e-16",
    })
    void testDurbinMTW(double x, int n, double p, double eps) {
        final double p2 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        TestUtils.assertProbability(p, p2, eps, "sf");
    }

    /**
     * Test the computation of the A factors for the Pomeranz algorithm is correct for the 3 cases.
     * The parameters have been chosen so that the use of floor(A - t) and ceil(A - t)
     * does not suffer rounding error. This verifies the implementation, which avoids
     * using floor/ceil, is correct.
     */
    @ParameterizedTest
    @CsvSource({
        // t = n*x; f = t - floor(t)

        // f = 0
        "0.125, 8",

        // 0 < f <= 1/2
        "0.15625, 8",
        "0.1875, 8",

        // 1/2 < f
        "0.203125, 8",
        "0.21875, 8",
    })
    void testPomeranzComputeA(double x, int n) {
        final double t = n * x;
        final double[] a = new double[2 * n + 3];
        final int[] amt = new int[a.length];
        final int[] apt = new int[a.length];
        KolmogorovSmirnovDistribution.Two.computeA(n, t, amt, apt);

        // Create A
        a[1] = 0;
        a[2] = Math.min(t - Math.floor(t), Math.ceil(t) - t);
        a[3] = 1 - a[2];
        a[2 * n + 2] = n;
        final int max = 2 * n + 2;
        final double f = t - Math.floor(t);
        // 3-cases (see Simard and L’Ecuyer (2011))
        if (f > 0.5) {
            // Case (iii): 1/2 < f < 1
            // for i = 1, 2, ...
            for (int i = 1; 2 * i < max; i++) {
                a[2 * i] = i - f;
                if (2 * i + 1 < max) {
                    a[2 * i + 1] = i - 1 + f;
                }
            }
        } else if (f > 0) {
            // Case (ii): 0 < f <= 1/2
            // for i = 1, 2, ...
            for (int i = 1; 2 * i < max; i++) {
                a[2 * i] = i - 1 + f;
                if (2 * i + 1 < max) {
                    a[2 * i + 1] = i - f;
                }
            }
        } else {
            // Case (i): f = 0
            // for i = 1, 2, ...
            for (int i = 1; 2 * i < max; i++) {
                a[2 * i] = i - 1;
                if (2 * i + 1 < max) {
                    a[2 * i + 1] = i;
                }
            }
        }
        // Check all floor/ceil elements for A[i-1] for i in [2, 2n+2]
        for (int i = 2; i <= max; i++) {
            final int im1 = i - 1;
            Assertions.assertEquals(Math.floor(a[im1] - t), amt[im1], () ->
                String.format("floor(A[%d] + t == floor(%s - %s)) = floor(%s)", im1, a[im1], t, a[im1] - t));
            Assertions.assertEquals(Math.ceil(a[im1] + t), apt[im1], () ->
                String.format("ceil(A[%d] + t) == ceil(%s + %s) = ceil(%s)", im1, a[im1], t, a[im1] + t));
        }
    }

    /**
     * Test cases of the two-sided survival function where it uses the Pomeranz algorithm.
     * This has been verified to invoke the correct function by throwing an exception from
     * within the method.
     *
     * <p>Note: Here Simard and L’Ecuyer expect 10 digits of precision so
     * eps less than 1e-10 is within the expected range.
     */
    @ParameterizedTest
    @CsvSource({
        // Note: f = n*x - floor(n*x). 3 cases are {f = 0, 0 < f <= 0.5, 0.5 < f}.
        // 1/n < n*x < 1 - 1/n; 0.754693 < n*x*x < 4
        // Values of x for n=10 create f of {0.75, 0, 0.25, 0, 0.5, 0.320}
        "0.275, 10, 0.36721918274907195, 2e-16",
        "0.3, 10, 0.27053557479999946, 5e-16",
        "0.325, 10, 0.19329796645948394, 2e-15",
        "0.5, 10, 0.00777741, 2e-13",
        "0.55, 10, 0.0022805103214843725, 4e-13",
        "0.632, 10, 0.00021216261775257054, 8e-14",
        // Values of x for n=100 create f of {0.690, 0, 0.5, 0.050, 0.90}
        "0.0869, 100, 0.41345306880916205, 5e-16",
        "0.12, 100, 0.10330374901819939, 9e-15",
        "0.125, 100, 0.08050040280210224, 9e-15",
        "0.1605, 100, 0.010204399956967765, 3e-14",
        "0.199, 100, 0.0006024947156633567, 2e-12",
        // Values of x for n=140 create f of {0.2899, 0.599, 0.40, 0, 0.660}
        "0.0735, 140, 0.41601087723723057, 5e-16",
        "0.09, 140, 0.1946946629845738, 2e-16",
        "0.11, 140, 0.06252511429470399, 7e-15",
        "0.15, 140, 0.0032495604415101703, 2e-13",
        "0.169, 140, 0.0005778682806183945, 2e-13",
    })
    void testPomeranz(double x, int n, double p, double eps) {
        final double p2 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        TestUtils.assertProbability(p, p2, eps, "sf");
    }

    @Test
    void testPelzGoodApproximation() {
        // Test ported from o.a.c.math where the Pelz-Good algorithm computed the CDF.
        // Note: These values are not really appropriate as the Pelz-Good method
        // is not used when n*x*x > 2.2. Reference values have been updated using
        // scipy.stats.kstwo.cdf to verify the method is valid for this range.
        final double[] ds = {0.15, 0.20, 0.25, 0.3, 0.35, 0.4};
        final int[] ns = {141, 150, 180, 220, 1000};
        final double[] ref = {
            0.9968940168727817, 0.9979326624184855, 0.9994677598604502, 0.9999128354780206, 1,
            0.9999797514476229, 0.9999902122242085, 0.9999991327060904, 0.9999999657682075, 1,
            0.9999999706445153, 0.9999999906571525, 0.9999999997949723, 0.9999999999987504, 1,
            0.9999999999916627, 0.9999999999984474, 0.9999999999999944, 1, 1,
            0.9999999999999996, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        };

        final double tol = 1e-15;
        int k = 0;
        for (final double x : ds) {
            for (final int n : ns) {
                TestUtils.assertProbability(ref[k++],
                    1 - KolmogorovSmirnovDistribution.Two.pelzGood(x, n), tol, () -> String.format("%s %d", x, n));
            }
        }
    }

    /**
     * Test cases of the two-sided survival function where it uses the Pelz-Good algorithm.
     * This has been verified to invoke the correct function by throwing an exception from
     * within the method.
     */
    @ParameterizedTest
    @CsvSource({
        // n * x^2 < 2.2
        // 140 < n <= 100000 && n * Math.pow(x, 1.5) >= 1.4. Expected precision is 5 digits.
        "0.0126, 1000, 0.9968143322478163, 2e-16",
        "0.02, 1000, 0.8108971656895577, 2e-16",
        "0.03, 1000, 0.3226902143914636, 2e-15",
        "0.0469, 1000, 0.023788979220138784, 2e-16",
        "0.00270, 10000, 0.999999494161441, 2e-16",
        "0.005, 10000, 0.9628778025204304, 2e-16",
        "0.01, 10000, 0.2682191277029192, 2e-15",
        "0.0148, 10000, 0.02478199615804111, 3e-14",
        "0.000923, 50000, 0.9999999999960723, 2e-16",
        "0.0025, 50000, 0.9126805535490892, 2e-16",
        "0.004, 50000, 0.3994316646755731, 5e-16",
        "0.00663, 50000, 0.02455135772431216, 9e-15",
        // 100000 < n
        "0.0006, 150000, 0.9999999985986439, 2e-16",
        "0.001, 150000, 0.9982358422822605, 2e-16",
        "0.002, 150000, 0.5852546878861703, 2e-16",
        "0.00382, 150000, 0.025043795432396876, 4e-15",
        // Threshold CDF ~ min_normal
        // z = x * sqrt(n) ~ 0.0417  [ sqrt(-pi^2 / 8 / ln(2^-1023)) ]
        "0.0001077, 150000, 1, 0",
        "0.0001078, 150000, 1, 0",
        // Threshold SF ~ 1
        // z = x * sqrt(n) ~ 0.18325  [ sqrt(-pi^2 / 8 / ln(2^-53)) ]
        "0.00048, 150000, 0.999999999999995, 0",
        "0.00046, 150000, 0.9999999999999998, 0",
        "0.00044, 150000, 1, 0",
    })
    void testPelzGood(double x, int n, double p, double eps) {
        final double p2 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        TestUtils.assertProbability(p, p2, eps, "sf");
    }

    /**
     * Test cases of the two-sided survival function where it uses the Miller approximation.
     * This has been verified to invoke the correct function by throwing an exception from
     * within the method.
     */
    @ParameterizedTest
    @CsvSource({
        // n <= 140. Expected precision is 10-digits
        // n*x*x > 4
        "0.64, 10, 0.00016378151748370424, 2e-16",
        "0.65, 10, 0.00011761597734374991, 2e-16",
        "0.66, 10, 8.372225302495224e-05, 2e-16",
        "0.7, 10, 1.9544800000000034e-05, 2e-16",
        "0.85, 10, 1.1566210937500017e-08, 2e-16",
        "0.21, 100, 0.00023947345214651332, 2e-16",
        "0.23, 100, 3.921924383021513e-05, 2e-16",
        "0.25, 100, 5.408871776434847e-06, 2e-16",
        "0.6, 100, 5.912822156396238e-35, 2e-16",
        "0.17, 140, 0.0005246108687519814, 2e-16",
        "0.18, 140, 0.0001932001834783552, 2e-16",
        "0.19, 140, 6.711071724199073e-05, 2e-16",
        "0.7, 140, 2.7611948816463573e-69, 2e-16",
        // n > 140 && n*x*x > 2.2
        "0.064, 1000, 0.0005275756095069879, 2e-16",
        "0.07, 1000, 0.00010494206285958879, 2e-16",
        "0.075, 1000, 2.445848323871376e-05, 2e-16",
        "0.08, 1000, 5.154189384789824e-06, 2e-16",
        "0.1, 1000, 3.703687096817711e-09, 2e-16",
        "0.3, 1000, 2.590715705736845e-80, 2e-16",
        "0.0201, 10000, 0.0006106415730821294, 2e-16",
        "0.022, 10000, 0.00012312039864307816, 2e-16",
        "0.025, 10000, 7.319405460271144e-06, 2e-16",
        "0.03, 10000, 2.9761211950626197e-08, 2e-16",
        "0.04, 10000, 2.4399624749525048e-14, 2e-16",
        "0.1, 10000, 1.6633113315950355e-87, 2e-16",
    })
    void testMillerApproximation(double x, int n, double p, double eps) {
        final double p2 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        TestUtils.assertProbability(p, p2, eps, "sf");
    }

    /**
     * Test cases of the one-sided survival function where there is an exact representation.
     */
    @ParameterizedTest
    @CsvSource({
        // Lower limit
        "1, 10, 0",
        "1, 100, 0",
        "1.23, 10, 0", // invalid x
        // 2 * nxx >= 745
        "0.5, 1490, 0",
        "0.1, 37250, 0",
        "1e-3, 372500000, 0",
        // Upper limit
        "0, 10, 1",
        "0, 100, 1",
        "-1, 10, 1", // invalid x
        // n = 1
        "0.012345, 1, 0.012345",
        "0.025, 1, 0.025",
        "0.99999999999, 1, 0.99999999999",
        // x <= 1/n
        "0.05, 10, 0.9224335892010742",
        "0.025, 20, 0.9600337453587708",
        "0.0125, 40, 0.9797084016853456",
        "0.075, 10, 0.8562071003140899",
        "0.045, 20, 0.8961462860117863",
        "0.025, 40, 0.9345106380880495",
        "2.5e-6, 400000, 0.9999932043209127",
        "1e-15, 400000, 0.999999999999999",
        // 1 - 1/n <= x < 1
        "0.99, 10, 1.0000000000000088e-20",
        "0.95, 10, 9.765625000000086e-14",
        "0.9999999, 10, 9.999999947364416e-71",
        "0.999, 100, 1.0000000000000889e-300",
        "0.995, 100, 7.888609052210818e-231",
    })
    void testOneSFExact(double x, int n, double p) {
        final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n);
        final double p2 = onesf(x, n, DEFAULT_MC);
        TestUtils.assertProbability(p, p1, EPS, "sf");
        TestUtils.assertProbability(p, p2, EPS, "sf BigDecimal");
        // The double-double computation should be correct when x does not approach sub-normal
        if (x > 0x1.0p-1000) {
            final double p3 = onesf(x, n, DEFAULT_POWER);
            TestUtils.assertProbability(p, p3, EPS, "sf double-double");
        }
    }

    /**
     * Test cases of the asymptotic approximation to the one-sided survival function.
     * Cases must use {@code 1/n < x < 1 - 1/n} to avoid the exact computation
     * and a large n to trigger the asymptotic computation.
     *
     * <p>Notes:
     * <ul>
     * <li>At the switch point the asymptotic approximation agrees to ~ 6 digits
     * until p is much smaller than a realistic alpha for significance testing.
     * <li>Use powScaled is with 1 ulp of fastPowScaled and ~ 20% slower. Thus the power
     * function is not changing the precision of the result.
     * </ul>
     */
    @ParameterizedTest
    @CsvSource({
        // n = 10^6 + 1 (threshold to switch to the asymptotic
        // Computed using: numpy.exp(-pow(6.0*n*x+1.0, 2)/(18.0*n)):
        "1e-4, 1000001, 0.9801332548522448, 2e-16",
        "1e-3, 1000001, 0.1352448117787778, 2e-16",
        "1.5e-3, 1000001, 0.011097842537398537, 2e-16",
        "1.75e-3, 1000001, 0.0021849270292376324, 2e-16",
        "2e-3, 1000001, 0.0003350129437289286, 2e-16",
        "2.5e-3, 1000001, 3.7204005445026563e-06, 2e-16",
        "3e-3, 1000001, 1.5199275791041033e-08, 2e-16",
        "3.5e-3, 1000001, 2.284342265345614e-11, 2e-16",
        "4e-3, 1000001, 1.2630034559846114e-14, 2e-16", // Requires 6.0*n*x not 6.0*(n*x)
        "7e-3, 1000001, 2.7357189636541477e-43, 2e-16",
        "1e-2, 1000001, 1.374426245809477e-87, 2e-16",
        // XXX repeat timings
        // Computed using the double-double implementation (fastPowScaled, with timings)
        "1.0E-4, 1000001, 0.9801333136251243, 6e-8", // 0.634074s
        "0.001, 1000001, 0.13524481927494492, 6e-8", // 0.504505s
        "0.0015, 1000001, 0.011097829274951359, 2e-6", // 0.527094s
        "0.00175, 1000001, 0.0021849210146805925, 3e-6", // 0.512000s
        "0.002, 1000001, 3.3501117508103243E-4, 6e-6", // 0.505427s
        "0.0025, 1000001, 3.720346483702557E-6, 2e-5", // 0.488436s
        "0.003, 1000001, 1.519879017846374E-8, 4e-5", // 0.464440s
        "0.0035, 1000001, 2.2842024589556203E-11, 7e-5", // 0.457859s
        "0.004, 1000001, 1.2628687945778359E-14, 2e-4", // 0.440603s
        "0.007, 1000001, 2.7328605923747603E-43, 2e-3", // 0.383890s
        "0.01, 1000001, 1.3683915090193192E-87, 5e-3", // 0.348474s
        // Computed using the double-double implementation (powScaled, with timings)
        "1.0E-4, 1000001, 0.9801333136251243, 6e-8", // 0.736033s
        "0.001, 1000001, 0.13524481927494492, 6e-8", // 0.629389s
        "0.0015, 1000001, 0.011097829274951359, 2e-6", // 0.596412s
        "0.00175, 1000001, 0.0021849210146805925, 3e-6", // 0.591704s
        "0.002, 1000001, 3.350111750810325E-4, 6e-6", // 0.583618s
        "0.0025, 1000001, 3.720346483702557E-6, 2e-5", // 0.573333s
        "0.003, 1000001, 1.519879017846374E-8, 4e-5", // 0.554925s
        "0.0035, 1000001, 2.2842024589556203E-11, 7e-5", // 0.532223s
        "0.004, 1000001, 1.2628687945778359E-14, 2e-4", // 0.529170s
        "0.007, 1000001, 2.7328605923747603E-43, 2e-3", // 0.458782s
        "0.01, 1000001, 1.3683915090193192E-87, 5e-3", // 0.416742s
        // n = 10^7
        // Computed using: numpy.exp(-pow(6.0*n*x+1.0, 2)/(18.0*n)):
        "1e-6, 10000000, 0.9999793279914467, 2e-16",
        "2e-6, 10000000, 0.9999186644190289, 2e-16",
        "2e-4, 10000000, 0.44926905508659115, 2e-16",
        "8e-4, 10000000, 2.7593005372427517e-06, 2e-16",
        "1e-3, 10000000, 2.059779966512744e-09, 2e-16",
        // Computed using the double-double implementation (fastPowScaled, with timings)
        "1.0E-6, 10000000, 0.9999793335473409, 1e-8", // 5.499433s
        "2.0E-6, 10000000, 0.9999186699759279, 1e-8", // 5.536423s
        "2.0E-4, 10000000, 0.4492690623747514, 2e-8", // 5.431289s
        "8.0E-4, 10000000, 2.7592963140010787E-6, 2e-6", // 4.979058s
        "0.001, 10000000, 2.059771738419037E-9, 5e-6", // 4.947680s
        // Computed using the double-double implementation (powScaled, with timings)
        "1.0E-6, 10000000, 0.9999793335473409, 1e-8", // 6.965408s
        "2.0E-6, 10000000, 0.9999186699759279, 1e-8", // 6.737508s
        "2.0E-4, 10000000, 0.4492690623747514, 2e-8", // 6.905447s
        "8.0E-4, 10000000, 2.7592963140010787E-6, 2e-6", // 6.539566s
        "0.001, 10000000, 2.059771738419037E-9, 5e-6", // 6.196324s
        // n = 2^30
        // Using scipy ksone.sf found an intermediate overflow bug in v1.9.3:
        // https://github.com/scipy/scipy/issues/17775
        // Computed using: numpy.exp(-pow(6.0*n*x+1.0, 2)/(18.0*n)):
        "1e-6, 1073741824, 0.9978541552573539, 2e-16",
        "1e-5, 1073741824, 0.8067390416113425, 2e-16",
        "1e-4, 1073741824, 4.715937751790102e-10, 2e-16",
        "2e-4, 1073741824, 4.94686617627386e-38, 2e-16",
        "3e-4, 1073741824, 1.1542138829781214e-84, 2e-16",
        "5e-4, 1073741824, 6.914816492890092e-234, 2e-16",
        // Computed using the double-double implementation (fastPowScaled, with timings)
        "1.0E-6, 1073741824, 0.9978541553094259, 6e-11", // 798.105071s
        "1.0E-5, 1073741824, 0.8067390416850888, 1e-10", // 799.473844s
        "1.0E-4, 1073741824, 4.715937547939537E-10, 5e-8", // 711.366173s
        "2.0E-4, 1073741824, 4.946862487288552E-38, 8e-7", // 608.890882s
        "3.0E-4, 1073741824, 1.154209467628123E-84, 4e-6", // 552.846079s
        "5.0E-4, 1073741824, 6.914611021963401E-234, 3e-5", // 481.010631s
    })
    void testOneSFAsymptotic(double x, int n, double p, double eps) {
        final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n);
        // Use to obtain double-double result with timing
        //final double p1 = onesf(x, n, DEFAULT_POWER);
        //final double p1 = onesf(x, n, DD::powScaled);
        //final double p1 = onesf(x, n, DEFAULT_MC);
        //TestUtils.assertProbability(p, p1, eps, "sf");
    }

    /**
     * Test cases of the one-sided survival function with small n.
     * Test the BigDecimal implementation and the double-double implementation
     * with their default configuration. Test the double-double implementation
     * with each available power function. This test verifies the implementations
     * all agree with the scipy reference.
     */
    @ParameterizedTest
    @CsvSource({
        // Values from scipy ksone.sf
        // 1/n < x < n - 1/n; small n
        "0.17, 6, 0.6307293944351234, 2e-16",
        "0.3, 6, 0.28207240740740747, 2e-16",
        "0.5, 6, 0.03279320987654321, 2e-16",
        "0.7, 6, 0.0009059876543209886, 2e-16",
        "0.2, 10, 0.39676169159999997, 2e-16",
        "0.4, 10, 0.02949469039999999, 2e-16",
        "0.6, 10, 0.0002840836000000002, 2e-16",
        "0.8, 10, 1.1039999999999974e-07, 2e-16",
        "0.02, 60, 0.940517651518819, 2e-16",
        "0.04, 60, 0.8041968460900994, 2e-16",
        "0.2, 60, 0.007011703368333355, 2e-16",
        "0.4, 60, 1.7743971854364937e-09, 2e-16",
        "0.02, 100, 0.9109784815556481, 2e-16",
        "0.03, 100, 0.8190484857999243, 2e-16",
        "0.3, 100, 8.859934946331458e-09, 2e-16",
        "0.5, 100, 6.065717185908929e-24, 2e-16",
        "0.7, 100, 6.105702490608996e-50, 2e-16",
        // BigDecimal computation is < 0.1 seconds
        "1e-2, 1000, 0.8133237754763678, 2e-16",
        "2e-2, 1000, 0.44342498843949424, 2e-16",
        "3e-2, 1000, 0.16203171395455085, 2e-16",
        "5e-2, 1000, 0.006506037390545166, 2e-16",
        "8e-2, 1000, 2.577094692394912e-06, 2e-16",
        "1e-1, 1000, 1.8518435484088554e-09, 2e-16",
        // BigDecimal computation is < 0.3 seconds
        "2e-2, 5000, 0.01806981293722266, 2e-16",
        // BigDecimal computation is < 0.6 seconds
        "1.5e-2, 10000, 0.01099707871209626, 2e-16",
    })
    void testOneSFSmallN(double x, int n, double p, double eps) {
        final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n);
        final double p2 = onesf(x, n, DEFAULT_MC);
        TestUtils.assertProbability(p, p1, eps, "sf");
        TestUtils.assertProbability(p, p2, eps, "sf BigDecimal");
        // For small N we can use either double-double power function.
        final double p3 = onesf(x, n, DD::fastPowScaled);
        final double p4 = onesf(x, n, DD::powScaled);
        // Check at least one computation matched the default
        Assertions.assertTrue(p1 == p3 || p1 == p4, "Default implementation differs");
        TestUtils.assertProbability(p, p3, eps, "sf fastPow");
        TestUtils.assertProbability(p, p4, eps, "sf pow");
        // simplePow also works
        final double p5 = onesf(x, n, DDExt::simplePowScaled);
        TestUtils.assertProbability(p, p5, eps, "sf simplePow");
    }

    /**
     * Test cases of the one-sided survival function with large n.
     * Test the double-double implementation with the default configuration.
     */
    @ParameterizedTest
    @CsvSource({
        // Values from scipy ksone.sf
        // 1/n < x < n - 1/n; large n; n < 1e6
        "1e-5, 12345, 0.9999886861857871, 2e-16",
        "1e-3, 12345, 0.9749625482896954, 2e-16",
        "1e-2, 12345, 0.08410601145926598, 2e-16",
        "1e-1, 12345, 3.2064615455964645e-108, 2e-16",
        "0.15, 12345, 3.007486476486515e-243, 2e-16",
        "1e-6, 123456, 0.9999988686009794, 2e-16",
        "1e-4, 123456, 0.9974674302237663, 2e-16",
        "1.5e-3, 123456, 0.5731824062390436, 2e-16",
        "5e-3, 123456, 0.002078400774523863, 2e-16",
        "6e-3, 123456, 0.00013736249820594647, 2e-16",
        "8e-3, 123456, 1.3636949699766825e-07, 2e-16",
        "3e-2, 123456, 2.9034013257947777e-97, 2e-16",
        // Close to the asymptotic limit (1e6).
        // XXX repeat timings
        // Timings for the fastPowScaled.
        "1e-6, 999000, 0.9999972844391667, 2e-16", // 0.000033s
        "1e-5, 999000, 0.9997935546914299, 2e-15", // 0.586500s
        "1e-3, 999000, 0.13551585067528177, 2e-15", // 0.526206s
        "1.5e-3, 999000, 0.011147932231639623, 2e-15", // 0.510105s
        "1e-2, 999000, 1.671698905805492e-87, 2e-15", // 0.358407s
        // Test cases requiring careful computation of
        // k = floor(n*x), alpha = nx - k; x = (k+alpha)/n with 0 <= alpha < 1
        // 0.8e-5 * 5e5 = 4.0, actually 3.99999999999999897195950...
        // 1.0e-5 * 5e5 = 5.0, actually 4.99999999999999956198232...
        // 1.4e-5 * 5e5 = 7.0, actually 6.99999999999999989499501...
        // First two cases use Smirnov-Dwass.
        // Final case uses regular computation where this implementation
        // has a more accurate sum and the scipy p-value is 11 ULP lower.
        "0.8e-5, 500000, 0.9999306695974379, 2e-16",
        "1.0e-5, 500000, 0.9998933391142139, 4e-16",
        "1.4e-5, 500000, 0.9997946878340761, 2e-15",
    })
    void testOneSFLargeN(double x, int n, double p, double eps) {
        final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n);
        //final double p1 = onesf(x, n, DEFAULT_POWER);
        TestUtils.assertProbability(p, p1, eps, "sf");
    }

    /**
     * Test cases of splitting x into {@code x = (k + alpha) / n},
     * where {@code k} is integer and {@code alpha} is in [0, 1).
     *
     * <p>Some cases (*) required to hit execution code paths
     * use {@code n} above the threshold for the asymptotic
     * approximation, or {@code n*x <= 1}. In practice the
     * code path where {@code alpha = (1.0, -eps)} *may* not occur
     * with the configured One.sf computation but is included for
     * completeness.
     */
    @ParameterizedTest
    @CsvSource({
        // 5.0/n * n = 5.0 (exact)
        "0.0390625, 128",
        // 0.2e-5 * 5e5 = 1.0, actually 0.99999999999999995474811...
        // (*) Creates k=0 alpha=(1.0,-4.525188817411374E-17)
        "0.2e-5, 500000",
        // 0.4e-5 * 5e5 = 2.0, actually 1.99999999999999990949622...
        "0.4e-5, 500000",
        // 0.8e-5 * 5e5 = 4.0, actually 3.99999999999999981899244...
        "0.8e-5, 500000",
        // 0.6e-6 * 5e6 = 3.0, actually 2.99999999999999986424433...
        "0.6e-6, 5000000",
        // 0.6e-7 * 5e7 = 3.0, actually 2.99999999999999973189543...
        "0.6e-7, 50000000",
        // 0.6e-8 * 5e8 = 3.0, actually 2.99999999999999998004962...
        // (*) Creates k=2 alpha=(1.0,-1.995037876491735E-17)
        "0.6e-8, 500000000",
    })
    void testSplitX(double x, int n) {
        final DD alpha = DD.create();
        final BigDecimal bn = bd(n);
        for (final double x1 : new double[] {x, Math.nextDown(x), Math.nextUp(x)}) {
            final int k = KolmogorovSmirnovDistribution.One.splitX(n, x1, alpha);
            final BigDecimal nx = bn.multiply(bd(x1));
            int ek = nx.round(new MathContext(16, RoundingMode.FLOOR)).intValue();
            double ealpha = nx.subtract(bd(ek)).doubleValue();
            if (ealpha == 1) {
                ek++;
                ealpha = 0;
            }
            Assertions.assertEquals(ek, k, () -> String.format("n=%d, x=%s : k", n, x1));
            Assertions.assertEquals(ealpha, alpha.hi(), () -> String.format("n=%d, x=%s : alpha", n, x1));
        }
    }

    /**
     * Test the two-sided one-sample distribution for small n. Resource files are used to
     * include a range of values uniformly for x in the interval [0, 1].
     *
     * <p>Note: Executing this test alone will hit all branches of the
     * {@link KolmogorovSmirnovDistribution.Two#sf(double, int)} with {@code n <= 140};
     */
    @ParameterizedTest
    @CsvFileSource(resources = {"ks.onesample.twosided.small.txt"}, delimiter = ' ')
    void testTwoSmall(double x, int n, double p) {
        final double p1 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        // Uses n <= 140 where there are ~ 10 digits of precision.
        // Passes at 3e-12
        TestUtils.assertProbability(p, p1, 3e-12, "sf");
    }

    /**
     * Test the two-sided one-sample distribution for large n. Resource files are used to
     * include a range of values uniformly for x in the interval [0, 1].
     *
     * <p>Note: Executing this test alone will hit all branches of the
     * {@link KolmogorovSmirnovDistribution.Two#sf(double, int)} with {@code n > 140};
     */
    @ParameterizedTest
    @CsvFileSource(resources = {"ks.onesample.twosided.large.txt"}, delimiter = ' ')
    void testTwoLarge(double x, int n, double p) {
        final double p1 = KolmogorovSmirnovDistribution.Two.sf(x, n);
        // Uses n > 140 where there are ~ 6 digits of precision.
        // Passes at 3e-14 unless p is sub-normal.
        // Note: The two-sided distribution uses the one-sided distribution when
        // the p-value is small. The SciPy implementation of ksone does not use a scaled
        // sum and thus when term Aj is zero the algorithm stops. This can make the ksone
        // sf result lower than the correct result and it cannot accurately sum sub-normal
        // numbers.
        if (p < Double.MIN_NORMAL) {
            // Test the values are approximately the same
            if (p > 1e-317) {
                // Close
                TestUtils.assertProbability(p, p1, 1e-6, "sf");
            } else if (p > 1e-321) {
                // Quite close
                TestUtils.assertProbability(p, p1, 1e-2, "sf");
            } else {
                // Very small numbers have issues. Just check p1 is also very small.
                Assertions.assertEquals(p, p1, 1e-317);
            }
        } else {
            TestUtils.assertProbability(p, p1, 3e-14, "sf");
        }
    }

    /**
     * Simple speed test to provide an approximate run-time for the computation for different
     * {@code n}. Uses {@code 1/n < x < n - 1/n} where exact computations do not exist.
     * When x > 1/sqrt(n) terms peak in magnitude around (n-nx)/2.
     * Worst case is when Smirnov-Dwass does not apply and x is small.
     * This uses powers of two and the speed approximately halves at each increasing n.
     * The asymptotic limit is at 1e7 (approximately 2^23.25).
     *
     * <p>Note: This is not a substitute for a JMH benchmark but does provide useful
     * information. Disable by setting {@code maxP < minP}.
     */
    @ParameterizedTest
    @CsvSource({
        // Exercises all the methods and checks they compute the same result (takes 0.2 seconds)
        "4, 7, false",
        // For useful information on speed (takes 5+ minutes to run)
        //"7, 24, true",
    })
    void testSpeed(int minP, int maxP, boolean output) {
        Assumptions.assumeTrue(minP <= maxP, "Skipping speed test");
        // At small n the Smirnov-Dwass option is ignored.
        // Since we directly call the computation keep x within [0, 1]
        // by limiting n to 16 (thus 11/n is always valid).
        Assertions.assertTrue(4 <= minP, () -> "Test requires n >= 16: found 2^" + minP);
        Assertions.assertTrue(maxP <= 30, () -> "Test requires integer n = 2^p: found 2^" + maxP);
        // Limit for the slow computations to avoid long run-time
        final int limit3 = 1000000;
        final int limit4 = 100000;

        final long start = System.currentTimeMillis();
        if (output) {
            // Do a trivial warm-up and use the p-values
            final double x = 0.05;
            final int n = 100;
            final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n, DDExt::simplePowScaled);
            final double p2 = KolmogorovSmirnovDistribution.One.sf(x, n, DD::fastPowScaled);
            final double p3 = KolmogorovSmirnovDistribution.One.sf(x, n, DD::powScaled);
            final double p4 = onesf(x, n, DEFAULT_MC);
            Assertions.assertTrue(p1 < p2 + p3 + p4, "Trivial use of the return values");
            // Add a header to allow to be pasted into a Jira ticket as a table
            printf("||x||n||p||fastPow||pow||Relative||slowPow||Relative||BigDecimal||Relative||%n");
        }

        for (int p = minP; p <= maxP; p++) {
            final int n = 1 << p;
            // Small x where Smirnov-Dwass does not apply (n * x > 4)
            // n * x * x < 372.5 or else p = 0
            final double rn = Math.sqrt(n);
            for (final double x : new double[] {5.0 / n, 7.0 / n, 11.0 / n, 1.0 / rn, 2.0 / rn}) {
                final long t1 = System.nanoTime();
                final double p1 = KolmogorovSmirnovDistribution.One.sf(x, n, DDExt::simplePowScaled);
                final long t2 = System.nanoTime();
                final double p2 = KolmogorovSmirnovDistribution.One.sf(x, n, DD::fastPowScaled);
                final long t3 = System.nanoTime();
                // Avoid really long computation
                final double p3 = n < limit3 ? KolmogorovSmirnovDistribution.One.sf(x, n, DD::powScaled) : 0;
                final long t4 = System.nanoTime();
                final double p4 = n < limit4 ? onesf(x, n, DEFAULT_MC) : 0;
                final long t5 = System.nanoTime();
                final double time1 = (t2 - t1) * 1e-9;
                final double time2 = (t3 - t2) * 1e-9;
                final double time3 = (t4 - t3) * 1e-9;
                final double time4 = (t5 - t4) * 1e-9;
                // Check (the pow computation is the reference).
                // This limit supports the entire test range for p [4, 24].
                TestUtils.assertProbability(p2, p1, 3e-15, () -> String.format("pow vs fastPow: %s %d", x, n));
                if (n < limit3) {
                    TestUtils.assertProbability(p2, p3, 2e-16, () -> String.format("pow vs slowPow: %s %d", x, n));
                }
                if (n < limit4) {
                    TestUtils.assertProbability(p2, p4, 2e-16, () -> String.format("pow vs BigDecimal: %s %d", x, n));
                }
                if (output) {
                    printf("|%12.6g|%10d|%25s|%10.6f|%10.6f|%.3f|%10.6f|%.3f|%10.6f|%.3f|%n",
                        x, n, p2,
                        time1,
                        time2, time2 / time1,
                        time3, time3 / time1,
                        time4, time4 / time1);
                }
            }
        }
        if (output) {
            printf("Finished in %.3f seconds%n", (System.currentTimeMillis() - start) * 1e-3);
        }
    }

    /**
     * Calculates complementary probability {@code P[D_n^+ >= x]}, or survival
     * function (SF), for the one-sided one-sample Kolmogorov-Smirnov distribution.
     *
     * <p>Computes the result using double-double arithmetic. The power function
     * can use a fast approximation or a full power computation.
     *
     * <p>This function is safe for {@code x > 1/n}. When {@code x} approaches
     * sub-normal then division or multiplication by x can under/overflow.
     *
     * <p>This method is here to allow the result of the computation to be tested
     * without the use of the exact SF when {@code x < 1/n} or {@code x > 1 - 1/n}.
     *
     * @param x Statistic (typically in (1/n, 1 - 1/n)).
     * @param n Sample size (assumed to be positive).
     * @param power Function to compute the scaled power (can be null).
     * @return \(P(D_n^+ &ge; x)\)
     * @see DD#fastPowScaled
     * @see DD#powScaled
     */
    private static double onesf(double x, int n, ScaledPower power) {
        // Guard input to allow all test cases.
        if (n * x * x >= 372.5 || x >= 1) {
            // p would underflow, or x is out of the domain
            return 0;
        }
        if (x <= 0) {
            // edge-of, or out-of, the domain
            return 1;
        }
        if (n == 1) {
            return x;
        }
        // Debugging
        final long start = System.nanoTime();
        final double p = KolmogorovSmirnovDistribution.One.sf(x, n, power);
        //printf("\"%s, %d, %s\", // %.6fs%n", x, n, p, (System.nanoTime() - start) * 1e-9);
        return p;
    }

    /**
     * Calculates complementary probability {@code P[D_n^+ >= x]}, or survival
     * function (SF), for the one-sided one-sample Kolmogorov-Smirnov distribution.
     *
     * <p>Computes the result using BigDecimal to the precision specified by the
     * {@link MathContext}.
     *
     * <p>This method is here to allow the result of the computation to be output for
     * spot test cases including the time of the computation.
     *
     * @param x Statistic.
     * @param n Sample size (assumed to be positive).
     * @param mathContext MathContext for precision (can be null).
     * @return \(P(D_n^+ &ge; x)\)
     */
    private static double onesf(double x, int n, MathContext mc) {
        // Guard input to allow all test cases.
        if (n * x * x >= 372.5 || x >= 1) {
            // p would underflow, or x is out of the domain
            return 0;
        }
        if (x <= 0) {
            // edge-of, or out-of, the domain
            return 1;
        }
        if (n == 1) {
            return x;
        }
        // Debugging
        final long start = System.nanoTime();
        final double p = sf(x, n, mc);
        //printf("\"%s, %d, %s\", // %.6fs%n", x, n, p, (System.nanoTime() - start) * 1e-9);
        return p;
    }

    /**
     * Calculates complementary probability {@code P[D_n^+ >= x]}, or survival
     * function (SF), for the one-sided one-sample Kolmogorov-Smirnov distribution.
     *
     * <p>Computes the result using BigDecimal to the precision specified by the
     * {@link MathContext}. This is safe for {@code n < 1e9} which is the limit of BigDecimal
     * power computations.
     *
     * <p>This implementation was developed first closely following the description
     * in van Mulbregt (2018). It has been moved here for reference to allow testing
     * the against the double-double implementation. At large n it is prohibitively slow.
     *
     * @param x Statistic.
     * @param n Sample size (assumed to be positive).
     * @param mathContext MathContext for precision (can be null).
     * @return \(P(D_n^+ &ge; x)\)
     */
    private static double sf(double x, int n, MathContext mathContext) {
        // Compute only the SF using Algorithm 1 pp 12.
        // This omits computation of D for the PDF.

        final BigDecimal bx = bd(x);
        final BigDecimal nbx = bx.negate();

        // Compute x = (k + alpha) / n ; 0 <= alpha < 1
        // Round-off issues with alpha -> 1 are ignored as a zero term for Aj
        // will be computed as zero. Here we only require floor(nx).
        final int k = bx.multiply(bd(n)).round(
            new MathContext(MathContext.DECIMAL128.getPrecision(), RoundingMode.FLOOR)).intValue();

        int maxN;
        BigDecimal a;

        // Eq (13) Smirnov/Birnbaum-Tingey; or Smirnov/Dwass Eq (31)
        // Use the same decision criteria as the double-double implementation.
        boolean sd = false;
        if (k < n - k - 1) {
            sd = k <= 1;
            sd |= k <= 3 && n >= 8;
        }

        // Configure working precision
        // When using BigDecimal add and multiply are correctly rounded.
        // Pow is accurate to 2 ulp of the precision.
        // For SD the terms may cancel so we use a precision at least 2x of a double.
        // For the regular computation we can use 1x a double and maintain guard digits.
        // Since more precision is required for SD only choose it when the number
        // of terms is 1/4 of the regular computation
        MathContext mc;
        if (mathContext != null) {
            mc = mathContext;
        } else {
            // Add some guard digits.
            // Use more for larger n (log10(n+1) is the number of digits).
            final int guardDigits = 5 + (int) Math.ceil(Math.log10(n + 1));
            final int precision = sd ? MathContext.DECIMAL128.getPrecision() :
                MathContext.DECIMAL64.getPrecision();
            mc = new MathContext(precision + guardDigits);
        }

        // Use to determine if term Aj will change the sum. Include guard digits.
        final int sumPrecision = mc.getPrecision() + 2;

        // Compute A0
        if (sd) {
            maxN = k;
            // A0 = (1+x)^(n-1)
            a = BigDecimal.ONE.add(bx).pow(n - 1, mc);
        } else {
            maxN = n - k - 1;
            // A0 = (1-x)^n / x
            a = BigDecimal.ONE.subtract(bx).pow(n).divide(bx, mc);
        }

        // Binomial coefficient c(n, j)
        // This value is integral but maintained to limited precision
        BigDecimal c = BigDecimal.ONE;
        BigDecimal sum = a;
        // Use this to print out progress (which may be very slow when N is large).
        // Report at < 1/128 intervals (or every i if n < 128). Set to -1 to disable.
        final int mask = -1; // (Integer.highestOneBit(maxN) - 1) >>> 7;
        for (int i = 1; i <= maxN; i++) {
            // c(n, j) = c(n, j-1) * (n-j+1) / j
            // Exact multiply then round on divide
            c = c.multiply(bd(n - i + 1)).divide(bd(i), mc);
            // Compute Aj
            final int j = sd ? n - i : i;
            // Algorithm 4 pp. 27
            // S = ((j/n) + x)^(j-1)
            // T = ((n-j)/n - x)^(n-j)
            final BigDecimal p = addDD(div22(j, n, mc), bx, mc);
            final BigDecimal q = addDD(div22(n - j, n, mc), nbx, mc);
            final BigDecimal s = p.pow(j - 1, mc);
            final BigDecimal t = q.pow(n - j, mc);
            a = mulDD(mulDD(c, t, mc), s, mc);
            // Only add to the sum when the value would change the sum.
            // Approximate size of the BigDecimal is (precision - scale).
            // scale is a property but precision must be computed and
            // should be the same as the working precision.
            // So neglect the precision for a first check.
            // Allow print debugging
            if ((a.scale() - sum.scale()) < sumPrecision ||
                (sum.precision() - sum.scale()) - (a.precision() - a.scale()) < sumPrecision) {
                if ((i & mask) == 0) {
                    printf("%s + A[%d] %s%n", sum.doubleValue(), j, a.doubleValue());
                }
                sum = sum.add(a, mc);
            } else {
                // Effectively Aj -> eps * sum, and most of the computation is done.
                if (mask >= 0) {
                    printf("%s + A[%d] %s%n", sum.doubleValue(), j, a.doubleValue());
                }
                break;
            }
        }

        // This ignores the working precision
        // p = sum(Aj) * x
        BigDecimal p = bx.multiply(sum);
        if (sd) {
            // SF = 1 - CDF
            p = BigDecimal.ONE.subtract(p);
        }
        // Clip to [0, 1]
        return Math.max(0, Math.min(1, p.doubleValue()));
    }

    /**
     * Create a BigDecimal for the given value.
     *
     * @param v Value
     * @return the BigDecimal
     */
    private static BigDecimal bd(double v) {
        return new BigDecimal(v);
    }

    /**
     * Create a BigDecimal for the given value.
     *
     * @param v Value
     * @return the BigDecimal
     */
    private static BigDecimal bd(int v) {
        return BigDecimal.valueOf(v);
    }

    /**
     * Returns an approximation of the number {@code a+b}.
     *
     * @param a Addend.
     * @param b Addend.
     * @param mc Math context (for rounding)
     * @return {@code a+b}
     */
    private static BigDecimal addDD(BigDecimal a, BigDecimal b, MathContext mc) {
        return a.add(b, mc);
    }

    /**
     * Returns an approximation of the number {@code a*b}.
     *
     * @param a Factor.
     * @param b Factor.
     * @param mc Math context (for rounding)
     * @return {@code a*b}
     */
    private static BigDecimal mulDD(BigDecimal a, BigDecimal b, MathContext mc) {
        return a.multiply(b, mc);
    }

    /**
     * Returns an approximation of the real number {@code a/b}.
     *
     * @param a Dividend.
     * @param b Divisor.
     * @param mc Math context (for rounding)
     * @return {@code a/b}
     */
    private static BigDecimal divDD(BigDecimal a, BigDecimal b, MathContext mc) {
        return a.divide(b, mc);
    }

    /**
     * Returns an approximation of the real number {@code a/b}.
     *
     * @param a Dividend.
     * @param b Divisor.
     * @param mc Math context (for rounding)
     * @return {@code a/b}
     */
    private static BigDecimal div22(int a, int b, MathContext mc) {
        return divDD(bd(a), bd(b), mc);
    }

    /**
     * Print a formatted message to stdout.
     * Provides a single point to disable checkstyle warnings on print statements.
     *
     * @param format Format string.
     * @param args Arguments.
     */
    private static void printf(String format, Object... args) {
        // CHECKSTYLE: stop regex
        System.out.printf(format, args);
        // CHECKSTYLE: resume regex
    }
}
