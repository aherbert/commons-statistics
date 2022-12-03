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

/**
 * Computations for the Kolmogorov-Smirnov distribution.
 *
 * <p>This class contains extension methods to supplement the functionality in
 * {@link KolmogorovSmirnovDistribution} for use in computing p-values for
 * the Kolmogorov-Smirnov test.
 *
 * <p>The methods are tested in {@link KolmogorovSmirnovTestTest} to ensure validity.
 *
 * @since 1.1
 */
final class KolmogorovSmirnovDistributionExt {
    /** pi^2. */
    private static final double PI2 = 9.8696044010893586188344909;
    /** sqrt(2*pi). */
    private static final double ROOT_TWO_PI = 2.5066282746310005024157652;
    /** Factor 4a in the quadratic equation to solve max k: log(2^-52) * 8. */
    private static final double FOUR_A = -288.3492271129372;
    /** Log(epsilon), ln(2^-52). */
    private static final double LN_EPS = -36.04365338911715;

    /**
     * No instances.
     */
    private KolmogorovSmirnovDistributionExt() {}

    /**
     * Computes {@code P(sqrt(n) D_n > x)}, the limiting form for the distribution of
     * Kolmolgorov's D_n as described in Simard and L’Ecuyer (2011).
     *
     * <p>Computes \( 2 \sum_{i=1}^\infty (-1)^(i-1) e^{-2 i^2 x^2} \), or
     * \( 1 - (\sqrt{2 \pi} / x) * \sum_{i=1}^\infty { e^{-(2i-1)^2 \pi^2 / (8x^2) } } \)
     * when x is small.
     *
     * <p>Note: This computes the upper Kolmogorov sum.
     *
     * @param x2 Argument x^2 (n * d * d; x = sqrt(n) * d)
     * @return Upper Kolmogorov sum evaluated at x
     */
    static double ksSum(double x2) {
        if (x2 == 0) {
            return 1;
        }

        final double x = -2 * x2;
        double sum = 0;

        // When x -> 0 then exp(x i^2) -> 1 and alternating summation has cancellation.
        // Switch computation for small x.
        if (x > -1) {
            final double f = PI2 / (4 * x);
            // Iterate j=(2i - 1) for i=1, 2, ...
            // Terms reduce in size. Stop when:
            // exp(-pi^2 / 8t^2) * eps = exp((2i-1)^2 * -pi^2 / 8t^2)
            // (2i-1)^2 = 1 - log(eps) * 8t^2 / pi^2
            // 0 = i^2 - i + log(eps) * 2t^2 / pi^2
            // Solve using quadratic equation and eps = ulp(1.0): 4a ~ -288
            final int max = (int) Math.ceil((1 + Math.sqrt(1 - FOUR_A * x2 / PI2)) / 2);

            // Sum smallest terms first
            for (int i = max; i > 0; i--) {
                final int j = 2 * i - 1;
                final double delta = Math.exp(f * j * j);
                sum += delta;
            }
            sum *= ROOT_TWO_PI / Math.sqrt(x2);
            return clipProbability(1 - sum);
        }

        // Sum of alternating terms of reducing magnitude.
        // Will converge when exp(x) * eps >= exp(x)^(i^2)
        // i = sqrt( (x + ln(eps)) / x )
        final int max = (int) Math.ceil(Math.sqrt((x + LN_EPS) / x));

        // Sum ascending magnitudes. The sign is positive for odd i.
        int sign = (max & 1) == 1 ? 1 : -1;
        for (int i = max; i > 0; i--) {
            sum += sign * Math.exp(x * i * i);
            sign *= -1;
        }
        return clipProbability(sum * 2);
    }

    /**
     * Clip the probability to the range [0, 1].
     *
     * @param p Probability.
     * @return p in [0, 1]
     */
    private static double clipProbability(double p) {
        return Math.min(1, Math.max(0, p));
    }
}
