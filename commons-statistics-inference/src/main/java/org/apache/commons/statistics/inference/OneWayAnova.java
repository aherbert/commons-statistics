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

import java.util.ArrayList;
import java.util.Collection;
import org.apache.commons.statistics.distribution.FDistribution;

/**
 * Implements one-way ANOVA (analysis of variance) statistics.
 *
 * <p>Tests for differences between two or more categories of univariate data
 * (for example, the body mass index of accountants, lawyers, doctors and
 * computer programmers). When two categories are given, this is equivalent to
 * the {@link TTest}.
 *
 * <p>This implementation computes the F statistic using the definitional formula:
 *
 * <p>\[ F = \frac{\text{between-group variability}}{\text{within-group variability}} \]
 *
 * @see <a href="https://en.wikipedia.org/wiki/Analysis_of_variance">Analysis of variance (Wikipedia)</a>
 * @see <a href="https://en.wikipedia.org/wiki/F-test#Multiple-comparison_ANOVA_problems">
 * Multiple-comparison ANOVA problems (Wikipedia)</a>
 * @since 1.1
 */
public class OneWayAnova {
    /**
     * Summary statistics for a stream of data values.
     *
     * @since 1.1
     */
    public interface AnovaSummaryStatistics {
        /**
         * Returns the sum of the values that have been added.
         *
         * @return the sum of the values
         */
        double getSum();

        /**
         * Returns the sum of the squares of the values that have been added.
         *
         * @return the sum of squares
         */
        double getSumOfSquares();

        /**
         * Returns the number of values.
         *
         * @return the number of values
         */
        int getN();

        /**
         * Creates a summary of the specified {@code values}.
         * The sum is created by addition; the sum of squares by addition of the squared values.
         *
         * <p>If {@code values} is empty then the sum and sum of squares will be zero.
         *
         * @param values Values.
         * @return the summary statistics
         * @throws NullPointerException if {@code values} is null
         */
        static AnovaSummaryStatistics of(double... values) {
            // Ensure not null or generate implicit NPE
            final int n = values.length;
            double s = 0;
            double ss = 0;
            for (final double v : values) {
                s += v;
                ss += v * v;
            }
            final double sum = s;
            final double sumOfSquares = ss;
            return new AnovaSummaryStatistics() {
                @Override
                public double getSum() {
                    return sum;
                }

                @Override
                public double getSumOfSquares() {
                    return sumOfSquares;
                }

                @Override
                public int getN() {
                    return n;
                }
            };
        }
    }

    /**
     * Computes the ANOVA F-value for a collection of category samples.
     *
     * @param categoryData Category samples.
     * @return F statistic
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    public double anovaFValue(Collection<double[]> categoryData) {
        return computeAnovaStats(categoryData).getF();
    }

    /**
     * Computes the ANOVA p-value for a collection of category samples.
     *
     * @param categoryData Category samples.
     * @return p-value
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    public double anovaPValue(Collection<double[]> categoryData) {
        final AnovaStats a = computeAnovaStats(categoryData);
        return FDistribution.of(a.getDfbg(), a.getDfwg()).survivalProbability(a.getF());
    }

    /**
     * Performs an ANOVA test for a collection of category samples, evaluating the
     * null hypothesis that there is no difference among the means of the data
     * categories.
     *
     * @param categoryData Category data.
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}.
     * @throws IllegalArgumentException if the number of categories is less than
     * two; a contained category does not have at least two values; or {@code alpha}
     * is not in the range {@code (0, 0.5]}
     */
    public boolean anovaTest(Collection<double[]> categoryData,
                             double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return anovaPValue(categoryData) < alpha;
    }

    /**
     * Computes the ANOVA F-value for a collection of category summary data.
     *
     * @param categoryData Category summary data.
     * @return F statistic
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    public double anovaFValue1(Collection<AnovaSummaryStatistics> categoryData) {
        return createAnovaStats(categoryData).getF();
    }

    /**
     * Computes the ANOVA p-value for a collection of category summary data.
     *
     * @param categoryData Category summary data.
     * @return p-value
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    public double anovaPValue1(Collection<AnovaSummaryStatistics> categoryData) {
        final AnovaStats a = createAnovaStats(categoryData);
        return FDistribution.of(a.getDfbg(), a.getDfwg()).survivalProbability(a.getF());
    }

    /**
     * Performs an ANOVA test for a collection of category summary data, evaluating the
     * null hypothesis that there is no difference among the means of the data
     * categories.
     *
     * @param categoryData Category summary data.
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with confidence
     * {@code 1 - alpha}.
     * @throws IllegalArgumentException if the number of categories is less than
     * two; a contained category does not have at least two values; or {@code alpha}
     * is not in the range {@code (0, 0.5]}
     */
    public boolean anovaTest1(Collection<AnovaSummaryStatistics> categoryData,
                              double alpha) {
        InferenceUtils.checkSignificance(alpha);
        return anovaPValue1(categoryData) < alpha;
    }

    /**
     * Compute the ANOVA statistics.
     *
     * @param categoryData Category data.
     * @return ANOVA statistics
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    private static AnovaStats computeAnovaStats(Collection<double[]> categoryData) {
        final Collection<AnovaSummaryStatistics> stats = new ArrayList<>(categoryData.size());
        categoryData.forEach(data -> stats.add(AnovaSummaryStatistics.of(data)));
        return createAnovaStats(stats);
    }

    /**
     * Compute the ANOVA statistics.
     *
     * @param categoryData Category data.
     * @return ANOVA statistics
     * @throws IllegalArgumentException if the number of categories is less than two; or a
     * contained category does not have at least two values
     */
    private static AnovaStats createAnovaStats(Collection<AnovaSummaryStatistics> categoryData) {
        // check if we have enough categories
        InferenceUtils.checkCategoriesRequiredSize(categoryData.size(), 2);
        // check if each category has enough data
        for (final AnovaSummaryStatistics array : categoryData) {
            InferenceUtils.checkValuesRequiredSize(array.getN(), 2);
        }

        long dfwg = 0;
        double sswg = 0;
        double totsum = 0;
        double totsumsq = 0;
        long totnum = 0;

        for (final AnovaSummaryStatistics data : categoryData) {
            final double sum = data.getSum();
            final double sumsq = data.getSumOfSquares();
            final int num = data.getN();
            totnum += num;
            totsum += sum;
            totsumsq += sumsq;

            dfwg += num - 1;
            final double ss = sumsq - ((sum * sum) / num);
            sswg += ss;
        }

        final double sst = totsumsq - ((totsum * totsum) / totnum);
        final double ssbg = sst - sswg;
        final int dfbg = categoryData.size() - 1;
        final double msbg = ssbg / dfbg;
        final double mswg = sswg / dfwg;
        final double f = msbg / mswg;

        return new AnovaStats(dfbg, dfwg, f);
    }

    /**
     * Convenience class to pass (dfbg, dfwg, F) values around within {@link OneWayAnova}.
     */
    private static final class AnovaStats {
        /** Degrees of freedom in numerator (between groups). */
        private final int dfbg;
        /** Degrees of freedom in denominator (within groups). */
        private final long dfwg;
        /** Statistic. */
        private final double f;

        /**
         * @param dfbg degrees of freedom in numerator (between groups)
         * @param dfwg degrees of freedom in denominator (within groups)
         * @param f statistic
         */
        AnovaStats(int dfbg, long dfwg, double f) {
            this.dfbg = dfbg;
            this.dfwg = dfwg;
            this.f = f;
        }

        /**
         * Gets the degrees of freedom in numerator (between groups).
         *
         * @return dfbg
         */
        int getDfbg() {
            return dfbg;
        }

        /**
         * Gets the degrees of freedom in denominator (within groups).
         *
         * @return dfwg
         */
        long getDfwg() {
            return dfwg;
        }

        /**
         * Gets the F statistic.
         *
         * @return F
         */
        double getF() {
            return f;
        }
    }
}
