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

import java.util.Objects;

/**
 * Options for the inference tests.
 *
 * <p>This class is immutable.
 */
public class Options {
    /** Default options. */
    private static final Options DEFAULT_OPTIONS = new Options();

    /** Alternative hypothesis. */
    private final AlternativeHypothesis alternative;
    /** Assume the two samples have the same population variance. */
    private final boolean equalVariances;
    /** The true value of the mean (or difference in means for a two sample test). */
    private final double mu;
    /** Adjustment for the degrees of freedom. */
    private final int adjust;
    /** Method to compute the p-value. */
    private final PValueMethod pValue;
    /** Use a strict inequality for the two-sample exact p-value. */
    private final boolean strictInequality;
    /** Perform continuity correction. */
    private final boolean continuityCorrection;

    /**
     * Builder for the {@link Options}.
     */
    public static class Builder {
        /** Alternative hypothesis. */
        private AlternativeHypothesis alternative;
        /** Assume the two samples have the same population variance. */
        private boolean equalVariances;
        /** The true value of the mean (or difference in means for a two sample test). */
        private double mu;
        /** Adjustment for the degrees of freedom. */
        private int adjust;
        /** Method to compute the p-value. */
        private PValueMethod pValue;
        /** Use a strict inequality for the two-sample exact p-value. */
        private boolean strictInequality;
        /** Perform continuity correction. */
        private boolean continuityCorrection;

        /**
         * @param source Source to copy.
         */
        Builder(Options source) {
            alternative = source.alternative;
            equalVariances = source.equalVariances;
            mu = source.mu;
            adjust = source.adjust;
            pValue = source.pValue;
            strictInequality = source.strictInequality;
            continuityCorrection = source.continuityCorrection;
        }

        /**
         * Sets the alternative hypothesis.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#getAlternative()
         */
        public Builder setAlternative(AlternativeHypothesis v) {
            this.alternative = Objects.requireNonNull(v);
            return this;
        }

        /**
         * Set the assumption of equal variances.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#isEqualVariances()
         */
        public Builder setEqualVariances(boolean v) {
            equalVariances = v;
            return this;
        }

        /**
         * Set the expected value of the mean (or difference in means for a two-sample test).
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#getMu()
         */
        public Builder setMu(double v) {
            mu = v;
            return this;
        }

        /**
         * Sets the adjustment to the degrees of freedom.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @throws IllegalArgumentException if the adjustment is negative.
         * @see Options#getDegreesOfFreedomAdjustment()
         */
        public Builder setDegreesOfFreedomAdjustment(int v) {
            InferenceUtils.checkNonNegative(v);
            this.adjust = v;
            return this;
        }

        /**
         * Sets the method to compute the p-value.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#getPValueMethod()
         */
        public Builder setPValueMethod(PValueMethod v) {
            this.pValue = Objects.requireNonNull(v);
            return this;
        }

        /**
         * Set to {@code true} to compute the two-sample exact p-value using a strict inquality.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#isStrictInequality()
         */
        public Builder setStrictInequality(boolean v) {
            this.strictInequality = v;
            return this;
        }

        /**
         * Set to {@code true} to use a continuity correction for the approximate
         * p-value computation.
         *
         * @param v Value.
         * @return a reference to {@code this}
         * @see Options#isContinuityCorrection()
         */
        public Builder setContinuityCorrection(boolean v) {
            continuityCorrection = v;
            return this;
        }

        /**
         * Builds the options.
         *
         * @return the options
         */
        Options build() {
            return new Options(this);
        }
    }

    /**
     * Create the default options.
     */
    Options() {
        alternative = AlternativeHypothesis.TWO_SIDED;
        equalVariances = false;
        mu = 0;
        adjust = 0;
        pValue = PValueMethod.AUTO;
        strictInequality = false;
        continuityCorrection = true;
    }

    /**
     * @param source Source to copy.
     */
    Options(Builder source) {
        alternative = source.alternative;
        equalVariances = source.equalVariances;
        mu = source.mu;
        adjust = source.adjust;
        pValue = source.pValue;
        strictInequality = source.strictInequality;
        continuityCorrection = source.continuityCorrection;
    }

    /**
     * Return the default options.
     *
     * <ul>
     * <li>{@link #getAlternative getAlternative = two-sided}
     * <li>{@link #isEqualVariances() isEqualVariances = false}
     * <li>{@link #getMu() getMu = 0}
     * <li>{@link #getDegreesOfFreedomAdjustment getDegreesOfFreedomAdjustment = 0}
     * <li>{@link #getPValueMethod() getPValueMethod = auto}
     * <li>{@link #isStrictInequality() isStrictInequality = false}
     * <li>{@link #isContinuityCorrection() isContinuityCorrection = true}
     * </ul>
     *
     * @return the options
     */
    public static Options defaults() {
        return DEFAULT_OPTIONS;
    }

    /**
     * Create a new {@link Builder} with the default options.
     *
     * @return the builder
     */
    public static Builder builder() {
        return DEFAULT_OPTIONS.toBuilder();
    }

    /**
     * Create a {@link Builder} from the current options.
     *
     * @return the builder
     */
    public Builder toBuilder() {
        return new Builder(this);
    }

    /**
     * Return the alternative hypothesis.
     *
     * @return the alternative hypothesis
     */
    public AlternativeHypothesis getAlternative() {
        return alternative;
    }

    /**
     * If {@code true}, perform the independent t-test under the assumption of equal
     * sub-population variances (homoscedastic t-test).
     *
     * <p>If {@code false}, perform the independent t-test without the assumption of equal
     * sub-population variances (heteroscedastic t-test).
     *
     * <p>Applies to {@link TTest#test(double[], double[], Options)}.
     *
     * @return true the variance are equal
     */
    public boolean isEqualVariances() {
        return equalVariances;
    }

    /**
     * Return the expected value of the mean (or difference in means for a two-sample test).
     *
     * @return the expected mean
     */
    public double getMu() {
        return mu;
    }

    /**
     * Return the adjustment to the degrees of freedom.
     *
     * <p>The default degrees of freedom for a sample of length {@code n} are
     * {@code n - 1}. An intrinsic null hypothesis is one where you estimate one or
     * more parameters from the data in order to get the numbers for your null
     * hypothesis. For a distribution with {@code p} parameters where up to
     * {@code p} parameters have been estimated from the data the degrees of freedom
     * is in the range {@code [n - 1 - p, n - 1]}.
     *
     * @return the adjustment
     */
    public int getDegreesOfFreedomAdjustment() {
        return adjust;
    }

    /**
     * Gets the method to compute the p-value.
     *
     * <p>For the two-sided test the exact p-value is only valid if there are no matching
     * samples {@code x[i] == y[j]}; otherwise the p-value resorts to the asymptotic
     * approximation.
     *
     * @return the p-value method
     */
    public PValueMethod getPValueMethod() {
        return pValue;
    }

    /**
     * Compute the p-value for the two-sample test as
     * \(P(D_{n,m} &gt; d)\) if {@code true}; otherwise \(P(D_{n,m} \ge d)\),
     * where \(D_{n,m}\) is the 2-sample Kolmogorov-Smirnov statistic, either the two-sided
     * \(D_{n,m}\) or one-sided \(D_{n,m}^+\}.
     *
     * <p>Applies to {@link KolmogorovSmirnovTest#test(double[], double[], Options)}.
     *
     * @return true to use a strict inequality
     */
    public boolean isStrictInequality() {
        return strictInequality;
    }

    /**
     * If {@code true}, adjust the statistic by 0.5 towards the
     * mean value when computing the z-statistic if a normal approximation is used
     * to compute the p-value.
     *
     * @return true to perform continuity correction
     */
    public boolean isContinuityCorrection() {
        return continuityCorrection;
    }
}
