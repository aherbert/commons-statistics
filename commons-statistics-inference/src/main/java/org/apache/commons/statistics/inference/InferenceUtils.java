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

import java.util.Arrays;
import java.util.stream.IntStream;
import org.apache.commons.numbers.core.Precision;

/**
 * Utility validation and computation methods.
 *
 * @since 1.1
 */
final class InferenceUtils {
    /** No instances. */
    private InferenceUtils() {}

    /**
     * Check the significance level is in the correct range.
     *
     * @param alpha Significance level of the test.
     * @throws IllegalArgumentException if {@code alpha} is not in the range
     * {@code (0, 0.5]}
     */
    static void checkSignificance(double alpha) {
        if (alpha > 0 && alpha <= 0.5) {
            return;
        }
        // Not in (0, 0.5], or NaN
        throw new InferenceException(InferenceException.INVALID_SIGNIFICANCE, alpha);
    }

    /**
     * Check that the value is {@code >= 0}.
     *
     * @param v Value to be tested.
     * @throws IllegalArgumentException if the value is less than 0.
     */
    static void checkNonNegative(int v) {
        if (v < 0) {
            throw new InferenceException(InferenceException.NEGATIVE, v);
        }
    }

    /**
     * Check that the value is {@code >= 0}.
     *
     * @param v Value to be tested.
     * @throws IllegalArgumentException if the value is less than 0.
     */
    static void checkNonNegative(double v) {
        if (v >= 0) {
            return;
        }
        // Negative, or NaN
        throw new InferenceException(InferenceException.NEGATIVE, v);
    }

    /**
     * Check that all values are {@code >= 0}.
     *
     * @param values Values to be tested.
     * @throws IllegalArgumentException if any values are less than 0.
     */
    static void checkNonNegative(long[] values) {
        for (final long v : values) {
            if (v < 0) {
                throw new InferenceException(InferenceException.NEGATIVE, v);
            }
        }
    }

    /**
     * Check that all values are {@code >= 0}.
     *
     * @param values Values to be tested.
     * @throws IllegalArgumentException if any values are less than 0.
     */
    static void checkNonNegative(long[][] values) {
        for (final long[] v : values) {
            checkNonNegative(v);
        }
    }

    /**
     * Check that all values are {@code > 0}.
     *
     * @param v Value to be tested.
     * @throws IllegalArgumentException if the value is not strictly positive.
     */
    static void checkStrictlyPositive(int v) {
        if (v <= 0) {
            throw new InferenceException(InferenceException.NOT_STRICTLY_POSITIVE, v);
        }
    }

    /**
     * Check that all values are {@code > 0}.
     *
     * @param values Values to be tested.
     * @throws IllegalArgumentException if any values are not strictly positive.
     */
    static void checkStrictlyPositive(double[] values) {
        for (final double v : values) {
            // Logic negation detects NaN
            if (!(v > 0)) {
                throw new InferenceException(InferenceException.NOT_STRICTLY_POSITIVE, v);
            }
        }
    }

    /**
     * Check that all values are not {@link Double#NaN}.
     *
     * @param values Values to be tested.
     * @throws IllegalArgumentException if any values are NaN.
     */
    static void checkNonNaN(double[] values) {
        for (final double v : values) {
            if (Double.isNaN(v)) {
                throw new InferenceException("NaN input value");
            }
        }
    }

    /**
     * Checks if the input array is rectangular. It is assumed the array is non-null
     * and has a non-zero length.
     *
     * @param array Array to be tested.
     * @throws NullPointerException if input array is null
     * @throws IndexOutOfBoundsException if input array is zero length
     * @throws IllegalArgumentException if input array is not rectangular
     */
    static void checkRectangular(long[][] array) {
        final int first = array[0].length;
        for (int i = 1; i < array.length; i++) {
            if (array[i].length != first) {
                throw new InferenceException(InferenceException.NOT_RECTANGULAR, array[i].length, first);
            }
        }
    }

    /**
     * Check the values size is the minimum required, {@code size >= required}.
     *
     * @param size Values size.
     * @param required Required size.
     * @throws IllegalArgumentException if {@code size < required}
     */
    static void checkValuesRequiredSize(int size, int required) {
        if (size < required) {
            throw new InferenceException(InferenceException.VALUES_REQUIRED, size, required);
        }
    }

    /**
     * Check the categories size is the minimum required, {@code size >= required}.
     *
     * @param size Values size.
     * @param required Required size.
     * @throws IllegalArgumentException if {@code size < required}
     */
    static void checkCategoriesRequiredSize(int size, int required) {
        if (size < required) {
            throw new InferenceException(InferenceException.CATEGORIES_REQUIRED, size, required);
        }
    }

    /**
     * Check the values sizes are equal, {@code size1 == size2}.
     *
     * @param size1 First size.
     * @param size2 Second size.
     * @throws IllegalArgumentException if {@code size1 != size2}
     */
    static void checkValuesSizeMatch(int size1, int size2) {
        if (size1 != size2) {
            throw new InferenceException(InferenceException.VALUES_MISMATCH, size1, size2);
        }
    }

    /**
     * Gets the ratio between the sum of the observed and expected values.
     * The ratio can be used to scale the expected values to have the same sum
     * as the observed values:
     *
     * <pre>
     * sum(o) = sum(e * ratio)
     * </pre>
     *
     * <p>This method is common functionality shared between the Chi-square test and
     * G-test. The pre-conditions for those tests are performed by this method.
     *
     * @param expected Expected values.
     * @param observed Observed values.
     * @return the ratio
     * @throws IllegalArgumentException if the sample size is less than 2; the array
     * sizes do not match; {@code expected} has entries that are not strictly
     * positive; or {@code observed} has negative entries.
     */
    static double computeRatio(double[] expected, long[] observed) {
        checkValuesRequiredSize(expected.length, 2);
        checkValuesSizeMatch(expected.length, observed.length);
        checkStrictlyPositive(expected);
        checkNonNegative(observed);
        final DD e = DD.create();
        final DD o = DD.create();
        for (int i = 0; i < observed.length; i++) {
            DD.fastAdd(e.hi(), e.lo(), expected[i], e);
            DD.fastAdd(o.hi(), o.lo(), observed[i], o);
        }
        // sum(o) / sum(e)
        final double ratio = DD.divide(o.hi(), o.lo(), e.hi(), e.lo(), e).doubleValue();
        // Allow a sum within 1 ulp of 1.0
        return Precision.equals(ratio, 1.0, 0) ? 1.0 : ratio;
    }

    /**
     * Returns the arithmetic mean of the entries in the specified portion of the input
     * array, or {@code NaN} if the designated subarray is empty.
     *
     * <p>A two-pass, corrected algorithm is used, starting with the definitional formula
     * computed using the array of stored values and then correcting this by adding the
     * mean deviation of the data values from the arithmetic mean. See, e.g. "Comparison
     * of Several Algorithms for Computing Sample Means and Variances," Robert F. Ling,
     * Journal of the American Statistical Association, Vol. 69, No. 348 (Dec., 1974), pp.
     * 859-866.
     *
     * @param sample1 the input array
     * @return the mean of the values or NaN if length = 0
     */
    static double mean(double[] sample1) {
        final int n = sample1.length;
        // No check for n == 0 -> return NaN.
        // This internal method is only called with non-zero length arrays.
        // The divide by zero creates NaN anyway.

        // Adapted from org.apache.commons.math4.legacy.stat.descriptive.moment.Mean
        // Updated to use a stream to support high-precision summation as the stream maintains
        // a rounding-error term during the aggregation. This is important
        // when summing differences which can create cancellation: x + -x => 0.

        // Compute initial estimate using definitional formula
        final double xbar = Arrays.stream(sample1).sum() / n;

        // Compute correction factor in second pass
        return xbar + Arrays.stream(sample1).map(x -> x - xbar).sum() / n;
    }

    /**
     * Returns the variance of the entries in the input array, or {@code NaN} if the array
     * is empty.
     *
     * <p>This method returns the bias-corrected sample variance (using {@code n - 1} in
     * the denominator).
     *
     * <p>Uses a two-pass algorithm. Specifically, these methods use the "corrected
     * two-pass algorithm" from Chan, Golub, Levesque, <i>Algorithms for Computing the
     * Sample Variance</i>, American Statistician, vol. 37, no. 3 (1983) pp.
     * 242-247.
     *
     * <p>Returns 0 for a single-value (i.e. length = 1) sample.
     *
     * @param sample1 the input array
     * @param mean the mean of the input array
     * @return the variance of the values or NaN if the array is empty
     */
    static double variance(double[] sample1, double mean) {
        final int n = sample1.length;
        // No check for n == 0 -> return NaN.
        // This internal method is only called with non-zero length arrays.
        // The input mean of NaN for zero length creates NaN anyway.
        if (n == 1) {
            return 0;
        }

        // Adapted from org.apache.commons.math4.legacy.stat.descriptive.moment.Variance
        // Use a stream to accumulate the sum of deviations in high precision.
        // This compensation term for the sum of deviations from the mean -> 0.
        // We sum the squares in standard precision as there is no cancellation of summands.
        final double[] sumSq = {0};
        final double sum2 = Arrays.stream(sample1).map(x -> {
            final double dx = x - mean;
            sumSq[0] += dx * dx;
            return dx;
        }).sum();

        final double sum1 = sumSq[0];
        // Bias corrected
        // Note: variance ~ sum1 / (n-1) but with a correction term sum2
        return (sum1 - (sum2 * sum2 / n)) / (n - 1);
    }

    /**
     * Returns the mean of the (signed) differences between corresponding elements of the
     * input arrays.
     *
     * <pre>
     * sum(sample1[i] - sample2[i]) / sample1.length
     * </pre>
     *
     * <p>This computes the same result as creating an array {@code x = sample1 - sample2}
     * and calling {@link #mean(double[]) mean(x)}, but without the intermediate array
     * allocation.
     *
     * @param sample1 the first array
     * @param sample2 the second array
     * @return mean of paired differences
     * @throws IllegalArgumentException if the arrays do not have the same length.
     * @see #mean(double[])
     */
    static double meanDifference(double[] sample1, double[] sample2) {
        final int n = sample1.length;
        if (n != sample2.length) {
            throw new InferenceException(InferenceException.VALUES_MISMATCH, n, sample2.length);
        }
        // See mean(double[]) for details.
        final double xbar = IntStream.range(0, n).mapToDouble(i -> sample1[i] - sample2[i]).sum() / n;
        return xbar + IntStream.range(0, n).mapToDouble(i -> (sample1[i] - sample2[i]) - xbar).sum() / n;
    }

    /**
     * Returns the variance of the (signed) differences between corresponding elements of
     * the input arrays.
     *
     * <pre>
     * var(sample1[i] - sample2[i])
     * </pre>
     *
     * <p>This computes the same result as creating an array {@code x = sample1 - sample2}
     * and calling {@link #variance(double[], double) variance(x, mean(x))}, but without the
     * intermediate array allocation.
     *
     * @param sample1 the first array
     * @param sample2 the second array
     * @param mean the mean difference between corresponding entries
     * @return variance of paired differences
     * @throws IllegalArgumentException if the arrays do not have the same length.
     * @see #meanDifference(double[], double[])
     * @see #variance(double[], double)
     */
    static double varianceDifference(double[] sample1, double[] sample2, double mean) {
        final int n = sample1.length;
        if (n != sample2.length) {
            throw new InferenceException(InferenceException.VALUES_MISMATCH, n, sample2.length);
        }
        // See variance(double[]) for details.
        if (n == 1) {
            return 0;
        }
        final double[] sumSq = {0};
        final double sum2 = IntStream.range(0, n).mapToDouble(i -> {
            final double dx = (sample1[i] - sample2[i]) - mean;
            sumSq[0] += dx * dx;
            return dx;
        }).sum();
        final double sum1 = sumSq[0];
        return (sum1 - (sum2 * sum2 / n)) / (n - 1);
    }
}
