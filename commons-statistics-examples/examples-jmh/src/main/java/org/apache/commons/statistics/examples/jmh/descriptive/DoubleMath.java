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

/**
 * Support class for double math.
 *
 * @since 1.1
 */
final class DoubleMath {
    /** No instances. */
    private DoubleMath() {}

    /**
     * Return {@code true} if {@code x > y}.
     *
     * <p>Respects the sort ordering of {@link Double#compare(double, double)}:
     *
     * <pre>{@code
     * Double.compare(x, y) > 0
     * }</pre>
     *
     * @param x Value.
     * @param y Value.
     * @return {@code x > y}
     */
    static boolean greaterThan(double x, double y) {
        if (x > y) {
            return true;
        }
        if (x < y) {
            return false;
        }
        // Equal numbers; signed zeros (-0.0, 0.0); or NaNs
        final long a = Double.doubleToLongBits(x);
        final long b = Double.doubleToLongBits(y);
        return a > b;
    }

    /**
     * Return {@code true} if {@code x < y}.
     *
     * <p>Respects the sort ordering of {@link Double#compare(double, double)}:
     *
     * <pre>{@code
     * Double.compare(x, y) < 0
     * }</pre>
     *
     * @param x Value.
     * @param y Value.
     * @return {@code x < y}
     */
    static boolean lessThan(double x, double y) {
        if (x < y) {
            return true;
        }
        if (x > y) {
            return false;
        }
        // Equal numbers; signed zeros (-0.0, 0.0); or NaNs
        final long a = Double.doubleToLongBits(x);
        final long b = Double.doubleToLongBits(y);
        return a < b;
    }

    /**
     * Compute the arithmetic mean of the two values taking care to avoid overflow.
     *
     * @param x Value.
     * @param y Value.
     * @return the mean
     */
    static double mean(double x, double y) {
        final double sum = x + y;
        return Double.isFinite(sum) ? sum * 0.5 : x * 0.5 + y * 0.5;
    }

    /**
     * Linear interpolation between {@code x} and {@code y} using the fraction {@code alpha}
     * taking care to avoid overflow.
     * <pre>
     * value = x + alpha * (y - x)
     * </pre>
     *
     * @param x Value.
     * @param y Value.
     * @param alpha Alpha.
     * @return the value
     */
    static double interpolate(double x, double y, double alpha) {
        final double v = x + alpha * (y - x);
        if (Double.isFinite(v)) {
            return v;
        }
        // Overflow safe difference (y-x)
        return 2 * (x * 0.5 + alpha * (y * 0.5 - x * 0.5));
    }
}
