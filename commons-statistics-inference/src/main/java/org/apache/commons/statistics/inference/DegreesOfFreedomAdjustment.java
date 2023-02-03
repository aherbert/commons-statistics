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
 * Represents an adjustment to the degrees of freedom.
 *
 * <p>Example:
 *
 * <p>The default degrees of freedom for a sample of length {@code n} are
 * {@code n - 1}. An intrinsic null hypothesis is one where you estimate one or
 * more parameters from the data in order to produce the numbers for your null
 * hypothesis. For a distribution with {@code p} parameters where up to
 * {@code p} parameters have been estimated from the data the degrees of freedom
 * is in the range {@code [n - 1 - p, n - 1]}.
 *
 * @since 1.1
 */
public final class DegreesOfFreedomAdjustment {
    /** Zero value. */
    static final DegreesOfFreedomAdjustment ZERO = new DegreesOfFreedomAdjustment(0);

    /** The value. */
    private final int value;

    /**
     * @param value Value.
     */
    private DegreesOfFreedomAdjustment(int value) {
        this.value = value;
    }

    /**
     * Create an instance.
     *
     * @param value Value.
     * @return an instance
     * @throws IllegalArgumentException if the value is negative
     */
    public static DegreesOfFreedomAdjustment of(int value) {
        InferenceUtils.checkNonNegative(value);
        return new DegreesOfFreedomAdjustment(value);
    }

    /**
     * Return the value as an {@code int}.
     *
     * @return the value
     */
    public int intValue() {
        return value;
    }
}
