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
 * Represents a difference.
 *
 * @since 1.1
 */
public final class Difference {
    /** Zero value. */
    static final Difference ZERO = new Difference(0);

    /** The value. */
    private final double value;

    /**
     * @param value Value.
     */
    private Difference(double value) {
        this.value = value;
    }

    /**
     * Create an instance.
     *
     * @param value Value.
     * @return an instance
     */
    public static Difference of(double value) {
        // Potentially this could be non-finite. However testing
        // for non-finite values will not detect errors from using MAX_VALUE.
        // It is left to the caller to detect their own input error.
        return new Difference(value);
    }

    /**
     * Return the value as a {@code double}.
     *
     * @return the value
     */
    public double doubleValue() {
        return value;
    }
}
