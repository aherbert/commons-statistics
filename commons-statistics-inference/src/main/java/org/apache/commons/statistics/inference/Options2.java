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
 * Extracts options from generic arguments.
 *
 * @since 1.1
 * @see java.util.Optional
 */
final class Options2 {
    /** No instances. */
    private Options2() {}

    /**
     * Gets the first object that is assignment-compatible with the object
     * represented by the specified {@code option}, or {@code option} if the options
     * contains no instance that can be cast to the reference type of the specified
     * {@code option}.
     *
     * <p>This method does not specify {@code options} using varargs as without any
     * options the result is the input {@code option}.
     *
     * @param <T> Type of the option.
     * @param option Option.
     * @param options Array of possible options.
     * @return the first assignable instance, or {@code option}
     * @see Class#isInstance(Object)
     */
    @SuppressWarnings("unchecked")
    static <T> T orElse(T option, Object[] options) {
        final Class<?> c = option.getClass();
        for (final Object o : options) {
            if (c.isInstance(o)) {
                return (T) o;
            }
        }
        return option;
    }

    /**
     * If the {@code option} is present is the {@code options}, returns {@code true},
     * otherwise {@code false}.
     *
     * @param <T> Type of the option.
     * @param option Option.
     * @param options Array of possible options.
     * @return {@code true} if a value is present, otherwise {@code false}
     */
    static <T> boolean isPresent(T option, Object[] options) {
        for (final Object o : options) {
            if (option.equals(o)) {
                return true;
            }
        }
        return false;
    }
}
