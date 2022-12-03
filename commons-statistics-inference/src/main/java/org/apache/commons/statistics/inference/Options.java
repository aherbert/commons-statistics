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
 * Defines the options for a test for significance.
 *
 * <p>The options is a container for instances of classes that define behaviour
 * of a significance test. It provides type-safe conversion of a set of objects
 * to objects of a specific type.
 *
 * <p>Instances of {@link Options} are immutable.
 *
 * @since 1.1
 */
public abstract class Options {
    /**
     * Gets the first option that is assignment-compatible with the object represented
     * by the specified {@code option}, or {@code null} if the options contains no
     * instance that can be cast to the reference type of the specified
     * {@code option}.
     *
     * @param <T> Type of the option.
     * @param option Option.
     * @return the first assignable instance, or null
     * @see Class#isInstance(Object)
     */
    public abstract <T> T instanceOf(T option);

    /**
     * Gets the first option that is assignment-compatible with the object
     * represented by the specified {@code option}, or {@code option} if the options
     * contains no instance that can be cast to the reference type of the specified
     * {@code option}.
     *
     * @param <T> Type of the option.
     * @param option Option.
     * @return the first assignable instance, or {@code option}
     * @see Class#isInstance(Object)
     */
    public <T> T orElse(T option) {
        final T v = instanceOf(option);
        return v != null ? v : option;
    }

    /**
     * Returns an empty options.
     *
     * @return the options
     */
    public static Options none() {
        return Options0.INSTANCE;
    }

    /**
     * Return options containing the specified option instance.
     *
     * @param o Object.
     * @return the options
     * @throws NullPointerException if the option instance is null
     */
    public static Options of(Object o) {
        return new Options1(o);
    }

    /**
     * Return options containing the specified option instances.
     *
     * @param o1 First option.
     * @param o2 Second option.
     * @return the options
     * @throws NullPointerException if any of the option instances are null
     */
    public static Options of(Object o1, Object o2) {
        return new Options2(o1, o2);
    }

    /**
     * Return options containing the specified option instances.
     *
     * @param o1 First option.
     * @param o2 Second option.
     * @param o3 Third option.
     * @return the options
     * @throws NullPointerException if any of the option instances are null
     */
    public static Options of(Object o1, Object o2, Object o3) {
        return new Options3(o1, o2, o3);
    }

    /**
     * Return options containing the specified option instances.
     *
     * @param o1 First option.
     * @param o2 Second option.
     * @param o3 Third option.
     * @param o4 Fourth option.
     * @return the options
     * @throws NullPointerException if any of the option instances are null
     */
    public static Options of(Object o1, Object o2, Object o3, Object o4) {
        return new Options4(o1, o2, o3, o4);
    }

    /**
     * Return options containing the specified option instances.
     *
     * @param o Array of options.
     * @return the options
     * @throws NullPointerException if any of the option instances are null
     */
    public static Options of(Object... o) {
        return new OptionsN(o);
    }

    /**
     * Options implementation with 0 entries.
     */
    private static class Options0 extends Options {
        /** Singleton instance. */
        static final Options INSTANCE = new Options0();

        @Override
        public <T> T instanceOf(T option) {
            return null;
        }

        @Override
        public <T> T orElse(T option) {
            return Objects.requireNonNull(option);
        }
    }

    /**
     * Options implementation with 1 entry.
     */
    private static class Options1 extends Options {
        /** First option. */
        private final Object o1;

        /**
         * @param o1 First option.
         */
        Options1(Object o1) {
            this.o1 = Objects.requireNonNull(o1);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T instanceOf(T option) {
            return option.getClass().isInstance(o1) ? (T) o1 : null;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T orElse(T option) {
            return option.getClass().isInstance(o1) ? (T) o1 : option;
        }
    }

    /**
     * Options implementation with 2 entries.
     */
    private static class Options2 extends Options {
        /** First option. */
        private final Object o1;
        /** Second option. */
        private final Object o2;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         */
        Options2(Object o1, Object o2) {
            this.o1 = o1;
            this.o2 = o2;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T instanceOf(T option) {
            final Class<?> c = option.getClass();
            if (c.isInstance(o1)) {
                return (T) o1;
            }
            return c.isInstance(o2) ? (T) o2 : null;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T orElse(T option) {
            final Class<?> c = option.getClass();
            if (c.isInstance(o1)) {
                return (T) o1;
            }
            return c.isInstance(o2) ? (T) o2 : option;
        }
    }

    /**
     * Options implementation with 3 entries.
     */
    private static class Options3 extends Options {
        /** First option. */
        private final Object o1;
        /** Second option. */
        private final Object o2;
        /** Third option. */
        private final Object o3;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         * @param o3 Third option.
         */
        Options3(Object o1, Object o2, Object o3) {
            this.o1 = o1;
            this.o2 = o2;
            this.o3 = o3;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T instanceOf(T option) {
            final Class<?> c = option.getClass();
            if (c.isInstance(o1)) {
                return (T) o1;
            }
            if (c.isInstance(o2)) {
                return (T) o2;
            }
            return c.isInstance(o3) ? (T) o3 : null;
        }
    }

    /**
     * Options implementation with 4 entries.
     */
    private static class Options4 extends Options {
        /** First option. */
        private final Object o1;
        /** Second option. */
        private final Object o2;
        /** Third option. */
        private final Object o3;
        /** Fourth option. */
        private final Object o4;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         * @param o3 Third option.
         * @param o4 Fourth option.
         */
        Options4(Object o1, Object o2, Object o3, Object o4) {
            this.o1 = o1;
            this.o2 = o2;
            this.o3 = o3;
            this.o4 = o4;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T instanceOf(T option) {
            final Class<?> c = option.getClass();
            if (c.isInstance(o1)) {
                return (T) o1;
            }
            if (c.isInstance(o2)) {
                return (T) o2;
            }
            if (c.isInstance(o3)) {
                return (T) o3;
            }
            return c.isInstance(o4) ? (T) o4 : null;
        }
    }

    /**
     * Options implementation with n entries.
     */
    private static class OptionsN extends Options {
        /** Array of options. */
        private final Object[] options;

        /**
         * @param o Array of option.
         */
        OptionsN(Object[] o) {
            options = o.clone();
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T instanceOf(T option) {
            final Class<?> c = option.getClass();
            for (final Object o : options) {
                if (c.isInstance(o)) {
                    return (T) o;
                }
            }
            return null;
        }

        @Override
        public <T> T orElse(T option) {
            return getOrElse(option, options);
        }
    }

    /**
     * Gets the first object that is assignment-compatible with the object
     * represented by the specified {@code option}, or {@code option} if the options
     * contains no instance that can be cast to the reference type of the specified
     * {@code option}.
     *
     * @param <T> Type of the option.
     * @param option Option.
     * @param options Arrays of possible options.
     * @return the first assignable instance, or {@code option}
     * @see Class#isInstance(Object)
     */
    @SuppressWarnings("unchecked")
    static <T> T getOrElse(T option, Object... options) {
        final Class<?> c = option.getClass();
        for (final Object o : options) {
            if (c.isInstance(o)) {
                return (T) o;
            }
        }
        return null;
    }
}
