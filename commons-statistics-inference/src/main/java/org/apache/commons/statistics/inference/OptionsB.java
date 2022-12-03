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

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Defines the options for a test for significance.
 *
 * <p>The options function as a map between keys specified by an {@link Option} class
 * and instances of the {@link Option} class.
 *
 * <p>Instances of {@link OptionsB} are immutable.
 *
 * @since 1.1
 */
public abstract class OptionsB {
    /** Marker interface for any class that can be used as an option. */
    public interface Option {
        // marker interface
    }

    /**
     * Gets the instance of the specified class stored in the options,
     * or {@code null} if the options contains no instance of the specified class.
     *
     * @param <T> Type of the option.
     * @param key Option class.
     * @return the enumeration instance (or null)
     */
    public abstract <T extends Option> T get(Class<T> key);

    /**
     * Gets the instance of the specified {@code option}'s class stored in the options,
     * or else return the specified {@code option}.
     *
     * <p>Note the {@code option} class must be an exact match; any stored option that is
     * an {@link Class#isInstance(Object) instance} of the specified {@code option} as
     * a subclass will not be returned.
     *
     * @param <T> Type of the enumeration.
     * @param option Option.
     * @return the enumeration instance (or null)
     */
    public <T extends Option> T getOrElse(T option) {
        @SuppressWarnings("unchecked")
        final T v = (T) get(option.getClass());
        return v != null ? v : option;
    }

    /**
     * Returns an empty options.
     *
     * @return the options
     */
    public static OptionsB none() {
        return Options0.INSTANCE;
    }

    /**
     * Return options containing the specified option instance.
     *
     * @param o Option.
     * @return the options
     * @throws NullPointerException if the option instance is null
     */
    public static OptionsB of(Option o) {
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
    public static OptionsB of(Option o1, Option o2) {
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
    public static OptionsB of(Option o1, Option o2, Option o3) {
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
    public static OptionsB of(Option o1, Option o2, Option o3, Option o4) {
        return new Options4(o1, o2, o3, o4);
    }

    /**
     * Return options containing the specified option instances.
     *
     * @param o Array of options.
     * @return the options
     * @throws NullPointerException if any of the option instances are null
     */
    public static OptionsB of(Option... o) {
        return new OptionsN(o);
    }

    /**
     * Options implementation with 0 entries.
     */
    private static class Options0 extends OptionsB {
        /** Singleton instance. */
        static final OptionsB INSTANCE = new Options0();

        @Override
        public <T extends Option> T get(Class<T> key) {
            return null;
        }

        @Override
        public <T extends Option> T getOrElse(T option) {
            return Objects.requireNonNull(option);
        }
    }

    /**
     * Options implementation with 1 entry.
     */
    private static class Options1 extends OptionsB {
        /** First option. */
        private final Option o1;

        /**
         * @param o1 First option.
         */
        Options1(Option o1) {
            this.o1 = Objects.requireNonNull(o1);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T get(Class<T> key) {
            return o1.getClass().equals(key) ? (T) o1 : null;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T getOrElse(T option) {
            return o1.getClass().equals(option.getClass()) ? (T) o1 : option;
        }
    }

    /**
     * Options implementation with 2 entries.
     */
    private static class Options2 extends OptionsB {
        /** First option. */
        private final Option o1;
        /** Second option. */
        private final Option o2;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         */
        Options2(Option o1, Option o2) {
            this.o1 = Objects.requireNonNull(o1);
            this.o2 = Objects.requireNonNull(o2);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T get(Class<T> key) {
            if (o2.getClass().equals(key)) {
                return (T) o2;
            }
            return o1.getClass().equals(key) ? (T) o1 : null;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T getOrElse(T option) {
            final Object c = option.getClass();
            if (o2.getClass().equals(c)) {
                return (T) o2;
            }
            return o1.getClass().equals(c) ? (T) o1 : option;
        }
    }

    /**
     * Options implementation with 3 entries.
     */
    private static class Options3 extends OptionsB {
        /** First option. */
        private final Option o1;
        /** Second option. */
        private final Option o2;
        /** Third option. */
        private final Option o3;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         * @param o3 Third option.
         */
        Options3(Option o1, Option o2, Option o3) {
            this.o1 = Objects.requireNonNull(o1);
            this.o2 = Objects.requireNonNull(o2);
            this.o3 = Objects.requireNonNull(o3);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T get(Class<T> key) {
            if (o3.getClass().equals(key)) {
                return (T) o3;
            }
            if (o2.getClass().equals(key)) {
                return (T) o2;
            }
            return o1.getClass().equals(key) ? (T) o1 : null;
        }
    }

    /**
     * Options implementation with 4 entries.
     */
    private static class Options4 extends OptionsB {
        /** First option. */
        private final Option o1;
        /** Second option. */
        private final Option o2;
        /** Third option. */
        private final Option o3;
        /** Fourth option. */
        private final Option o4;

        /**
         * @param o1 First option.
         * @param o2 Second option.
         * @param o3 Third option.
         * @param o4 Fourth option.
         */
        Options4(Option o1, Option o2, Option o3, Option o4) {
            this.o1 = Objects.requireNonNull(o1);
            this.o2 = Objects.requireNonNull(o2);
            this.o3 = Objects.requireNonNull(o3);
            this.o4 = Objects.requireNonNull(o4);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T get(Class<T> key) {
            if (o4.getClass().equals(key)) {
                return (T) o4;
            }
            if (o3.getClass().equals(key)) {
                return (T) o3;
            }
            if (o2.getClass().equals(key)) {
                return (T) o2;
            }
            return o1.getClass().equals(key) ? (T) o1 : null;
        }
    }

    /**
     * Options implementation with n entries.
     */
    private static class OptionsN extends OptionsB {
        /** Map of options. */
        private final Map<Object, Option> map;

        /**
         * @param o Array of option.
         */
        OptionsN(Option[] o) {
            map = new HashMap<>(o.length);
            for (final Option v : o) {
                map.put(v.getClass(), v);
            }
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T get(Class<T> key) {
            return (T) map.get(key);
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T extends Option> T getOrElse(T option) {
            return (T) map.getOrDefault(option.getClass(), option);
        }
    }
}
