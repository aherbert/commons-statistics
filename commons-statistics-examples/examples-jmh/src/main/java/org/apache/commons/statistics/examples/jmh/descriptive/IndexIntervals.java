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
 * Support for creating {@link IndexInterval} implementations.
 *
 * @since 1.1
 */
final class IndexIntervals {
    /** No instances. */
    private IndexIntervals() {}

    /**
     * Returns an interval that covers all indices.
     *
     * @return the predicate
     */
    static IndexInterval anyIndex() {
        return AnyIndex.INSTANCE;
    }

    /**
     * {@link IndexInterval} for range {@code [0, MAX_VALUE]}.
     */
    private static final class AnyIndex implements IndexInterval {
        /** Singleton instance. */
        private static final AnyIndex INSTANCE = new AnyIndex();

        @Override
        public int left() {
            return 0;
        }

        @Override
        public int right() {
            return Integer.MAX_VALUE;
        }

        @Override
        public int previousIndex(int k) {
            return k;
        }

        @Override
        public int nextIndex(int k) {
            return k;
        }
    }
}
