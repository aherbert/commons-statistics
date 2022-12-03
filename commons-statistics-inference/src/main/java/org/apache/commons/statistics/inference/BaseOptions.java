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
 * Base implementation for the options of a test for significance.
 *
 * @since 1.1
 */
class BaseOptions {
    /** Alternative hypothesis. */
    private final AlternativeHypothesis alternative;

    /**
     * Builder for the {@link BaseOptions}.
     */
    class Builder {
        /** Alternative hypothesis. */
        private AlternativeHypothesis alternative;

        /**
         * @param source Source to copy.
         */
        Builder(BaseOptions source) {
            alternative = source.alternative;
        }

        /**
         * Sets the alternative hypothesis.
         *
         * @param value Value.
         * @return a reference to {@code this}
         * @see BaseOptions#getAlternative()
         */
        public Builder setAlternative(AlternativeHypothesis value) {
            this.alternative = value;
            return this;
        }

        /**
         * Builds the options.
         *
         * @return the options
         */
        BaseOptions build() {
            return new BaseOptions(this);
        }
    }

    /**
     * Create the default options.
     */
    BaseOptions() {
        alternative = AlternativeHypothesis.TWO_SIDED;
    }

    /**
     * @param source Source to copy.
     */
    BaseOptions(Builder source) {
        alternative = source.alternative;
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
}
