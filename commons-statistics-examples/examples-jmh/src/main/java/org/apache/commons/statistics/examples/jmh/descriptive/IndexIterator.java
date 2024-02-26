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
 * An iterator of indices used for partitioning an array into multiple regions.
 *
 * <p>The iterator provides the functionality to iterate over blocks of indices
 * defined by an inclusive interval {@code [left, right]}:
 *
 * <pre>
 *   l----r
 *            l----r
 *                    lr
 *                              lr
 *                                       l----------------r
 * </pre>
 *
 * @since 1.1
 */
interface IndexIterator {

    /**
     * The start (inclusive) of the current block of indices.
     *
     * @return start index
     */
    int left();

    /**
     * The end (inclusive) of the current block of indices.
     *
     * @return end index
     */
    int right();

    /**
     * The end index.
     *
     * @return the end index
     */
    int end();

    /**
     * Advance the iterator to the next block of indices.
     *
     * <p>If there are no more indices the result of {@link #left()} and
     * {@link #right()} is undefined.
     *
     * @return true if the iterator was advanced
     */
    boolean next();

    /**
     * Advance the iterator so that {@code right > index}.
     *
     * <p>If there are no more indices the result of {@link #left()} and
     * {@link #right()} is undefined.
     *
     * <p>The default implementation is:
     * <pre>{@code
     * while (right() <= index) {
     *     if (!next()) {
     *         return false;
     *     }
     * }
     * return false;
     * }</pre>
     *
     * @param index Index.
     * @return true if {@code right > index}
     */
    default boolean positionAfter(int index) {
        while (right() <= index) {
            if (!next()) {
                return false;
            }
        }
        return true;
    }
}
