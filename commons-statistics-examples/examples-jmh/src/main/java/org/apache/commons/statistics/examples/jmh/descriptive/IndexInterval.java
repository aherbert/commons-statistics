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
 * A searchable interval that contains indices used for partitioning an array into multiple regions.
 *
 * <p>The interval provides the following functionality:
 *
 * <ul>
 * <li>Return the supported bounds of the search {@code [left <= right]}.
 * <li>Return the previous index contained in the interval from a search point {@code k}.
 * <li>Return the next index contained in the interval from a search point {@code k}.
 * </ul>
 *
 * <p>Note that the interval provides the supported bounds. If a search passes the end
 * of the bounds then the result is only defined as below or above the bounds. The interval
 * is intended to be used for searching inside an interval.
 *
 * <p>Implementations may assume indices are positive.
 *
 * @since 1.1
 */
interface IndexInterval {

    /**
     * The start (inclusive) of the range of indices supported.
     *
     * @return start of the supported range
     */
    int left();

    /**
     * The end (inclusive) of the range of indices supported.
     *
     * @return end of the supported range
     */
    int right();

    /**
     * Returns the nearest index that occurs on or before the specified starting
     * index. If none exist then a value below the {@code left} bound of support is returned.
     *
     * @param k Index to start checking from (inclusive).
     * @return the previous index, or a value below {@code left} if there is no index
     */
    int previousIndex(int k);

    /**
     * Returns the nearest index that occurs on or after the specified starting
     * index. If none exist then a value above the {@code right} bound of support is returned.
     *
     * @param k Index to start checking from (inclusive).
     * @return the next index, or a value above {@code right} if there is no index
     */
    int nextIndex(int k);
}
