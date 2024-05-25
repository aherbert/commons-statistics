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

import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link DoubleMath}.
 */
class DoubleMathTest {
    @Test
    void testGreaterThanLessThan() {
        final double[] values = {0.0, 1.0, Double.POSITIVE_INFINITY, Double.NaN};
        final int[] sign = {-1, 1};
        for (final double a : values) {
            for (final double b : values) {
                for (final int i : sign) {
                    final double x = i * a;
                    for (final int j : sign) {
                        final double y = j * b;
                        Assertions.assertEquals(Double.compare(x, y) > 0, DoubleMath.greaterThan(x, y),
                            () -> x + " > " + y);
                        Assertions.assertEquals(Double.compare(x, y) < 0, DoubleMath.lessThan(x, y),
                            () -> x + " < " + y);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource
    void testMean(double x, double y, double expected) {
        Assertions.assertEquals(expected, DoubleMath.mean(x, y));
    }

    static Stream<Arguments> testMean() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        builder.add(Arguments.of(2, 3, 2.5));
        builder.add(Arguments.of(-4, 3, -0.5));
        builder.add(Arguments.of(-4, 4, 0));
        builder.add(Arguments.of(-0.0, -0.0, -0.0));
        builder.add(Arguments.of(-Double.MAX_VALUE, Double.MAX_VALUE, 0));
        builder.add(Arguments.of(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE));
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource
    void testInterpolate(double x, double y, double alpha, double expected) {
        Assertions.assertEquals(expected, DoubleMath.interpolate(x, y, alpha), 0.0);
    }

    static Stream<Arguments> testInterpolate() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Same cases as the mean
        builder.add(Arguments.of(2, 3, 0.5, 2.5));
        builder.add(Arguments.of(-4, 3, 0.5, -0.5));
        builder.add(Arguments.of(-4, 4, 0.5, 0));
        builder.add(Arguments.of(-0.0, -0.0, 0.5, -0.0));
        builder.add(Arguments.of(-Double.MAX_VALUE, Double.MAX_VALUE, 0.5, 0));
        // Interpolation
        builder.add(Arguments.of(1, 11, 0, 1));
        builder.add(Arguments.of(1, 11, 0.1, 2));
        builder.add(Arguments.of(1, 11, 0.2, 3));
        builder.add(Arguments.of(1, 11, 0.7, 8));
        builder.add(Arguments.of(1, 11, 1, 11));
        return builder.build();
    }
}
