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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Test cases for {@link InferenceUtils}.
 */
class InferenceUtilsTest {

    @ParameterizedTest
    @ValueSource(doubles = {0, 0.5000000000000001, 1, -1, -Double.MIN_VALUE, Double.NaN})
    void testCheckSignificanceThrows(double alpha) {
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkSignificance(alpha));
        Assertions.assertTrue(ex.getMessage().contains(Double.toString(alpha)));
    }

    @ParameterizedTest
    @ValueSource(ints = {Integer.MIN_VALUE, -1})
    void testCheckNonNegativeIntThrows(double v) {
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkNonNegative(v));
        Assertions.assertTrue(ex.getMessage().contains(Double.toString(v)));
    }

    @ParameterizedTest
    @ValueSource(doubles = {-Double.MIN_VALUE, -1, Double.NEGATIVE_INFINITY, Double.NaN})
    void testCheckNonNegativeDoubleThrows(double v) {
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkNonNegative(v));
        Assertions.assertTrue(ex.getMessage().contains(Double.toString(v)));
    }

    @ParameterizedTest
    @ValueSource(longs = {Long.MIN_VALUE, -1})
    void testCheckNonNegativeLongArrayThrows(long v) {
        final long[] a = {v};
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkNonNegative(a));
        Assertions.assertTrue(ex.getMessage().contains(Long.toString(v)));
    }

    @ParameterizedTest
    @ValueSource(longs = {Long.MIN_VALUE, -1})
    void testCheckNonNegativeLongArrayArrayThrows(long v) {
        final long[][] a = {{v}};
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkNonNegative(a));
        Assertions.assertTrue(ex.getMessage().contains(Long.toString(v)));
    }

    @ParameterizedTest
    @ValueSource(ints = {0, -1, Integer.MIN_VALUE})
    void testCheckStrictlyPositiveIntThrows(int v) {
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkStrictlyPositive(v));
        Assertions.assertTrue(ex.getMessage().contains(Integer.toString(v)));
    }

    @ParameterizedTest
    @ValueSource(doubles = {0, -1, Double.NEGATIVE_INFINITY, Double.NaN})
    void testCheckStrictlyPositiveDoubleArrayThrows(double v) {
        final double[] a = {v};
        final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkStrictlyPositive(a));
        Assertions.assertTrue(ex.getMessage().contains(Double.toString(v)));
    }

    @Test
    void testCheckNonNanArrayThrows() {
        Assertions.assertDoesNotThrow(() -> InferenceUtils.checkNonNaN(new double[0]));
        final double[] a = new double[3];
        Assertions.assertDoesNotThrow(() -> InferenceUtils.checkNonNaN(a));
        for (int i = 0; i < a.length; i++) {
            a[i] = Double.NaN;
            final IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class,
                () -> InferenceUtils.checkNonNaN(a));
            Assertions.assertTrue(ex.getMessage().contains("NaN"));
            a[i] = 0;
        }
    }

    @Test
    void testCheckRectangular() {
        // Input is assumed to be non-zero length: this test what happens
        Assertions.assertThrows(NullPointerException.class,
            () -> InferenceUtils.checkRectangular(null));
        Assertions.assertThrows(IndexOutOfBoundsException.class,
            () -> InferenceUtils.checkRectangular(new long[0][0]));
        InferenceUtils.checkRectangular(new long[1][0]);
        InferenceUtils.checkRectangular(new long[1][1]);
        InferenceUtils.checkRectangular(new long[1][2]);
        InferenceUtils.checkRectangular(new long[2][0]);
        InferenceUtils.checkRectangular(new long[2][1]);
        InferenceUtils.checkRectangular(new long[2][2]);
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkRectangular(new long[][] {{0}, {}}));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkRectangular(new long[][] {{0, 0}, {}}));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkRectangular(new long[][] {{0, 0}, {0}}));
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> InferenceUtils.checkRectangular(new long[][] {{0, 0}, {0, 0, 0}}));
    }

    @Test
    void testCheckValuesRequiredSize() {
        InferenceUtils.checkValuesRequiredSize(1, 1);
        InferenceUtils.checkValuesRequiredSize(10, 2);
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> InferenceUtils.checkValuesRequiredSize(0, 1), "values", "0", "1");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> InferenceUtils.checkValuesRequiredSize(1, 2), "values", "1", "2");
    }

    @Test
    void testCheckCategoriesRequiredSize() {
        InferenceUtils.checkCategoriesRequiredSize(1, 1);
        InferenceUtils.checkCategoriesRequiredSize(10, 2);
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> InferenceUtils.checkCategoriesRequiredSize(0, 1), "categories", "0", "1");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> InferenceUtils.checkCategoriesRequiredSize(1, 2), "categories", "1", "2");
    }

    @Test
    void testCheckValuesSizeMatch() {
        InferenceUtils.checkValuesSizeMatch(1, 1);
        InferenceUtils.checkValuesSizeMatch(10, 10);
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
            () -> InferenceUtils.checkValuesSizeMatch(0, 1), "values", "mismatch", "0", "1");
        TestUtils.assertThrowsWithMessage(IllegalArgumentException.class,
                () -> InferenceUtils.checkValuesSizeMatch(3, 2), "values", "mismatch", "3", "2");
    }
}
