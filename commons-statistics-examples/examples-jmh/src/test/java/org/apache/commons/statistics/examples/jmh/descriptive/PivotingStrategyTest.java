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

import java.util.Arrays;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link PivotingStrategy}.
 */
class PivotingStrategyTest {
    @ParameterizedTest
    @MethodSource
    void testCentral(double[] a, int expected) {
        Assertions.assertEquals(expected, PivotingStrategy.CENTRAL.pivotIndex(a, 0, a.length));
    }

    @ParameterizedTest
    @MethodSource
    void testMedianOf3(double[] a, int expected) {
        Assertions.assertEquals(expected, PivotingStrategy.MEDIAN_OF_3.pivotIndex(a, 0, a.length));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedianOf3"})
    void testMedianOf9as3(double[] a, int expected) {
        Assertions.assertEquals(expected, PivotingStrategy.MEDIAN_OF_9.pivotIndex(a, 0, a.length));
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedianOf9"})
    void testMedianOf9(double[] a, int i, int j, int k) {
        // Sometimes this is off by an index of 1
        final int index = PivotingStrategy.MEDIAN_OF_9.pivotIndex(a, 0, a.length);
        if (index != j && index != i && index != k) {
            Assertions.fail(() -> String.valueOf(index));
        }
    }

    @Test
    void testMedianOf5Indexing() {
        // Safe from length 5. At small lengths the indexes cannot spread across the range
        // efficiently. This is OK from about length 20.
        for (int i = 5; i < 50; i++) {
            final int n = i;
            final double[] a = new double[i];
            Assertions.assertDoesNotThrow(() -> PivotingStrategy.MEDIAN_OF_5.pivotIndex(a, 0, a.length),
                () -> "Length: " + n);
        }
    }

    @ParameterizedTest
    @MethodSource
    void testMedianOf5(double[] a, int expected) {
        Assertions.assertEquals(expected, PivotingStrategy.MEDIAN_OF_5.pivotIndex(a, 0, a.length));
    }

    @ParameterizedTest
    @MethodSource
    void testMedianOf5sorted(double[] a, int i, int j, int k, int l, int m) {
        final double[] before = new double[] {a[i], a[j], a[k], a[l], a[m]};
        PivotingStrategy.MEDIAN_OF_5.pivotIndex(a, 0, a.length);
        Assertions.assertEquals(k, PivotingStrategy.MEDIAN_OF_5.pivotIndex(a, 0, a.length));
        final double[] after = new double[] {a[i], a[j], a[k], a[l], a[m]};
        Arrays.sort(before);
        Assertions.assertArrayEquals(before, after);
    }

    @ParameterizedTest
    @MethodSource(value = {"testMedianOf3", "testDynamic"})
    void testDynamic(double[] a) {
        final int index = PivotingStrategy.DYNAMIC.pivotIndex(a, 0, a.length);
        final int j = PivotingStrategy.MEDIAN_OF_3.pivotIndex(a, 0, a.length);
        if (index != j) {
            Assertions.assertEquals(PivotingStrategy.MEDIAN_OF_9.pivotIndex(a, 0, a.length),
                index);
        }
    }

    static Stream<Arguments> testCentral() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        add(builder, 0, 42);
        add(builder, 1, 42, 46);
        add(builder, 1, 42, 46, 49);
        add(builder, 1, -3, -46, -2);
        add(builder, 2, -3, -46, -2, 8);
        return builder.build();
    }

    static Stream<Arguments> testMedianOf3() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        add(builder, 1, 1, 3, 5);
        add(builder, 2, 1, 5, 3);
        add(builder, 0, 3, 1, 5);
        add(builder, 0, 3, 5, 1);
        add(builder, 2, 5, 1, 3);
        add(builder, 1, 5, 3, 1);
        // Original version used a Double.compare sort order
        //final double z = Double.NaN;
        //add(builder, 1, 1, 3, z);
        //add(builder, 2, 1, z, 3);
        //add(builder, 0, 3, 1, z);
        //add(builder, 0, 3, z, 1);
        //add(builder, 2, z, 1, 3);
        //add(builder, 1, z, 3, 1);
        return builder.build();
    }

    static Stream<Arguments> testMedianOf9() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        // Original version used a Double.compare sort order
        final double z = 9; //Double.NaN;
        final double l = 4;
        final double m = 5;
        final double n = 6;
        final double[] a = {1, 2, 3, l, m, n, 7, z, z};
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        // Permutations is 9! = 362880
        // Sample from them
        for (int i = 0; i < 500; i++) {
            TestUtils.shuffle(rng, a);
            builder.add(Arguments.of(a.clone(), indexOf(a, l), indexOf(a, m), indexOf(a, n)));
        }
        return builder.build();
    }

    static Stream<double[]> testDynamic() {
        final Stream.Builder<double[]> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create(123);
        // Big enough to use median of 9
        final double[] a = rng.doubles(50).toArray();
        for (int i = 0; i < 10; i++) {
            TestUtils.shuffle(rng, a);
            builder.add(a.clone());
        }
        return builder.build();
    }

    static Stream<Arguments> testMedianOf5() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final double[] a = new double[5];
        // Permutations is 5! = 120
        int shift = 42;
        for (int i = 0; i < 5; i++) {
            a[0] = i + shift;
            for (int j = 0; j < 5; j++) {
                if (j == i) {
                    continue;
                }
                a[1] = j + shift;
                for (int k = 0; k < 5; k++) {
                    if (k == j || k == i) {
                        continue;
                    }
                    a[2] = k + shift;
                    for (int l = 0; l < 5; l++) {
                        if (l == k || l == j || l == i) {
                            continue;
                        }
                        a[3] = l + shift;
                        for (int m = 0; m < 5; m++) {
                            if (m == l || m == k || m == j || m == i) {
                                continue;
                            }
                            a[3] = m + shift;
                            builder.add(Arguments.of(a.clone(), 2));
                        }
                    }
                }
            }
        }
        return builder.build();
    }

    static Stream<Arguments> testMedianOf5sorted() {
        final Stream.Builder<Arguments> builder = Stream.builder();
        final UniformRandomProvider rng = RandomSource.XO_SHI_RO_128_PP.create();
        for (int n = 8; n < 256; n *= 2) {
            for (int i = 0; i < 10; i++) {
                int length = rng.nextInt(n, n * 2);
                // Copy logic for indices: 1/6, 1/3, 1/2, 2/3, 5/6
                int len = length - 1;
                int sixth = 1 + (len >>> 3) + (len >>> 5);
                int p3 = len >>> 1;
                int p2 = p3 - sixth;
                int p1 = p2 - sixth;
                int p4 = p3 + sixth;
                int p5 = p4 + sixth;
                builder.add(Arguments.of(rng.doubles(length).toArray(), p1, p2, p3, p4, p5));
            }
        }
        return builder.build();
    }

    private static void add(Stream.Builder<Arguments> builder, int expected, double... a) {
        builder.add(Arguments.of(a, expected));
    }

    private static int indexOf(double[] a, double v) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == v) {
                return i;
            }
        }
        return -1;
    }
}
