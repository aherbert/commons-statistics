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
import java.util.BitSet;
import java.util.stream.Stream;
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test for {@link IndexSet}.
 */
class IndexSetTest {

    @ParameterizedTest
    @MethodSource
    void testGetSet(int[] indices, int n) {
        final IndexSet set = new IndexSet(n);
        final BitSet ref = new BitSet(n);
        for (final int i : indices) {
            Assertions.assertEquals(ref.get(i), set.get(i));
            set.set(i);
            ref.set(i);
            Assertions.assertTrue(set.get(i));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testSetRange1(int[] indices, int n) {
        final IndexSet set = new IndexSet(n);
        final BitSet ref = new BitSet(n);
        for (final int i : indices) {
            Assertions.assertEquals(ref.get(i), set.get(i));
            set.set(i, i + 1);
            ref.set(i);
            Assertions.assertTrue(set.get(i));
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testSetRange(int[] indices, int n) {
        final IndexSet set = new IndexSet(n);
        Arrays.sort(indices);
        Assertions.assertEquals(-1, set.previousSetBit(n >>> 1));
        Assertions.assertEquals(-1, set.nextSetBit(n >>> 1));
        for (int i = 1; i < indices.length; i++) {
            int from = indices[i - 1];
            int to = indices[i];
            // Invalid ranges are not checked
            if (to <= from) {
                continue;
            }
            for (int j = from; j < to; j++) {
                Assertions.assertFalse(set.get(j));
            }
            set.set(from, to);
            for (int j = from; j < to; j++) {
                Assertions.assertTrue(set.get(j));
            }
        }
    }

    @ParameterizedTest
    @MethodSource(value = {"testGetSet"})
    void testPreviousNextSetBit(int[] indices, int n) {
        final IndexSet set = new IndexSet(n);
        final BitSet ref = new BitSet(n);
        Arrays.sort(indices);
        Assertions.assertEquals(-1, set.previousSetBit(n >>> 1));
        Assertions.assertEquals(-1, set.nextSetBit(n >>> 1));
        for (int i = 1; i < indices.length; i++) {
            int from = indices[i - 1];
            int to = indices[i];
            int middle = (from + to) >>> 1;
            Assertions.assertEquals(ref.previousSetBit(middle), set.previousSetBit(middle));
            Assertions.assertEquals(ref.nextSetBit(middle), set.nextSetBit(middle));
            set.set(from);
            set.set(to);
            ref.set(from);
            ref.set(to);
            Assertions.assertEquals(ref.previousSetBit(middle), set.previousSetBit(middle));
            Assertions.assertEquals(ref.nextSetBit(middle), set.nextSetBit(middle));
        }
    }

    static Stream<Arguments> testGetSet() {
        final UniformRandomProvider rng = RandomSource.XO_RO_SHI_RO_128_PP.create();
        final Stream.Builder<Arguments> builder = Stream.builder();
        for (final int size : new int[] {5, 500}) {
            builder.accept(Arguments.of(rng.ints(10, 0, size).toArray(), size));
        }
        return builder.build();
    }
}
