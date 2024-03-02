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

import java.util.EnumSet;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource.Distribution;
import org.apache.commons.statistics.examples.jmh.descriptive.QuantilePerformance.AbstractDataSource.Modification;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Executes tests for {@link QuantilePerformance}.
 */
class QuantilePerformanceTest {
    @Test
    void testGetMinQuickSelectSize() {
        assertIntParameter(Partition.MIN_QUICKSELECT_SIZE, "QS", QuantilePerformance::getMinQuickSelectSize);
    }

    @Test
    void testGetHeapSelectShift() {
        assertIntParameter(Partition.HEAPSELECT_SHIFT, "HS", QuantilePerformance::getHeapSelectShift);
    }

    @Test
    void testGetHeapSelectConstant() {
        assertIntParameter(Partition.HEAPSELECT_CONSTANT, "HC", QuantilePerformance::getHeapSelectConstant);
    }

    @Test
    void testGetHeapSelectMaskShift() {
        assertIntParameter(Partition.HEAPSELECT_MASK_SHIFT, "MS", QuantilePerformance::getHeapSelectMaskShift);
    }

    @Test
    void testGetCompressionLevel() {
        assertDoubleParameter(Partition.RECURSION_MULTIPLE, "RM", QuantilePerformance::getRecursionMultiple);
    }

    @Test
    void testGetRecursionConstant() {
        assertIntParameter(Partition.RECURSION_CONSTANT, "RC", QuantilePerformance::getRecursionConstant);
    }

    @Test
    void testGetPivotingStrategy() {
        assertEnumParameter(Partition.PIVOTING_STRATEGY, QuantilePerformance::getPivotStrategy);
    }

    @Test
    void testGetDualPivotingStrategy() {
        assertEnumParameter(Partition.DUAL_PIVOTING_STRATEGY, QuantilePerformance::getDualPivotStrategy);
    }

    @Test
    void testGetKeyStrategy() {
        assertEnumParameter(Partition.KEY_STRATEGY, QuantilePerformance::getKeyStrategy);
    }

    @Test
    void testGetPairedKeyStrategy() {
        assertEnumParameter(Partition.PAIRED_KEY_STRATEGY, QuantilePerformance::getPairedKeyStrategy);
    }

    private static void assertIntParameter(int defaultValue, String pattern, ToIntFunction<String[]> fun) {
        final String[] s = {"nothing"};
        Assertions.assertEquals(defaultValue, fun.applyAsInt(s));
        Assertions.assertEquals("nothing", s[0]);
        s[0] = pattern + (defaultValue + 1);
        Assertions.assertEquals(defaultValue + 1, fun.applyAsInt(s));
        Assertions.assertEquals("", s[0]);
        s[0] = "before" + pattern + (defaultValue + 2);
        Assertions.assertEquals(defaultValue + 2, fun.applyAsInt(s));
        Assertions.assertEquals("before", s[0]);
        s[0] = pattern + (defaultValue + 3) + "after";
        Assertions.assertEquals(defaultValue + 3, fun.applyAsInt(s));
        Assertions.assertEquals("after", s[0]);
        s[0] = "before" + pattern + (defaultValue + 4) + "after";
        Assertions.assertEquals(defaultValue + 4, fun.applyAsInt(s));
        Assertions.assertEquals("beforeafter", s[0]);
    }

    private static void assertDoubleParameter(double defaultValue, String pattern, ToDoubleFunction<String[]> fun) {
        final String[] s = {"nothing"};
        Assertions.assertEquals(defaultValue, fun.applyAsDouble(s));
        Assertions.assertEquals("nothing", s[0]);
        s[0] = pattern + (defaultValue + 0.5);
        Assertions.assertEquals(defaultValue + 0.5, fun.applyAsDouble(s));
        Assertions.assertEquals("", s[0]);
        s[0] = "before" + pattern + (defaultValue + 1.5);
        Assertions.assertEquals(defaultValue + 1.5, fun.applyAsDouble(s));
        Assertions.assertEquals("before", s[0]);
        s[0] = pattern + (defaultValue + 2.5) + "after";
        Assertions.assertEquals(defaultValue + 2.5, fun.applyAsDouble(s));
        Assertions.assertEquals("after", s[0]);
        s[0] = "before" + pattern + (defaultValue + 3.5) + "after";
        Assertions.assertEquals(defaultValue + 3.5, fun.applyAsDouble(s));
        Assertions.assertEquals("beforeafter", s[0]);
    }

    private static <E extends Enum<E>> void assertEnumParameter(E defaultValue, Function<String[], E> fun) {
        final String[] s = {"nothing"};
        Assertions.assertEquals(defaultValue, fun.apply(s));
        Assertions.assertEquals("nothing", s[0]);
        EnumSet.allOf(defaultValue.getDeclaringClass()).forEach(e -> {
            s[0] = e.toString();
            Assertions.assertEquals(e, fun.apply(s));
            Assertions.assertEquals("", s[0]);
            s[0] = "before" + e;
            Assertions.assertEquals(e, fun.apply(s));
            Assertions.assertEquals("before", s[0]);
            s[0] = e + "after";
            Assertions.assertEquals(e, fun.apply(s));
            Assertions.assertEquals("after", s[0]);
            s[0] = "before" + e + "after";
            Assertions.assertEquals(e, fun.apply(s));
            Assertions.assertEquals("beforeafter", s[0]);
        });
    }

    @Test
    void testGetDistribution() {
        assertGetEnumFromParam(Distribution.class);
    }

    @Test
    void testGetModification() {
        assertGetEnumFromParam(Modification.class);
    }

    static <E extends Enum<E>> void assertGetEnumFromParam(Class<E> cls) {
        Assertions.assertEquals(EnumSet.allOf(cls),
            AbstractDataSource.getEnumFromParam(cls, "all"));
        Assertions.assertThrows(IllegalStateException.class,
            () -> AbstractDataSource.getEnumFromParam(cls, "nothing"));
        for (final E e1 : cls.getEnumConstants()) {
            final String s = e1.name().toLowerCase();
            Assertions.assertEquals(EnumSet.of(e1),
                AbstractDataSource.getEnumFromParam(cls, e1.name()));
            Assertions.assertEquals(EnumSet.of(e1),
                AbstractDataSource.getEnumFromParam(cls, s));
            for (final E e2 : cls.getEnumConstants()) {
                Assertions.assertEquals(EnumSet.of(e1, e2),
                    AbstractDataSource.getEnumFromParam(cls, s + ":" + e2.name()));
                Assertions.assertEquals(EnumSet.of(e1, e2),
                    AbstractDataSource.getEnumFromParam(cls, e2.name() + ":" + e1.name()));
            }
        }
    }
}
