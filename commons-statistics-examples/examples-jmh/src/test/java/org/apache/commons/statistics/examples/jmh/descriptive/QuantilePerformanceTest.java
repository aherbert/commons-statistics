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
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.PairedKeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.SPStrategy;
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
    void testGetEdgeSelectConstant() {
        assertIntParameter(Partition.EDGESELECT_CONSTANT, "EC", QuantilePerformance::getEdgeSelectConstant);
    }

    @Test
    void testGetLinearSortSelectConstant() {
        assertIntParameter(Partition.LINEAR_SORTSELECT_SIZE, "LC", QuantilePerformance::getLinearSortSelectConstant);
    }

    @Test
    void testGetSubSamplingSize() {
        assertIntParameter(Partition.SUBSAMPLING_SIZE, "SU", QuantilePerformance::getSubSamplingSize);
    }

    @Test
    void testGetRecursionMultiple() {
        assertDoubleParameter(Partition.RECURSION_MULTIPLE, "RM", QuantilePerformance::getRecursionMultiple);
    }

    @Test
    void testGetRecursionConstant() {
        assertIntParameter(Partition.RECURSION_CONSTANT, "RC", QuantilePerformance::getRecursionConstant);
    }

    @Test
    void testGetCompressionLevel() {
        assertIntParameter(Partition.COMPRESSION_LEVEL, "CL", QuantilePerformance::getCompressionLevel);
    }

    @Test
    void testGetControlFlags() {
        assertIntParameter(Partition.CONTROL_FLAGS, "CF", QuantilePerformance::getControlFlags);
    }

    @Test
    void testGetPivotingStrategy() {
        assertEnumParameter(Partition.PIVOTING_STRATEGY,
            s -> QuantilePerformance.getEnumOrElse(s, PivotingStrategy.class, Partition.PIVOTING_STRATEGY));
    }

    @Test
    void testGetDualPivotingStrategy() {
        assertEnumParameter(Partition.DUAL_PIVOTING_STRATEGY,
            s -> QuantilePerformance.getEnumOrElse(s, DualPivotingStrategy.class, Partition.DUAL_PIVOTING_STRATEGY));
    }

    @Test
    void testGetKeyStrategy() {
        assertEnumParameter(Partition.KEY_STRATEGY,
            s -> QuantilePerformance.getEnumOrElse(s, KeyStrategy.class, Partition.KEY_STRATEGY));
    }

    @Test
    void testGetPairedKeyStrategy() {
        assertEnumParameter(Partition.PAIRED_KEY_STRATEGY,
            s -> QuantilePerformance.getEnumOrElse(s, PairedKeyStrategy.class, Partition.PAIRED_KEY_STRATEGY));
    }

    @Test
    void testGetSPStrategy() {
        assertEnumParameter(Partition.SP_STRATEGY,
            s -> QuantilePerformance.getEnumOrElse(s, SPStrategy.class, Partition.SP_STRATEGY));
    }

    private static void assertIntParameter(int defaultValue, String pattern, ToIntFunction<String[]> fun) {
        final String[] s = {"nothing"};
        Assertions.assertEquals(defaultValue, fun.applyAsInt(s));
        Assertions.assertEquals("nothing", s[0]);
        // Prevent overflow when setting non-default values
        if (defaultValue + 4 < 0) {
            defaultValue = 0;
        }
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
