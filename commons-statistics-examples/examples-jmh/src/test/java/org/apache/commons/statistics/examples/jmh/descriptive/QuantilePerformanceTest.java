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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Executes tests for {@link QuantilePerformance}.
 */
class QuantilePerformanceTest {
    @Test
    void testGetInteger() {
        Assertions.assertEquals(42, QuantilePerformance.getInteger("text", 42));
        Assertions.assertEquals(33, QuantilePerformance.getInteger("text33", 42));
        Assertions.assertEquals(-33, QuantilePerformance.getInteger("text-33", 42));
    }

    @Test
    void testGetMinQuickSelectSize() {
        Assertions.assertEquals(Partition.MIN_QUICKSELECT_SIZE, QuantilePerformance.getMinQuickSelectSize("nothing"));
        Assertions.assertEquals(22, QuantilePerformance.getMinQuickSelectSize("QS22"));
        Assertions.assertEquals(23, QuantilePerformance.getMinQuickSelectSize("beforeQS23"));
        Assertions.assertEquals(24, QuantilePerformance.getMinQuickSelectSize("QS24after"));
        Assertions.assertEquals(25, QuantilePerformance.getMinQuickSelectSize("beforeQS25after"));
        Assertions.assertEquals(26, QuantilePerformance.getMinQuickSelectSize("beforeQS26_HS16after"));
    }

    @Test
    void testGetHeapSelectShift() {
        Assertions.assertEquals(Partition.HEAPSELECT_SHIFT, QuantilePerformance.getHeapSelectShift("nothing"));
        Assertions.assertEquals(12, QuantilePerformance.getHeapSelectShift("HS12"));
        Assertions.assertEquals(13, QuantilePerformance.getHeapSelectShift("beforeHS13"));
        Assertions.assertEquals(14, QuantilePerformance.getHeapSelectShift("HS14after"));
        Assertions.assertEquals(15, QuantilePerformance.getHeapSelectShift("beforeHS15after"));
        Assertions.assertEquals(16, QuantilePerformance.getHeapSelectShift("beforeQS26_HS16after"));
    }

    @Test
    void testGetHeapSelectConstant() {
        Assertions.assertEquals(Partition.HEAPSELECT_CONSTANT, QuantilePerformance.getHeapSelectConstant("nothing"));
        Assertions.assertEquals(12, QuantilePerformance.getHeapSelectConstant("HC12"));
        Assertions.assertEquals(13, QuantilePerformance.getHeapSelectConstant("beforeHC13"));
        Assertions.assertEquals(14, QuantilePerformance.getHeapSelectConstant("HC14after"));
        Assertions.assertEquals(15, QuantilePerformance.getHeapSelectConstant("beforeHC15after"));
        Assertions.assertEquals(16, QuantilePerformance.getHeapSelectConstant("beforeQS26_HC16after"));
    }

    @Test
    void testGetRecursionMultiple() {
        Assertions.assertEquals(Partition.RECURSION_MULTIPLE, QuantilePerformance.getRecursionMultiple("nothing"));
        Assertions.assertEquals(1.2, QuantilePerformance.getRecursionMultiple("RM1.2"));
        Assertions.assertEquals(1.3, QuantilePerformance.getRecursionMultiple("beforeRM1.3"));
        Assertions.assertEquals(1.4, QuantilePerformance.getRecursionMultiple("RM1.4after"));
        Assertions.assertEquals(1.5, QuantilePerformance.getRecursionMultiple("beforeRM1.5after"));
        Assertions.assertEquals(1.6, QuantilePerformance.getRecursionMultiple("beforeQS26_RM1.6after"));
        Assertions.assertEquals(0, QuantilePerformance.getRecursionMultiple("RM0"));
        Assertions.assertEquals(3, QuantilePerformance.getRecursionMultiple("RM3"));
        Assertions.assertEquals(4, QuantilePerformance.getRecursionMultiple("RM4"));
        Assertions.assertEquals(5, QuantilePerformance.getRecursionMultiple("RM5_"));
        Assertions.assertEquals(6, QuantilePerformance.getRecursionMultiple("RM6a"));
    }

    @Test
    void testGetRecursionConstant() {
        Assertions.assertEquals(Partition.RECURSION_CONSTANT, QuantilePerformance.getRecursionConstant("nothing"));
        Assertions.assertEquals(0, QuantilePerformance.getRecursionConstant("RC0"));
        Assertions.assertEquals(12, QuantilePerformance.getRecursionConstant("RC12"));
        Assertions.assertEquals(13, QuantilePerformance.getRecursionConstant("beforeRC13"));
        Assertions.assertEquals(14, QuantilePerformance.getRecursionConstant("RC14after"));
        Assertions.assertEquals(15, QuantilePerformance.getRecursionConstant("beforeRC15after"));
        Assertions.assertEquals(16, QuantilePerformance.getRecursionConstant("beforeQS26_RC16after"));
    }

    @Test
    void testGetCompressionLevel() {
        Assertions.assertEquals(Partition.COMPRESSION, QuantilePerformance.getCompressionLevel("nothing"));
        Assertions.assertEquals(2, QuantilePerformance.getCompressionLevel("CL2"));
        Assertions.assertEquals(12, QuantilePerformance.getCompressionLevel("CL12"));
        Assertions.assertEquals(13, QuantilePerformance.getCompressionLevel("beforeCL13"));
        Assertions.assertEquals(14, QuantilePerformance.getCompressionLevel("CL14after"));
        Assertions.assertEquals(15, QuantilePerformance.getCompressionLevel("beforeCL15after"));
        Assertions.assertEquals(16, QuantilePerformance.getCompressionLevel("beforeQS26_CL16after"));
    }

    @Test
    void testGetPivotingStrategy() {
        Assertions.assertEquals(Partition.PIVOTING_STRATEGY, QuantilePerformance.getPivotStrategy("nothing"));
        // Ensure partial name matching is avoided
        for (final PivotingStrategy s : PivotingStrategy.values()) {
            Assertions.assertEquals(s, QuantilePerformance.getPivotStrategy(s.toString()), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getPivotStrategy("before" + s), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getPivotStrategy(s + "after"), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getPivotStrategy("before" + s + "after"), () -> s.toString());
        }
    }

    @Test
    void testGetDualPivotingStrategy() {
        Assertions.assertEquals(Partition.DUAL_PIVOTING_STRATEGY, QuantilePerformance.getDualPivotStrategy("nothing"));
        // Ensure partial name matching is avoided
        for (final DualPivotingStrategy s : DualPivotingStrategy.values()) {
            Assertions.assertEquals(s, QuantilePerformance.getDualPivotStrategy(s.toString()), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getDualPivotStrategy("before" + s), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getDualPivotStrategy(s + "after"), () -> s.toString());
            Assertions.assertEquals(s, QuantilePerformance.getDualPivotStrategy("before" + s + "after"), () -> s.toString());
        }
    }
}
