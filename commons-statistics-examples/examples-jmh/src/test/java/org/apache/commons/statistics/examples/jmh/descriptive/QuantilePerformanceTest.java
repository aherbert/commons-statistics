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
    void testGetMinQuickSelectSize() {
        Assertions.assertEquals(Partition.MIN_QUICKSELECT_SIZE, QuantilePerformance.getMinQuickSelectSize("nothing"));
        Assertions.assertEquals(22, QuantilePerformance.getMinQuickSelectSize("QS22"));
        Assertions.assertEquals(23, QuantilePerformance.getMinQuickSelectSize("beforeQS23"));
        Assertions.assertEquals(24, QuantilePerformance.getMinQuickSelectSize("QS24after"));
        Assertions.assertEquals(25, QuantilePerformance.getMinQuickSelectSize("beforeQS25after"));
        Assertions.assertEquals(26, QuantilePerformance.getMinQuickSelectSize("beforeQS26_HS16after"));
    }

    @Test
    void testGetMinHeapSelectSize() {
        Assertions.assertEquals(Partition.MIN_HEAPSELECT_SIZE, QuantilePerformance.getMinHeapSelectSize("nothing"));
        Assertions.assertEquals(12, QuantilePerformance.getMinHeapSelectSize("HS12"));
        Assertions.assertEquals(13, QuantilePerformance.getMinHeapSelectSize("beforeHS13"));
        Assertions.assertEquals(14, QuantilePerformance.getMinHeapSelectSize("HS14after"));
        Assertions.assertEquals(15, QuantilePerformance.getMinHeapSelectSize("beforeHS15after"));
        Assertions.assertEquals(16, QuantilePerformance.getMinHeapSelectSize("beforeQS26_HS16after"));
    }
}
