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

package org.apache.commons.statistics.distribution;

/**
 * Test cases for {@link TrapezoidalDistribution}.
 * Extends {@link BaseContinuousDistributionTest}. See javadoc of that class for details.
 */
class TrapezoidalDistributionTest extends BaseContinuousDistributionTest {
    @Override
    ContinuousDistribution makeDistribution(Object... parameters) {
        final double a = (Double) parameters[0];
        final double b = (Double) parameters[1];
        final double c = (Double) parameters[2];
        final double d = (Double) parameters[3];
        return TrapezoidalDistribution.of(a, b, c, d);
    }


    @Override
    Object[][] makeInvalidParameters() {
        return new Object[][] {
            {0.0, 0.0, 0.0, 0.0},
            // 1.0, 2.0, 3.0, 4.0 is OK - move points to incorrect locations
            {5.0, 2.0, 3.0, 4.0}, // a > d
            {1.0, 5.0, 3.0, 4.0}, // b > d
            {1.0, 2.0, 5.0, 4.0}, // c > d
            {3.5, 2.0, 3.0, 4.0}, // a > c
            {1.0, 3.5, 3.0, 4.0}, // b > c
            {2.5, 2.0, 3.0, 4.0}, // a > b
            {1.0, 2.0, 3.0, 0.0}, // d < a
            {1.0, 2.0, 0.0, 4.0}, // c < a
            {1.0, 0.0, 3.0, 4.0}, // b < a
            {1.0, 2.0, 3.0, 1.5}, // d < b
            {1.0, 2.0, 1.5, 4.0}, // c < b
            {1.0, 2.0, 3.0, 2.5}, // d < c
        };
    }

    @Override
    String[] getParameterNames() {
        return new String[] {"SupportLowerBound", "B", "C", "SupportUpperBound"};
    }

    @Override
    protected double getRelativeTolerance() {
        // Tolerance is 4.440892098500626E-15.
        return 20 * RELATIVE_EPS;
    }
}
