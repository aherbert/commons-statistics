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
 * A strategy to pick two pivot indices of an array for partitioning.
 *
 * <p>An ideal strategy will pick the tertiles across a variety of data so
 * to divide the data into [1/3, 1/3, 1/3].
 *
 * <a href="https://en.wiktionary.org/wiki/tertile">Tertile (Wiktionary)</a>
 */
enum DualPivotingStrategy {
    /**
     * Pivot around the medians at 1/3 and 2/3 of the range.
     *
     * <p>Requires {@code right - left >= 2}.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0000   0.9922   0.3275   0.2328   0.0023   0.5887
     *   0.0003   0.9835   0.3358   0.2359   0.0028   0.5562
     *   0.0000   0.9924   0.3367   0.2371   0.0025   0.5445
     * </pre>
     */
    MEDIANS {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Original 'medians' method from the dual-pivot quicksort paper by Vladimir Yaroslavskiy
            final int len = right - left;
            // Do not pivot at the ends by setting 1/3 to at least 1.
            // This is safe if len >= 2.
            final int third = Math.max(1, len / 3);
            final int m1 = left + third;
            final int m2 = right - third;
            // Ensure p1 is lower
            if (data[m1] < data[m2]) {
                pivot2[0] = m2;
                return m1;
            }
            pivot2[0] = m1;
            return m2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int third = Math.max(1, len / 3);
            final int m1 = left + third;
            final int m2 = right - third;
            return new int[] {m1, m2};
        }

        @Override
        int samplingEffect() {
            return UNCHANGED;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- sixths from the median: 1/6, 1/3, 1/2, 2/3, 5/6.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0020   0.9788   0.3313   0.1770   0.0231   0.5037
     *   0.0029   0.9590   0.3341   0.1779   0.0232   0.4483
     *   0.0019   0.9405   0.3347   0.1796   0.0258   0.4829
     * </pre>
     */
    SORT_5 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/6 = 5/30 ~ 1/8 + 1/32 + 1/64 : 0.1666 ~ 0.1719
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int sixth = 1 + (len >>> 3) + (len >>> 5) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - sixth;
            final int p1 = p2 - sixth;
            final int p4 = p3 + sixth;
            final int p5 = p4 + sixth;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int sixth = 1 + (len >>> 3) + (len >>> 5) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - sixth;
            final int p1 = p2 - sixth;
            final int p4 = p3 + sixth;
            final int p5 = p4 + sixth;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- sevenths from the median: 3/14, 5/14, 1/2, 9/14, 11/14.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0025   0.9016   0.3329   0.1793   0.0203   0.4775
     *   0.0021   0.9232   0.3341   0.1780   0.0238   0.4626
     *   0.0030   0.9034   0.3330   0.1789   0.0243   0.4860
     * </pre>
     */
    SORT_5B {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/7 = 5/35 ~ 1/8 + 1/64 : 0.1429 ~ 0.1406
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int seventh = 1 + (len >>> 3) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - seventh;
            final int p1 = p2 - seventh;
            final int p4 = p3 + seventh;
            final int p5 = p4 + seventh;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int seventh = 1 + (len >>> 3) + (len >>> 6);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - seventh;
            final int p1 = p2 - seventh;
            final int p4 = p3 + seventh;
            final int p5 = p4 + seventh;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 approximately uniformly spaced within the range.
     * Uses points +/- eights from the median: 1/4, 3/8, 1/2, 5/8, 3/4.
     *
     * <p>Requires {@code right - left >= 4}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0023   0.9128   0.3314   0.1779   0.0220   0.4445
     *   0.0013   0.9287   0.3301   0.1773   0.0197   0.4777
     *   0.0026   0.9631   0.3386   0.1796   0.0220   0.4662
     * </pre>
     */
    SORT_5C {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/8 = 0.125
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int eighth = 1 + (len >>> 3);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p4 = p3 + eighth;
            final int p5 = p4 + eighth;
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int eighth = 1 + (len >>> 3);
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p4 = p3 + eighth;
            final int p5 = p4 + eighth;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 1st and 5th values from 5 approximately uniformly spaced within the range.
     *
     * <p>Requires {@code right - left >= 3}.
     *
     * <p>Warning: This has the side effect that the 5 values are also sorted.
     */
    // TODO - remove this. It is for testing
    SORT_5J {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // JDK 11 method

            // Does not work well for small range
            // step = size * 3 / 8 + 3
            int step = ((right - left + 1) >> 3) * 3 + 3;
            int p1 = left + step;
            int p5 = right - step;
            int p3 = (p1 + p5) >>> 1;
            int p2 = (p1 + p3) >>> 1;
            int p4 = (p3 + p5) >>> 1;

            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p5;
            return p1;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            int step = ((right - left + 1) >> 3) * 3 + 3;
            int p1 = left + step;
            int p5 = right - step;
            int p3 = (p1 + p5) >>> 1;
            int p2 = (p1 + p3) >>> 1;
            int p4 = (p3 + p5) >>> 1;
            return new int[] {p1, p2, p3, p4, p5};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 medians approximately uniformly spaced within
     * the range. The medians are from 3 samples. The 5 samples of 3 do not overlap thus this
     * method requires {@code right - left >= 14}. The samples can be visualised as 5 sorted
     * columns:
     *
     * <pre>
     * v w x y z
     * 1 2 3 4 5
     * a b c d e
     * </pre>
     *
     * <p>The pivots are points 2 and 4. The other points are either known to be below or
     * above the pivots; or potentially below or above the pivots.
     *
     * <p>Pivot 1: below {@code 1,a,b}; potentially below {@code v,c,d,e}. This ranks
     * pivot 1 from 4/15 to 8/15 and exactly 5/15 if the input data is sorted/reverse sorted.
     *
     * <p>Pivot 2: above {@code 5,y,z}; potentially above {@code e,v,w,x}. This ranks
     * pivot 2 from 7/15 to 11/15 and exactly 10/15 if the input data is sorted/reverse sorted.
     *
     * <p>Warning: This has the side effect that the 15 samples values are partially sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0420   0.8264   0.3784   0.1328   0.0871   0.2028
     *   0.0040   0.7892   0.2429   0.1342   0.0185   0.6171
     *   0.0362   0.8325   0.3787   0.1336   0.0878   0.2273
     * </pre>
     * <p>Note the bias towards the outer regions.
     */
    SORT_5_OF_3 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Step size of 1/16 of the length
            final int len = right - left;
            final int step = Math.max(1, len >>> 4);
            final int step3 = step * 3;
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - step3;
            final int p1 = p2 - step3;
            final int p4 = p3 + step3;
            final int p5 = p4 + step3;
            // 5 medians of 3
            Sorting.sort3(data, p1 - step, p1, p1 + step);
            Sorting.sort3(data, p2 - step, p2, p2 + step);
            Sorting.sort3(data, p3 - step, p3, p3 + step);
            Sorting.sort3(data, p4 - step, p4, p4 + step);
            Sorting.sort3(data, p5 - step, p5, p5 + step);
            // Sort the medians
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int step = Math.max(1, len >>> 4);
            final int step3 = step * 3;
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - step3;
            final int p1 = p2 - step3;
            final int p4 = p3 + step3;
            final int p5 = p4 + step3;
            return new int[] {
                p1 - step, p1, p1 + step,
                p2 - step, p2, p2 + step,
                p3 - step, p3, p3 + step,
                p4 - step, p4, p4 + step,
                p5 - step, p5, p5 + step,
            };
        }

        @Override
        int samplingEffect() {
            return PARTIAL_SORT;
        }
    },
    /**
     * Pivot around the 2nd and 3rd values from 4 medians approximately uniformly spaced within
     * the range. The medians are from 3 samples. The 4 samples of 3 do not overlap thus this
     * method requires {@code right - left >= 11}. The samples can be visualised as 4 sorted
     * columns:
     *
     * <pre>
     * w x y z
     * 1 2 3 4
     * a b c d
     * </pre>
     *
     * <p>The pivots are points 2 and 3. The other points are either known to be below or
     * above the pivots; or potentially below or above the pivots.
     *
     * <p>Pivot 1: below {@code 1,a,b}; potentially below {@code w,c,d}. This ranks
     * pivot 1 from 4/12 to 7/12 and exactly 5/12 if the input data is sorted/reverse sorted.
     *
     * <p>Pivot 2: above {@code 4,y,z}; potentially above {@code d,w,x}. This ranks
     * pivot 2 from 5/15 to 8/12 and exactly 7/12 if the input data is sorted/reverse sorted.
     *
     * <p>Warning: This has the side effect that the 12 samples values are partially sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0250   0.9151   0.4282   0.1451   0.1050   0.1390
     *   0.0002   0.7777   0.1443   0.1192   0.0011   1.1811
     *   0.0288   0.9375   0.4275   0.1457   0.0997   0.1206
     * </pre>
     * <p>Note the large bias towards the outer regions.
     */
    SORT_4_OF_3 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Step size of 1/13 of the length: 1/13 ~ 1/16 + 1/64 : 0.0769 ~ 0.0781
            final int len = right - left;
            final int step = Math.max(1, (len >>> 4) + (len >>> 6));
            final int step3 = step * 3;
            final int p1 = left + (step << 1) - 1;
            final int p2 = p1 + step3;
            final int p3 = p2 + step3;
            final int p4 = p3 + step3;
            // 5 medians of 3
            Sorting.sort3(data, p1 - step, p1, p1 + step);
            Sorting.sort3(data, p2 - step, p2, p2 + step);
            Sorting.sort3(data, p3 - step, p3, p3 + step);
            Sorting.sort3(data, p4 - step, p4, p4 + step);
            // Sort the medians
            Sorting.sort4(data, p1, p2, p3, p4);
            pivot2[0] = p3;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int step = Math.max(1, (len >>> 4) + (len >>> 6));
            final int step3 = step * 3;
            final int p1 = left + (step << 1) - 1;
            final int p2 = p1 + step3;
            final int p3 = p2 + step3;
            final int p4 = p3 + step3;
            return new int[] {
                p1 - step, p1, p1 + step,
                p2 - step, p2, p2 + step,
                p3 - step, p3, p3 + step,
                p4 - step, p4, p4 + step,
            };
        }

        @Override
        int samplingEffect() {
            return PARTIAL_SORT;
        }
    },
    /**
     * Pivot around the 1st and 3rd values from 3 medians approximately uniformly spaced within
     * the range. The medians are from 3 samples. The 3 samples of 3 do not overlap thus this
     * method requires {@code right - left >= 8}. The samples can be visualised as 3 sorted
     * columns:
     *
     * <pre>
     * x y z
     * 1 2 3
     * a b c
     * </pre>
     *
     * <p>The pivots are points 1 and 3. The other points are either known to be below or
     * above the pivots; or potentially below or above the pivots.
     *
     * <p>Pivot 1: below {@code a}; potentially below {@code b, c}. This ranks
     * pivot 1 from 2/9 to 4/9 and exactly 2/9 if the input data is sorted/reverse sorted.
     *
     * <p>Pivot 2: above {@code z}; potentially above {@code x,y}. This ranks
     * pivot 2 from 6/9 to 8/9 and exactly 8/9 if the input data is sorted/reverse sorted.
     *
     * <p>Warning: This has the side effect that the 9 samples values are partially sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0024   0.8599   0.3039   0.1554   0.0234   0.4564
     *   0.0137   0.9396   0.3891   0.1805   0.0333   0.2575
     *   0.0023   0.8910   0.3070   0.1567   0.0204   0.4138
     * </pre>
     * <p>Note the bias towards the central region.
     */
    SORT_3_OF_3 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Step size of 1/8 of the length
            final int len = right - left;
            final int step = Math.max(1, len >>> 3);
            final int step3 = step * 3;
            final int p2 = left + (len >>> 1);
            final int p1 = p2 - step3;
            final int p3 = p2 + step3;
            // 3 medians of 3
            Sorting.sort3(data, p1 - step, p1, p1 + step);
            Sorting.sort3(data, p2 - step, p2, p2 + step);
            Sorting.sort3(data, p3 - step, p3, p3 + step);
            // Sort the medians
            Sorting.sort3(data, p1, p2, p3);
            pivot2[0] = p3;
            return p1;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int step = Math.max(1, len >>> 3);
            final int step3 = step * 3;
            final int p2 = left + (len >>> 1);
            final int p1 = p2 - step3;
            final int p3 = p2 + step3;
            return new int[] {
                p1 - step, p1, p1 + step,
                p2 - step, p2, p2 + step,
                p3 - step, p3, p3 + step,
            };
        }

        @Override
        int samplingEffect() {
            return PARTIAL_SORT;
        }
    },
    /**
     * Pivot around the 2nd and 4th values from 5 medians approximately uniformly spaced within
     * the range. The medians are from 5 samples. The 5 samples of 5 do not overlap thus this
     * method requires {@code right - left >= 24}. The samples can be visualised as 5 sorted
     * columns:
     *
     * <pre>
     * v w x y z
     * q r s t u
     * 1 2 3 4 5
     * f g h i j
     * a b c d e
     * </pre>
     *
     * <p>The pivots are points 2 and 4. The other points are either known to be below or
     * above the pivots; or potentially below or above the pivots.
     *
     * <p>Pivot 1: below {@code 1,a,b,f,g}; potentially below {@code q,v,c,d,e,h,i,j}. This ranks
     * pivot 1 from 6/25 to 14/25 and exactly 8/25 if the input data is sorted/reverse sorted.
     *
     * <p>Pivot 2 by symmetry from 12/25 to 20/25 and exactly 18/25 for sorted data.
     *
     * <p>Warning: This has the side effect that the 25 samples values are partially sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0316   0.7934   0.3993   0.1099   0.1334   0.0924
     *   0.0029   0.7115   0.1993   0.1116   0.0139   0.7039
     *   0.0850   0.8002   0.4013   0.1096   0.1429   0.1068
     * </pre>
     * <p>Note the bias towards the outer regions.
     */
    SORT_5_OF_5 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Step size of 1/25 of the length
            final int len = right - left;
            final int step = Math.max(1, len / 25);
            final int step2 = step << 1;
            final int step5 = step * 5;
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - step5;
            final int p1 = p2 - step5;
            final int p4 = p3 + step5;
            final int p5 = p4 + step5;
            // 5 medians of 3
            Sorting.sort5(data, p1 - step2, p1 - step, p1, p1 + step, p1 + step2);
            Sorting.sort5(data, p2 - step2, p2 - step, p2, p2 + step, p2 + step2);
            Sorting.sort5(data, p3 - step2, p3 - step, p3, p3 + step, p3 + step2);
            Sorting.sort5(data, p4 - step2, p4 - step, p4, p4 + step, p4 + step2);
            Sorting.sort5(data, p5 - step2, p5 - step, p5, p5 + step, p5 + step2);
            // Sort the medians
            Sorting.sort5(data, p1, p2, p3, p4, p5);
            pivot2[0] = p4;
            return p2;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            // Step size of 1/25 of the length
            final int len = right - left;
            final int step = Math.max(1, len / 25);
            final int step2 = step << 1;
            final int step5 = step * 5;
            final int p3 = left + (len >>> 1);
            final int p2 = p3 - step5;
            final int p1 = p2 - step5;
            final int p4 = p3 + step5;
            final int p5 = p4 + step5;
            return new int[] {
                p1 - step2, p1 - step, p1, p1 + step, p1 + step2,
                p2 - step2, p2 - step, p2, p2 + step, p2 + step2,
                p3 - step2, p3 - step, p3, p3 + step, p3 + step2,
                p4 - step2, p4 - step, p4, p4 + step, p4 + step2,
                p5 - step2, p5 - step, p5, p5 + step, p5 + step2,
            };
        }

        @Override
        int samplingEffect() {
            return PARTIAL_SORT;
        }
    },
    /**
     * Pivot around the 3rd and 5th values from 7 approximately uniformly spaced within the range.
     * Uses points +/- eights from the median: 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8.
     *
     * <p>Requires {@code right - left >= 6}.
     *
     * <p>Warning: This has the side effect that the 7 values are also sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0129   0.9330   0.3768   0.1611   0.0562   0.3257
     *   0.0017   0.8341   0.2499   0.1451   0.0154   0.6778
     *   0.0149   0.9032   0.3733   0.1616   0.0502   0.3109
     * </pre>
     * <p>Note the bias towards the outer regions.
     */
    SORT_7 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 4.
            final int len = right - left;
            final int eighth = Math.max(1, len >>> 3);
            final int p4 = left + (len >>> 1);
            final int p3 = p4 - eighth;
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p5 = p4 + eighth;
            final int p6 = p5 + eighth;
            final int p7 = p6 + eighth;
            Sorting.sort7(data, p1, p2, p3, p4, p5, p6, p7);
            pivot2[0] = p5;
            return p3;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int eighth = Math.max(1, len >>> 3);
            final int p4 = left + (len >>> 1);
            final int p3 = p4 - eighth;
            final int p2 = p3 - eighth;
            final int p1 = p2 - eighth;
            final int p5 = p4 + eighth;
            final int p6 = p5 + eighth;
            final int p7 = p6 + eighth;
            return new int[] {p1, p2, p3, p4, p5, p6, p7};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    },
    /**
     * Pivot around the 3rd and 6th values from 8 approximately uniformly spaced within the range.
     * Uses points +/- ninths from the median: m - 4/9, m - 3/9, m - 2/9, m - 1/9; m + 1 + 1/9,
     * m + 1 + 2/9, m + 1 + 3/9, m + 1 + 4/9.
     *
     * <p>Requires {@code right - left >= 7}. Smaller ranges will result in overlap of the sampled
     * points.
     *
     * <p>Warning: This has the side effect that the 8 values are also sorted.
     *
     * <p>On random data the tertiles are:
     * <pre>
     *      min      max     mean       sd   median     skew
     *   0.0122   0.9434   0.3335   0.1491   0.0491   0.4182
     *   0.0069   0.8798   0.3357   0.1491   0.0489   0.4026
     *   0.0078   0.8918   0.3307   0.1488   0.0439   0.4085
     * </pre>
     */
    SORT_8 {
        @Override
        int pivotIndex(double[] data, int left, int right, int[] pivot2) {
            // 1/9 = 4/36 = 8/72 ~ 7/64 ~ 1/16 + 1/32 + 1/64 : 0.11111 ~ 0.1094
            // Ensure the value is above zero to choose different points!
            // This is safe if len >= 8.
            final int len = right - left;
            final int ninth = Math.max(1, (len >>> 4) + (len >>> 5) + (len >>> 6));
            // Work from middle outward. This is deliberate to ensure data.length==7
            // throws an index out-of-bound exception.
            final int m = left + (len >>> 1);
            final int p4 = m - (ninth >> 1);
            final int p3 = p4 - ninth;
            final int p2 = p3 - ninth;
            final int p1 = p2 - ninth;
            final int p5 = m + (ninth >> 1) + 1;
            final int p6 = p5 + ninth;
            final int p7 = p6 + ninth;
            final int p8 = p7 + ninth;
            Sorting.sort8(data, p1, p2, p3, p4, p5, p6, p7, p8);
            pivot2[0] = p6;
            return p3;
        }

        @Override
        int[] getSampledIndices(int left, int right) {
            final int len = right - left;
            final int ninth = Math.max(1, (len >>> 4) + (len >>> 5) + (len >>> 6));
            final int m = left + (len >>> 1);
            final int p4 = m - (ninth >> 1);
            final int p3 = p4 - ninth;
            final int p2 = p3 - ninth;
            final int p1 = p2 - ninth;
            final int p5 = m + (ninth >> 1) + 1;
            final int p6 = p5 + ninth;
            final int p7 = p6 + ninth;
            final int p8 = p7 + ninth;
            return new int[] {p1, p2, p3, p4, p5, p6, p7, p8};
        }

        @Override
        int samplingEffect() {
            return SORT;
        }
    };

    /** Sampled points are unchanged. */
    static final int UNCHANGED = 0;
    /** Sampled points are partially sorted. */
    static final int PARTIAL_SORT = 0x1;
    /** Sampled points are sorted. */
    static final int SORT = 0x2;

    /**
     * Find two pivot indices of the array so that partitioning into 3-regions can be made.
     *
     * <pre>{@code
     * left <= p1 <= p2 <= right
     * }</pre>
     *
     * <p>Returns two pivots so that {@code data[p1] <= data[p2]}.
     *
     * @param data Array.
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param pivot2 Second pivot.
     * @return first pivot
     */
    abstract int pivotIndex(double[] data, int left, int right, int[] pivot2);

    // The following methods allow the strategy and side effects to be tested

    /**
     * Get the indices of points that will be sampled.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the indices
     */
    abstract int[] getSampledIndices(int left, int right);

    /**
     * Get the effect on the sampled points.
     * <ul>
     * <li>0 - Unchanged
     * <li>1 - Partially sorted
     * <li>2 - Sorted
     * </ul>
     *
     * @return the effect
     */
    abstract int samplingEffect();
}
