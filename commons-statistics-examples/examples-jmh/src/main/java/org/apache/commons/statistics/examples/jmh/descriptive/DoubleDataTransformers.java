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

import java.util.function.Supplier;

/**
 * Support for creating {@link DoubleDataTransformer} implementations.
 *
 * @since 1.1
 */
final class DoubleDataTransformers {

    /** No instances. */
    private DoubleDataTransformers() {}

    /**
     * Creates a factory to supply {@link DoubleDataTransformer} based on the 
     * {@code nanPolicy} and data {@code copy} policy.
     *
     * @param nanPolicy NaN policy.
     * @param copy Set to {@code true} to use a copy of the data.
     * @return the factory
     */
    static Supplier<DoubleDataTransformer> createFactory(NaNPolicy nanPolicy, boolean copy) {
        // TODO:
        // choose a constructor for a suitable implementation
        return null;
    }

//  /**
//  * Sort {@code NaN} values and count signed zeros. Any signed zero is replaced with {@code 0.0}.
//  *
//  * <p>Data can be repaired after any reordering by a parition method using
//  * {@link #replaceSignedZeros(double[], int, int)}; or a sort by
//  * {@link #replaceContinuousSignedZeros(double[], int, int)}.
//  *
//  * <p>Data transformation returns:
//  * <ul>
//  * <li>Low order 32-bits: length of the processed array without {@code NaN}
//  * <li>High order 32-bits: count of signed zeros
//  * </ul>
//  */
// SORT_NAN_COUNT_ZEROS {
//     @Override
//     long transform(double[] a) {
//         // Sort NaN / count signed zeros
//         int cn = 0;
//         int end = a.length;
//         for (int i = end; i > 0;) {
//             final double v = a[--i];
//             // Count negative zeros using a sign bit check.
//             // This requires a performance test. If the conversion to raw bits
//             // is natively supported this is faster than using the == check.
//             //if (v == 0.0 && Double.doubleToRawLongBits(v) < 0) {
//             if (Double.doubleToRawLongBits(v) == Long.MIN_VALUE) {
//                 cn++;
//                 // Change to positive zero.
//                 // Data must be repaired after sort.
//                 a[i] = 0.0;
//             } else if (v != v) {
//                 // Move NaN to end
//                 a[i] = a[--end];
//                 a[end] = v;
//             }
//         }
//         // pack [count of signed zeros, length of data]
//         return (((long) cn) << Integer.SIZE) | end;
//     }
// },
// /**
//  * Raise an {@link IllegalArgumentException} for {@code NaN} values and count signed zeros. 
//  */
// ERROR_NAN_COUNT_ZEROS {
//     @Override
//     double[] prepare(double[] data, boolean copy) {
//         // TODO Auto-generated method stub
//         return super.prepare(data, copy);
//     }
//     
//     @Override
//     long transform(double[] a) {
//         // Here we delay copy to not change the data if a NaN is found.
//         // But we commit to a double scan for signed zeros.
//         double[] a = data;
//         // Sort NaN / count signed zeros
//         int cn = 0;
//         final int end = a.length;
//         for (int i = end; i > 0;) {
//             final double v = a[--i];
//             // Count negative zeros using a sign bit check.
//             // This requires a performance test. If the conversion to raw bits
//             // is natively supported this is faster than using the == check.
//             //if (v == 0.0 && Double.doubleToRawLongBits(v) < 0) {
//             if (Double.doubleToRawLongBits(v) == Long.MIN_VALUE) {
//                 cn++;
//             } else if (v != v) {
//                 throw new IllegalArgumentException("NaN at: " + i);
//             }
//         }
//         if (copy) {
//             
//         }
//         // pack [count of signed zeros, length of data]
//         final long result = (((long) cn) << Integer.SIZE) | a.length;
//         // Re-write zeros if required
//         if (cn != 0) {
//             for (int i = end; i > 0;) {
//                 final double v = a[--i];
//                 // Count negative zeros using a sign bit check.
//                 // This requires a performance test. If the conversion to raw bits
//                 // is natively supported this is faster than using the == check.
//                 //if (v == 0.0 && Double.doubleToRawLongBits(v) < 0) {
//                 if (Double.doubleToRawLongBits(v) == Long.MIN_VALUE) {
//                     cn++;
//                 }
//             }
//         }
//         return result;
//     }
// };

    /**
     * Replace the first {@code count} occurrences of zero with {@code -0.0}
     * starting after the provided {@code from} index.
     *
     * <p>Zeros after {@code from} may be discontinuous, for example in data
     * that is partitioned.
     * 
     * <p>Warning: This method assumes that there are at least {@code count} zeros in
     * the range {@code [from + 1, a.length)}, otherwise an index out of bounds exception will
     * occur as the scan passes the end of the data.
     *
     * @param a Data.
     * @param from Index before the position to start the repair; use -1 for the start of the array.
     * @param count Count of signed zeros (assumed to be strictly positive).
     */
    static void replaceSignedZeros(double[] a, int from, int count) {
        // Assume the zeros are all present so no bounds checks
        // are used when incrementing j. But we have to check for zeros
        // before overwrite.
        for (int j = from, cn = count;;) {
            if (a[++j] == 0) {
                a[j] = -0.0;
                if (--cn == 0) {
                    break;
                }
            }
        }
    }

    /**
     * Replace the first {@code count} occurrences of zero with {@code -0.0}
     * starting after the provided {@code from} index. 
     *
     * <p>It is assumed that {@code a[from + 1] == 0} and zeros after are continuous,
     * for example in data that is sorted, or partitioned around zero using a method
     * that collects equal value together.
     * 
     * <p>Warning: This method assumes that there are at least {@code count} zeros in
     * the range {@code [from + 1, a.length)}, otherwise an index out of bounds exception will
     * occur as the scan passes the end of the data.
     *
     * @param a Data.
     * @param from Index to start the repair.
     * @param count Count of signed zeros (assumed to be strictly positive).
     */
    static void replaceContinuousSignedZeros(double[] a, int from, int count) {
        // Assume the zeros are continuous so just overwrite
        // the required count of signed zeros.
        for (int j = from, cn = count; --cn >= 0;) {
            a[++j] = -0.0;
        }
    }
}
