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
 * A fixed size set of indices.
 *
 * <p>This is a specialised class to implement a reduced API of a
 * {@link java.util.BitSet}. It uses no range checks and supports only a fixed size.
 * It contains the methods required to store and look-up intervals of indices.
 *
 * <p>See the BloomFilter code in Commons Collections for use of long[] data to
 * store bits.
 *
 * @since 1.1
 */
final class IndexSet implements IntIntBiPredicate {
    /** All 64-bits bits set. */
    private static final long LONG_MASK = -1L;
    /** A bit shift to apply to an integer to divided by 64 (2^6). */
    private static final int DIVIDE_BY_64 = 6;

    /** Bit indexes. */
    private final long[] data;

    /** Index offset. */
    private final int offset = 0;

    /**
     * Create an instance.
     *
     * @param size Size.
     */
    IndexSet(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Invalid size: " + size);
        }
        data = new long[getLongIndex(size - 1) + 1];
    }

    /**
     * Gets the filter index for the specified bit index assuming the filter is using
     * 64-bit longs to store bits starting at index 0.
     *
     * <p>The index is assumed to be positive. For a positive index the result will match
     * {@code bitIndex / 64}.</p>
     *
     * <p><em>The divide is performed using bit shifts. If the input is negative the
     * behavior is not defined.</em></p>
     *
     * @param bitIndex the bit index (assumed to be positive)
     * @return the index of the bit map in an array of bit maps.
     */
    private static int getLongIndex(final int bitIndex) {
        // An integer divide by 64 is equivalent to a shift of 6 bits if the integer is
        // positive.
        // We do not explicitly check for a negative here. Instead we use a
        // signed shift. Any negative index will produce a negative value
        // by sign-extension and if used as an index into an array it will throw an
        // exception.
        return bitIndex >> DIVIDE_BY_64;
    }

    /**
     * Gets the filter bit mask for the specified bit index assuming the filter is using
     * 64-bit longs to store bits starting at index 0. The returned value is a
     * {@code long} with only 1 bit set.
     *
     * <p>The index is assumed to be positive. For a positive index the result will match
     * {@code 1L << (bitIndex % 64)}.</p>
     *
     * <p><em>If the input is negative the behavior is not defined.</em></p>
     *
     * @param bitIndex the bit index (assumed to be positive)
     * @return the filter bit
     */
    private static long getLongBit(final int bitIndex) {
        // Bit shifts only use the first 6 bits. Thus it is not necessary to mask this
        // using 0x3f (63) or compute bitIndex % 64.
        // Note: If the index is negative the shift will be (64 - (bitIndex & 0x3f)) and
        // this will identify an incorrect bit.
        return 1L << bitIndex;
    }

    // Copy method API from BitSet

    /**
     * Returns the value of the bit with the specified index.
     *
     * @param bitIndex the bit index (assumed to be positive)
     * @return the value of the bit with the specified index
     */
    public boolean get(int bitIndex) {
        final int i = getLongIndex(bitIndex);
        final long m = getLongBit(bitIndex);
        return (data[i] & m) != 0;
    }

    /**
     * Sets the bit at the specified index to {@code true}.
     *
     * @param bitIndex the bit index (assumed to be positive)
     */
    public void set(int bitIndex) {
        final int i = getLongIndex(bitIndex);
        final long m = getLongBit(bitIndex);
        data[i] |= m;
    }

    /**
     * Sets the bits from the specified {@code fromIndex} (inclusive) to the
     * specified {@code toIndex} (exclusive) to {@code true}.
     *
     * <p><em>If {@code toIndex - fromIndex <= 0} the behavior is not defined.</em></p>
     *
     * @param  fromIndex index of the first bit to be set
     * @param  toIndex index after the last bit to be set
     */
    public void set(int fromIndex, int toIndex) {
        // Optimisation for the main use case of this index set
        if (fromIndex + 1 == toIndex) {
            set(fromIndex);
            return;
        }

        int i = getLongIndex(fromIndex);
        // WARNING: If toIndex == fromIndex == 0 this will error
        final int j = getLongIndex(toIndex - 1);

        // Fill in bits using (big-endian mask):
        // end      middle   start
        // 00011111 11111111 11111100

        // start = -1L << (fromIndex % 64)
        // end = -1L >>> (64 - (toIndex % 64))
        final long start = LONG_MASK << fromIndex;
        final long end  = LONG_MASK >>> -toIndex;
        if (i == j) {
            // Special case where the two masks overlap at the same long index
            // 11111100 & 00011111 => 00011100
            data[i] |= start & end;
        } else {
            // 11111100
            data[i] |= start;
            while (++i < j) {
                // 11111111
                // Note: -1L is all bits set
                data[i] = -1L;
            }
            // 00011111
            data[j] |= end;
        }
    }

    /**
     * Returns the index of the nearest bit that is set to {@code true} that occurs on or
     * before the specified starting index. If no such bit exists, or if {@code -1} is
     * given as the starting index, then {@code -1} is returned.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @return the index of the previous set bit, or {@code -1} if there is no such bit
     */
    public int previousSetBit(int fromIndex) {
        int i = getLongIndex(fromIndex);

        // Mask bits before the bit index
        // mask = 00011111 = -1L >>> (64 - ((fromIndex + 1) % 64))
        long bits = data[i] & (LONG_MASK >>> -(fromIndex + 1));
        for (;;) {
            if (bits != 0) {
                //(i+1)       i
                // |  index   |
                // |    |     |
                // 0  001010000
                return (i + 1) * Long.SIZE - Long.numberOfLeadingZeros(bits) - 1;
            }
            if (i == 0) {
                return -1;
            }
            bits = data[--i];
        }
    }

    /**
     * Returns the index of the first bit that is set to {@code true} that occurs on or
     * after the specified starting index. If no such bit exists then {@code -1} is
     * returned.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @return the index of the next set bit, or {@code -1} if there is no such bit
     */
    public int nextSetBit(int fromIndex) {
        int i = getLongIndex(fromIndex);

        // Mask bits after the bit index
        // mask = 11111000 = -1L << (fromIndex % 64)
        long bits = data[i] & (LONG_MASK << fromIndex);
        for (;;) {
            if (bits != 0) {
                //(i+1)       i
                // |    index |
                // |      |   |
                // 0  001010000
                return i * Long.SIZE + Long.numberOfTrailingZeros(bits);
            }
            if (++i == data.length) {
                return -1;
            }
            bits = data[i];
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean test(int left, int right) {
        return false;
    }
}
