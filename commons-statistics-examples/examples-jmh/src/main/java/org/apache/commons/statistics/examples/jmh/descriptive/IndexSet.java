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
import java.util.function.IntConsumer;

/**
 * A fixed size set of indices within an inclusive range {@code [left, right]}.
 *
 * <p>This is a specialised class to implement a reduced API similar to a
 * {@link java.util.BitSet}. It uses no upper bounds range checks and supports only a
 * fixed size. It contains the methods required to store and look-up intervals of indices.
 *
 * <p>An offset is supported to allow the fixed size to cover a range of indices starting
 * above 0 with the most efficient usage of storage.
 *
 * <p>See the BloomFilter code in Commons Collections for use of long[] data to store
 * bits.
 *
 * @since 1.1
 */
final class IndexSet implements PivotStore {
    /** All 64-bits bits set. */
    private static final long LONG_MASK = -1L;
    /** A bit shift to apply to an integer to divided by 64 (2^6). */
    private static final int DIVIDE_BY_64 = 6;

    /** Bit indexes. */
    private final long[] data;

    /** Index offset. */
    private final int offset;

    /**
     * Create an instance to store indices within the range {@code [offset, offset + size]}.
     *
     * @param offset Index offset.
     * @param size Number of additional indices to support after offset.
     */
    private IndexSet(int offset, int size) {
        this.offset = offset;
        data = new long[getLongIndex(size) + 1];
    }

    /**
     * Create an instance to store indices within the range {@code [left, right]}.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the index set
     * @throws IllegalArgumentException if {@code right < left}
     */
    static IndexSet ofRange(int left, int right) {
        if (left < 0) {
            throw new IllegalArgumentException("Invalid lower index: " + left);
        }
        checkRange(left, right);
        return new IndexSet(left, right - left);
    }

    /**
     * Initialise an instance with the {@code indices}. The capacity is defined by the
     * range required to store the minimum and maximum index.
     *
     * @param indices Indices.
     * @return the index set
     * @throws IllegalArgumentException if {@code indices.length == 0}
     */
    static IndexSet of(int[] indices) {
        return of(indices, indices.length);
    }

    /**
     * Initialise an instance with the {@code indices}. The capacity is defined by the
     * range required to store the minimum and maximum index.
     *
     * @param indices Indices.
     * @param n Number of indices.
     * @return the index set
     * @throws IllegalArgumentException if {@code n == 0}
     */
    static IndexSet of(int[] indices, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("No indices to define the range");
        }
        int min = indices[0];
        int max = min;
        for (int i = 1; i < n; i++) {
            min = Math.min(min, indices[i]);
            max = Math.max(max, indices[i]);
        }
        final IndexSet set = IndexSet.ofRange(min, max);
        for (int i = 0; i < n; i++) {
            set.add(indices[i]);
        }
        return set;
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

    // Adapt method API from BitSet

    /**
     * Returns the value of the bit with the specified index.
     *
     * @param bitIndex the bit index (assumed to be positive)
     * @return the value of the bit with the specified index
     */
    public boolean get(int bitIndex) {
        if (bitIndex < offset) {
            return false;
        }
        final int index = bitIndex - offset;
        final int i = getLongIndex(index);
        final long m = getLongBit(index);
        return (data[i] & m) != 0;
    }

    /**
     * Sets the bit at the specified index to {@code true}.
     *
     * @param bitIndex the bit index (assumed to be positive)
     */
    @Override
    public void add(int bitIndex) {
        if (bitIndex >= offset) {
            final int index = bitIndex - offset;
            final int i = getLongIndex(index);
            final long m = getLongBit(index);
            data[i] |= m;
        }
    }

    /**
     * Sets the bits from the specified {@code leftIndex} (inclusive) to the specified
     * {@code rightIndex} (inclusive) to {@code true}.
     *
     * <p><em>If {@code rightIndex - leftIndex < 0} the behavior is not defined.</em></p>
     *
     * <p>Note: In contrast to the BitSet API, this uses an inclusive end as this is the
     * main use case for the class.
     *
     * @param leftIndex the left index
     * @param rightIndex the right index
     */
    @Override
    public void add(int leftIndex, int rightIndex) {
        // Optimisation for the main use case of this index set
        if (leftIndex == rightIndex) {
            add(leftIndex);
            return;
        }

        final int left = leftIndex - offset;
        final int right = rightIndex - offset;

        // WARNING: If leftIndex or rightIndex < offset this will error
        int i = getLongIndex(left);
        final int j = getLongIndex(right);

        // Fill in bits using (big-endian mask):
        // end      middle   start
        // 00011111 11111111 11111100

        // start = -1L << (left % 64)
        // end = -1L >>> (64 - ((right+1) % 64))
        final long start = LONG_MASK << left;
        final long end = LONG_MASK >>> -(right + 1);
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
     * before the specified starting index. If no such bit exists, then {@code -1} is returned.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @return the index of the previous set bit, or {@code -1} if there is no such bit
     */
    public int previousSetBit(int fromIndex) {
        return previousSetBitOrElse(fromIndex, -1);
    }

    /**
     * Returns the index of the nearest bit that is set to {@code true} that occurs on or
     * before the specified starting index. If no such bit exists, then
     * {@code defaultValue} is returned.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @param defaultValue Default value.
     * @return the index of the previous set bit, or {@code defaultValue} if there is no such bit
     */
    int previousSetBitOrElse(int fromIndex, int defaultValue) {
        if (fromIndex < offset) {
            // index is in an unknown range
            return defaultValue;
        }
        final int index = fromIndex - offset;
        int i = getLongIndex(index);

        long bits = data[i];

        // Repeat logic of get(int) to check the bit
        if ((bits & getLongBit(index)) != 0) {
            return fromIndex;
        }

        // Mask bits before the bit index
        // mask = 00011111 = -1L >>> (64 - ((index + 1) % 64))
        bits &= LONG_MASK >>> -(index + 1);
        for (;;) {
            if (bits != 0) {
                //(i+1)       i
                // |  index   |
                // |    |     |
                // 0  001010000
                return (i + 1) * Long.SIZE - Long.numberOfLeadingZeros(bits) - 1 + offset;
            }
            if (i == 0) {
                return defaultValue;
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
        return nextSetBitOrElse(fromIndex, -1);
    }

    /**
     * Returns the index of the first bit that is set to {@code true} that occurs on or
     * after the specified starting index. If no such bit exists then {@code defaultValue} is
     * returned.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @param defaultValue Default value.
     * @return the index of the next set bit, or {@code defaultValue} if there is no such bit
     */
    int nextSetBitOrElse(int fromIndex, int defaultValue) {
        // Support searching forward through the known range
        final int index = fromIndex < offset ? 0 : fromIndex - offset;

        int i = getLongIndex(index);

        // Mask bits after the bit index
        // mask = 11111000 = -1L << (index % 64)
        long bits = data[i] & (LONG_MASK << index);
        for (;;) {
            if (bits != 0) {
                //(i+1)       i
                // |    index |
                // |      |   |
                // 0  001010000
                return i * Long.SIZE + Long.numberOfTrailingZeros(bits) + offset;
            }
            if (++i == data.length) {
                return defaultValue;
            }
            bits = data[i];
        }
    }

    /**
     * Returns the index of the first bit that is set to {@code false} that occurs on or
     * after the specified starting index <em>within the supported range</em>. If no such
     * bit exists then the {@code capacity} is returned where {@code capacity = index + 1}
     * with {@code index} the largest index that can be added to the set without an error.
     *
     * <p>If the starting index is less than the supported range the result is {@code fromIndex}.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @return the index of the next unset bit, or the {@code capacity} if there is no such bit
     */
    public int nextClearBit(int fromIndex) {
        if (fromIndex < offset) {
            return fromIndex;
        }
        // Support searching forward through the known range
        final int index = fromIndex - offset;

        int i = getLongIndex(index);

        // Note: This method is conceptually the same as nextSetBit with the exception
        // that: all the data is bit-flipped; the capacity is returned when the
        // scan reaches the end.

        // Mask bits after the bit index
        // mask = 11111000 = -1L << (fromIndex % 64)
        long bits = ~data[i] & (LONG_MASK << index);
        for (;;) {
            if (bits != 0) {
                return i * Long.SIZE + Long.numberOfTrailingZeros(bits) + offset;
            }
            if (++i == data.length) {
                // Capacity
                return data.length * Long.SIZE + offset;
            }
            bits = ~data[i];
        }
    }

    /**
     * Returns the index of the first bit that is set to {@code false} that occurs on or
     * before the specified starting index <em>within the supported range</em>. If no such
     * bit exists then {@code -1} is returned.
     *
     * <p>If the starting index is less than the supported range the result is {@code fromIndex}.
     * This can return {@code -1} only if the support begins at {@code index == 0}.
     *
     * @param fromIndex Index to start checking from (inclusive).
     * @return the index of the previous unset bit, or {@code -1} if there is no such bit
     */
    public int previousClearBit(int fromIndex) {
        if (fromIndex < offset) {
            // index is in an unknown range
            return fromIndex;
        }
        final int index = fromIndex - offset;
        int i = getLongIndex(index);

        // Note: This method is conceptually the same as previousSetBit with the exception
        // that: all the data is bit-flipped; the offset - 1 is returned when the
        // scan reaches the end.

        // Mask bits before the bit index
        // mask = 00011111 = -1L >>> (64 - ((index + 1) % 64))
        long bits = ~data[i] & (LONG_MASK >>> -(index + 1));
        for (;;) {
            if (bits != 0) {
                //(i+1)       i
                // |  index   |
                // |    |     |
                // 0  001010000
                return (i + 1) * Long.SIZE - Long.numberOfLeadingZeros(bits) - 1 + offset;
            }
            if (i == 0) {
                return offset - 1;
            }
            bits = ~data[--i];
        }
    }

    /**
     * Perform the {@code action} for each index in the set.
     *
     * @param action Action.
     */
    public void forEach(IntConsumer action) {
        // Adapted from o.a.c.collections4.IndexProducer
        int wordIdx = offset;
        for (int i = 0; i < data.length; i++) {
            long bits = data[i];
            int index = wordIdx;
            while (bits != 0) {
                if ((bits & 1L) == 1L) {
                    action.accept(index);
                }
                bits >>>= 1;
                index++;
            }
            wordIdx += Long.SIZE;
        }
    }

    /**
     * Return a {@link PivotCache} implementation re-using the same internal storage.
     *
     * <p>Note that the range for the {@link PivotCache} must fit inside the current
     * supported range of indices.
     *
     * <p>Warning: This operation clears all set bits within the range.
     *
     * <p><strong>Support</strong>
     *
     * <p>The returned {@link PivotCache} is suitable for storing all pivot points between
     * {@code [left, right]} and the closest bounding pivots outside that range. It can be
     * used for bracketing partition keys processed in a random order by storing pivots
     * found during each successive partition search.
     *
     * <p>The returned {@link PivotCache} is suitable for use when iterating over
     * partition keys in ascending order.
     *
     * <p>The cache allows incrementing the {@code left} support using
     * {@link PivotCache#moveLeft(int)}. Any calls to decrement the {@code left} support
     * at any time will result in an {@link IllegalStateException}; this prevents reseting
     * to within the original support used to create the cache. If the {@code left} is
     * moved beyond the {@code right} then the move is rejected.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the pivot cache
     * @throws IllegalArgumentException if {@code right < left} or the range cannot be
     * supported.
     */
    PivotCache asPivotCache(int left, int right) {
        return asPivotCache(left, right, true);
    }

    /**
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]}.
     *
     * <p>See {@link #asPivotCache(int, int)} for the details of the cache implementation.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @return the pivot cache
     * @throws IllegalArgumentException if {@code right < left}
     */
    static PivotCache createPivotCache(int left, int right) {
        final IndexSet set = ofRange(left, right);
        return set.asPivotCache(left, right, false);
    }

    /**
     * Return a {@link PivotCache} implementation to support the range
     * {@code [left, right]} re-using the same internal storage.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @param initialize Perform validation checks and initialize the storage.
     * @return the pivot cache
     * @throws IllegalArgumentException if {@code right < left} or the range cannot be
     * supported.
     */
    private PivotCache asPivotCache(int left, int right, boolean initialize) {
        if (initialize) {
            checkRange(left, right);
            final int capacity = data.length * Long.SIZE + offset;
            if (left < offset || right >= capacity) {
                throw new IllegalArgumentException(
                    String.format("Unsupported range: [%d, %d] is not within [%d, %d]", left, right,
                        offset, capacity - 1));
            }
            // Clear existing data
            Arrays.fill(data, 0);
        }
        return new IndexPivotCache(left, right);
    }

    /**
     * Check the range is valid.
     *
     * @param left Lower bound (inclusive).
     * @param right Upper bound (inclusive).
     * @throws IllegalArgumentException if {@code right < left}
     */
    private static void checkRange(int left, int right) {
        if (right < left) {
            throw new IllegalArgumentException(
                String.format("Invalid range: [%d, %d]", left, right));
        }
    }

    /**
     * Implementation of the {@link PivotCache} using the {@link IndexSet}.
     *
     * <p>Stores all pivots between the support {@code [left, right]}. Uses two
     * floating pivots which are the closest known pivots surrounding this range.
     *
     * <p>This class is bound to the enclosing {@link IndexSet} instance to provide
     * the functionality to read, write and search indexes.
     */
    class IndexPivotCache implements PivotCache {
        /** Default value for an unset upper floating pivot.
         * Set as a value higher than any valid array index. */
        private static final int UPPER_DEFAULT = Integer.MAX_VALUE;

        /** Left bound of the support. */
        private int left;
        /** Left bound of the support. */
        private final int right;
        /** The upstream pivot closest to the left bound of the support.
         * Provides a lower search bound for the range [left, right]. */
        private int lowerPivot = -1;
        /** The downstream pivot closest to the right bound of the support.
         * Provides an upper search bound for the range [left, right]. */
        private int upperPivot = UPPER_DEFAULT;

        /**
         * @param left Lower bound (inclusive).
         * @param right Upper bound (inclusive).
         */
        IndexPivotCache(int left, int right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public int left() {
            return left;
        }

        @Override
        public int right() {
            return right;
        }

        @Override
        public boolean sparse() {
            // Can store all pivots between [left, right]
            return false;
        }

        @Override
        public boolean moveLeft(int newLeft) {
            if (newLeft > right) {
                // Signal that this cache can no longer be used in that range
                return false;
            }
            if (newLeft < left) {
                throw new IllegalStateException(
                    String.format("New left is outside current support: %d < %d", newLeft, left));
            }
            // Here [left <= newLeft <= right]
            // Move the upstream pivot
            lowerPivot = previousPivot(newLeft);
            left = newLeft;
            return true;
        }

        @Override
        public boolean contains(int k) {
            // Assume [left <= k <= right]
            return IndexSet.this.get(k);
        }

        @Override
        public int previousPivot(int k) {
            // Assume scanning in [left <= k <= right]
            return IndexSet.this.previousSetBitOrElse(k, lowerPivot);
        }

        @Override
        public int nextPivotOrElse(int k, int other) {
            // Assume scanning in [left <= k <= right]
            final int p = upperPivot == UPPER_DEFAULT ? other : upperPivot;
            return IndexSet.this.nextSetBitOrElse(k, p);
        }

        @Override
        public int nextNonPivot(int k) {
            // Assume scanning in [left <= k <= right]
            return IndexSet.this.nextClearBit(k);
        }

        @Override
        public int previousNonPivot(int k) {
            // Assume scanning in [left <= k <= right]
            return IndexSet.this.previousClearBit(k);
        }

        @Override
        public void add(int index) {
            // Update the floating pivots if outside the support
            if (index < left) {
                lowerPivot = Math.max(index, lowerPivot);
            } else if (index > right) {
                upperPivot = Math.min(index, upperPivot);
            } else {
                IndexSet.this.add(index);
            }
        }

        @Override
        public void add(int fromIndex, int toIndex) {
            if (fromIndex == toIndex) {
                add(fromIndex);
                return;
            }
            // Note:
            // Storing all pivots allows regions of identical values
            // and sorted regions to be skipped in subsequent partitioning.
            // Repeat sorting these regions is typically more expensive
            // than caching them and moving over them during partitioning.
            // An alternative is to: store fromIndex and only store
            // toIndex if they are well separated, optionally storing
            // regions between. If they are not well separated (e.g. < 10)
            // then using a single pivot is an alternative to investigate
            // with performance benchmarks on a range of input data.

            // Pivots are required to bracket [L, R]:
            // LP-----L--------------R------UP
            // If the range [i, j] overlaps either L or R then
            // the floating pivots are no longer required:
            //     i-j                             Set lower pivot
            //     i--------j                      Ignore lower pivot
            //     i---------------------j         Ignore lower & upper pivots (no longer required)
            //           i-------j                 Ignore lower & upper pivots
            //           i---------------j         Ignore upper pivot
            //                         i-j         Set upper pivot
            if (fromIndex <= right && toIndex >= left) {
                // Clip the range between [left, right]
                final int i = Math.max(fromIndex, left);
                final int j = Math.min(toIndex, right);
                IndexSet.this.add(i, j);
            } else if (toIndex < left) {
                lowerPivot = Math.max(toIndex, lowerPivot);
            } else {
                // fromIndex > right
                upperPivot = Math.min(fromIndex, upperPivot);
            }
        }
    }
}
