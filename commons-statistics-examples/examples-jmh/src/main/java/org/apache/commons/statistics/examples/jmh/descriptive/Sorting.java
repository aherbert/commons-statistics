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

/**
 * Support class for sorting arrays.
 *
 * <p>Optimal sorting networks are used for small fixed size array sorting.
 *
 * @see <a href="https://en.wikipedia.org/wiki/Sorting_network">Sorting network (Wikipedia)</a>
 * @see <a href="https://bertdobbelaere.github.io/sorting_networks.html">Sorting Networks (Bert Dobbelaere)</a>
 *
 * @since 1.1
 */
final class Sorting {
    /** The upper threshold to use a modified insertion sort to find unique indices. */
    private static final int UNIQUE_INSERTION_SORT = 20;

    /** No instances. */
    private Sorting() {}

    /**
     * Sorts an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>This method is fast up to approximately 40 - 80 values.
     *
     * <p>The {@code internal} flag indicates that the value at {@code data[begin - 1]}
     * is sorted.
     *
     * @param data Data array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param internal Internal flag.
     */
    static void sort(double[] data, int begin, int end, boolean internal) {
        int j;
        if (internal) {
            // Assume data[begin - 1] is a pivot and acts as a sentinal on the range.
            // => no requirement to check j >= begin.
            for (int i = begin; ++i <= end;) {
                final double v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; v < data[--j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        } else {
            for (int i = begin; ++i <= end;) {
                final double v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; --j >= begin && v < data[j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c
     * data[a] < data[b] < data[c]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     */
    static void sort3(double[] data, int i0, int i1, int i2) {
        // Sorting network for size 3 is 3 comparisons.
        // Sorting network for size 2 is 1 comparison + 1 or 2 extra

        // Order pair:
        //[(0,2)]
        // Move point 1 above point 2 or below point 0
        if (data[i2] < data[i0]) {
            final double v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }
        if (data[i2] < data[i1]) {
            final double v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        } else if (data[i1] < data[i0]) {
            final double v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d
     * data[a] < data[b] < data[c] < data[d]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     */
    static void sort4(double[] data, int i0, int i1, int i2, int i3) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // 5 comparisons.
        // Order pairs:
        //[(0,2),(1,3)]
        //[(0,1),(2,3)]
        //[(1,2)]
        if (data[i3] < data[i1]) {
            final double u = data[i3];
            data[i3] = data[i1];
            data[i1] = u;
        }
        if (data[i2] < data[i0]) {
            final double v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }

        if (data[i3] < data[i2]) {
            final double u = data[i3];
            data[i3] = data[i2];
            data[i2] = u;
        }
        if (data[i1] < data[i0]) {
            final double v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }

        if (data[i2] < data[i1]) {
            final double u = data[i2];
            data[i2] = data[i1];
            data[i1] = u;
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d != e
     * data[a] < data[b] < data[c] < data[d] < data[e]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     * @param i4 Index.
     */
    static void sort5(double[] data, int i0, int i1, int i2, int i3, int i4) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // 9 comparisons.
        // Order pairs:
        // [(0,3),(1,4)]
        // [(0,2),(1,3)]
        // [(0,1),(2,4)]
        // [(1,2),(3,4)]
        // [(2,3)]
        if (data[i4] < data[i1]) {
            final double u = data[i4];
            data[i4] = data[i1];
            data[i1] = u;
        }
        if (data[i3] < data[i0]) {
            final double v = data[i3];
            data[i3] = data[i0];
            data[i0] = v;
        }

        if (data[i3] < data[i1]) {
            final double u = data[i3];
            data[i3] = data[i1];
            data[i1] = u;
        }
        if (data[i2] < data[i0]) {
            final double v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i2]) {
            final double u = data[i4];
            data[i4] = data[i2];
            data[i2] = u;
        }
        if (data[i1] < data[i0]) {
            final double v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i3]) {
            final double u = data[i4];
            data[i4] = data[i3];
            data[i3] = u;
        }
        if (data[i2] < data[i1]) {
            final double v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        }

        if (data[i3] < data[i2]) {
            final double u = data[i3];
            data[i3] = data[i2];
            data[i2] = u;
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d != e
     * data[a] < data[b] < data[c] < data[d] < data[e]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     * @param i4 Index.
     */
    static void sort5b(double[] data, int i0, int i1, int i2, int i3, int i4) {
        // Sorting network for size 5 is 9 comparisons (see sort5b).
        // Sorting network for size 4 is 5 comparisons + 2 or 3 extra.
        // This method benchmarks marginally faster (~1%) than the sorting network of size 5
        // on length 5 data. When the data is larger and the indices are uniformly
        // spread across the range, the difference is below the noise of the timings.

        // Order quadruple:
        //[(0,1,3,4)]
        // Move point 2 above points 3,4 or below points 0,1
        sort4(data, i0, i1, i3, i4);
        final double u = data[i2];
        if (u > data[i3]) {
            data[i2] = data[i3];
            data[i3] = u;
            if (u > data[i4]) {
                data[i3] = data[i4];
                data[i4] = u;
            }
        } else if (u < data[i1]) {
            data[i2] = data[i1];
            data[i1] = u;
            if (u < data[i0]) {
                data[i1] = data[i0];
                data[i0] = u;
            }
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d != e != f != g
     * data[a] < data[b] < data[c] < data[d] < data[e] < data[f] < data[g]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     * @param i4 Index.
     * @param i5 Index.
     * @param i6 Index.
     */
    static void sort7(double[] data, int i0, int i1, int i2, int i3, int i4, int i5, int i6) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // 16 comparisons.
        // Order pairs:
        //[(0,6),(2,3),(4,5)]
        //[(0,2),(1,4),(3,6)]
        //[(0,1),(2,5),(3,4)]
        //[(1,2),(4,6)]
        //[(2,3),(4,5)]
        //[(1,2),(3,4),(5,6)]
        if (data[i5] < data[i4]) {
            final double u = data[i5];
            data[i5] = data[i4];
            data[i4] = u;
        }
        if (data[i3] < data[i2]) {
            final double v = data[i3];
            data[i3] = data[i2];
            data[i2] = v;
        }
        if (data[i6] < data[i0]) {
            final double w = data[i6];
            data[i6] = data[i0];
            data[i0] = w;
        }

        if (data[i6] < data[i3]) {
            final double u = data[i6];
            data[i6] = data[i3];
            data[i3] = u;
        }
        if (data[i4] < data[i1]) {
            final double v = data[i4];
            data[i4] = data[i1];
            data[i1] = v;
        }
        if (data[i2] < data[i0]) {
            final double w = data[i2];
            data[i2] = data[i0];
            data[i0] = w;
        }

        if (data[i4] < data[i3]) {
            final double u = data[i4];
            data[i4] = data[i3];
            data[i3] = u;
        }
        if (data[i5] < data[i2]) {
            final double v = data[i5];
            data[i5] = data[i2];
            data[i2] = v;
        }
        if (data[i1] < data[i0]) {
            final double w = data[i1];
            data[i1] = data[i0];
            data[i0] = w;
        }

        if (data[i6] < data[i4]) {
            final double u = data[i6];
            data[i6] = data[i4];
            data[i4] = u;
        }
        if (data[i2] < data[i1]) {
            final double v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        }

        if (data[i5] < data[i4]) {
            final double u = data[i5];
            data[i5] = data[i4];
            data[i4] = u;
        }
        if (data[i3] < data[i2]) {
            final double v = data[i3];
            data[i3] = data[i2];
            data[i2] = v;
        }

        if (data[i6] < data[i5]) {
            final double u = data[i6];
            data[i6] = data[i5];
            data[i5] = u;
        }
        if (data[i4] < data[i3]) {
            final double v = data[i4];
            data[i4] = data[i3];
            data[i3] = v;
        }
        if (data[i2] < data[i1]) {
            final double w = data[i2];
            data[i2] = data[i1];
            data[i1] = w;
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d != e != f != g != h
     * data[a] < data[b] < data[c] < data[d] < data[e] < data[f] < data[g] < data[h]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     * @param i4 Index.
     * @param i5 Index.
     * @param i6 Index.
     * @param i7 Index.
     */
    static void sort8(double[] data, int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // 19 comparisons.
        // Order pairs:
        //[(0,2),(1,3),(4,6),(5,7)]
        //[(0,4),(1,5),(2,6),(3,7)]
        //[(0,1),(2,3),(4,5),(6,7)]
        //[(2,4),(3,5)]
        //[(1,4),(3,6)]
        //[(1,2),(3,4),(5,6)]
        if (data[i7] < data[i5]) {
            final double u = data[i7];
            data[i7] = data[i5];
            data[i5] = u;
        }
        if (data[i6] < data[i4]) {
            final double v = data[i6];
            data[i6] = data[i4];
            data[i4] = v;
        }
        if (data[i3] < data[i1]) {
            final double w = data[i3];
            data[i3] = data[i1];
            data[i1] = w;
        }
        if (data[i2] < data[i0]) {
            final double x = data[i2];
            data[i2] = data[i0];
            data[i0] = x;
        }

        if (data[i7] < data[i3]) {
            final double u = data[i7];
            data[i7] = data[i3];
            data[i3] = u;
        }
        if (data[i6] < data[i2]) {
            final double v = data[i6];
            data[i6] = data[i2];
            data[i2] = v;
        }
        if (data[i5] < data[i1]) {
            final double w = data[i5];
            data[i5] = data[i1];
            data[i1] = w;
        }
        if (data[i4] < data[i0]) {
            final double x = data[i4];
            data[i4] = data[i0];
            data[i0] = x;
        }

        if (data[i7] < data[i6]) {
            final double u = data[i7];
            data[i7] = data[i6];
            data[i6] = u;
        }
        if (data[i5] < data[i4]) {
            final double v = data[i5];
            data[i5] = data[i4];
            data[i4] = v;
        }
        if (data[i3] < data[i2]) {
            final double w = data[i3];
            data[i3] = data[i2];
            data[i2] = w;
        }
        if (data[i1] < data[i0]) {
            final double x = data[i1];
            data[i1] = data[i0];
            data[i0] = x;
        }

        if (data[i5] < data[i3]) {
            final double u = data[i5];
            data[i5] = data[i3];
            data[i3] = u;
        }
        if (data[i4] < data[i2]) {
            final double v = data[i4];
            data[i4] = data[i2];
            data[i2] = v;
        }

        if (data[i6] < data[i3]) {
            final double u = data[i6];
            data[i6] = data[i3];
            data[i3] = u;
        }
        if (data[i4] < data[i1]) {
            final double v = data[i4];
            data[i4] = data[i1];
            data[i1] = v;
        }

        if (data[i6] < data[i5]) {
            final double u = data[i6];
            data[i6] = data[i5];
            data[i5] = u;
        }
        if (data[i4] < data[i3]) {
            final double v = data[i4];
            data[i4] = data[i3];
            data[i3] = v;
        }
        if (data[i2] < data[i1]) {
            final double w = data[i2];
            data[i2] = data[i1];
            data[i1] = w;
        }
    }

    /**
     * Sorts an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>This method is fast up to approximately 40 - 80 values.
     *
     * <p>The {@code internal} flag indicates that the value at {@code data[begin - 1]}
     * is sorted.
     *
     * @param data Data array.
     * @param begin Lower bound (inclusive).
     * @param end Upper bound (inclusive).
     * @param internal Internal flag.
     */
    static void sort(int[] data, int begin, int end, boolean internal) {
        int j;
        if (internal) {
            // Assume data[begin - 1] is a pivot and acts as a sentinal on the range.
            // => no requirement to check j >= begin.
            for (int i = begin; ++i <= end;) {
                final int v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; v < data[--j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        } else {
            for (int i = begin; ++i <= end;) {
                final int v = data[i];
                // Move preceding higher elements above
                if (v < data[i - 1]) {
                    for (j = i; --j >= begin && v < data[j];) {
                        data[j + 1] = data[j];
                    }
                    data[j + 1] = v;
                }
            }
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c
     * data[a] < data[b] < data[c]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     */
    static void sort3(int[] data, int i0, int i1, int i2) {
        // Order pair:
        //[(0,2)]
        // Move point 1 above point 2 or below point 0
        if (data[i2] < data[i0]) {
            final int v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }
        if (data[i2] < data[i1]) {
            final int v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        } else if (data[i1] < data[i0]) {
            final int v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }
    }

    /**
     * Sorts the given indices in an array using an insertion sort.
     *
     * <p>Note: Requires that the range contains no NaN values. It does not respect the
     * order of signed zeros.
     *
     * <p>Assumes all indices are valid and distinct.
     *
     * <p>Data are arranged such that:
     * <pre>{@code
     * a != b != c != d != e
     * data[a] < data[b] < data[c] < data[d] < data[e]
     * }</pre>
     *
     * <p>If indices are duplicated elements will <em>not</em> be correctly ordered.
     * However in this case data will contain the same values and may be partially ordered.
     *
     * @param data Data array.
     * @param i0 Index.
     * @param i1 Index.
     * @param i2 Index.
     * @param i3 Index.
     * @param i4 Index.
     */
    static void sort5(int[] data, int i0, int i1, int i2, int i3, int i4) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // Order pairs:
        //[(0,3),(1,4)]
        //[(0,2),(1,3)]
        //[(0,1),(2,4)]
        //[(1,2),(3,4)]
        //[(2,3)]
        if (data[i4] < data[i1]) {
            final int u = data[i4];
            data[i4] = data[i1];
            data[i1] = u;
        }
        if (data[i3] < data[i0]) {
            final int v = data[i3];
            data[i3] = data[i0];
            data[i0] = v;
        }

        if (data[i3] < data[i1]) {
            final int u = data[i3];
            data[i3] = data[i1];
            data[i1] = u;
        }
        if (data[i2] < data[i0]) {
            final int v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i2]) {
            final int u = data[i4];
            data[i4] = data[i2];
            data[i2] = u;
        }
        if (data[i1] < data[i0]) {
            final int v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i3]) {
            final int u = data[i4];
            data[i4] = data[i3];
            data[i3] = u;
        }
        if (data[i2] < data[i1]) {
            final int v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        }

        if (data[i3] < data[i2]) {
            final int u = data[i3];
            data[i3] = data[i2];
            data[i2] = u;
        }
    }

    /**
     * Sort the unique indices in-place to the start of the array. Duplicates are moved
     * to the end of the array and set to negative. For convenience the maximum
     * index is set into the final position in the array. If this is a duplicate it is
     * set to negative using the twos complement representation:
     *
     * <pre>{@code
     * int[] indices = ...
     * IndexSet sortUnique(indices);
     * int min = indices[0];
     * int max = indices[indices.length - 1]
     * if (max < 0) {
     *     max = ~max;
     * }
     * }</pre>
     *
     * <p>A small number of indices is sorted in place. A large number will use an
     * IndexSet which is returned for reuse by the caller. The threshold for this
     * switch is provided by the caller. An index set is used when
     * {@code indices.length > countThreshold} and there is more than 1 index.
     *
     * <p>This method assumes the {@code data} contains only positive integers.
     *
     * @param countThreshold Threshold to use an IndexSet.
     * @param data Indices.
     * @param n Number of indices.
     * @return the index set (or null if not used)
     */
    static IndexSet sortUnique(int countThreshold, int[] data, int n) {
        if (n <= 1) {
            return null;
        }
        if (n > countThreshold) {
            return sortUnique(data, n);
        }
        int unique = 1;
        int j;
        // Do an insertion sort but only compare the current set of unique values.
        for (int i = 0; ++i < n;) {
            final int v = data[i];
            // Erase data
            data[i] = -1;
            j = unique;
            if (v > data[j - 1]) {
                // Insert at end
                data[j] = v;
                unique++;
            } else if (v < data[j - 1]) {
                // Find insertion point in the unique indices
                do {
                    --j;
                } while (j >= 0 && v < data[j]);
                // Either insert at the start, or insert non-duplicate
                if (j < 0 || v != data[j]) {
                    // Update j so it is the insertion position
                    j++;
                    // Process the delayed moves
                    // Move from [j, unique) to [j+1, unique+1)
                    // System.arraycopy(data, j, data, j + 1, unique - j)
                    for (int k = unique; k-- > j;) {
                        data[k + 1] = data[k];
                    }
                    data[j] = v;
                    unique++;
                }
            }
        }
        // TODO: Is it faster to sort then compress.
        // The insert loop above only has an advantage when there are a lot of duplicates
        // as the list to compare against gets smaller as the sort continues.
//        java.util.Arrays.sort(data);
//        // Compress to remove duplicates
//        int i = 0;
//        OUTER:
//        while (++i < n) {
//            final int previous = data[i - 1];
//            while (data[i] == previous) {
//                data[i] = -1;
//                if (++i == n) {
//                    break OUTER;
//                }
//            }
//            data[unique++] = data[i];
//        }
        // Set the max value at the end, bit flipped
        if (unique < n) {
            data[n - 1] = ~data[unique - 1];
        }
        return null;
    }

    /**
     * Sort the unique indices in-place to the start of the array. Duplicates are moved
     * to the end of the array and set to negative. For convenience the maximum
     * index is set into the final position in the array. If this is a duplicate it is
     * set to negative using the twos complement representation:
     *
     * <pre>{@code
     * int[] indices = ...
     * IndexSet sortUnique(indices);
     * int min = indices[0];
     * int max = indices[indices.length - 1]
     * if (max < 0) {
     *     max = ~max;
     * }
     * }</pre>
     *
     * <p>Uses an IndexSet which is returned to the caller. Assumes the indices
     * are non-zero in length.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the index set
     */
    private static IndexSet sortUnique(int[] data, int n) {
        final IndexSet set = IndexSet.of(data, n);
        // Iterate
        final int[] unique = {0};
        set.forEach(i -> data[unique[0]++] = i);
        if (unique[0] < n) {
            for (int i = unique[0]; i < n; i++) {
                data[i] = -1;
            }
            // Set the max value at the end, bit flipped
            data[n - 1] = ~data[unique[0] - 1];
        }
        return set;
    }

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <pre>{@code
     * int[] indices = ...
     * int n sortIndices(indices, indices.length);
     * int min = indices[0];
     * int max = indices[n - 1]
     * }</pre>
     *
     * <p>This method assumes the {@code data} contains only positive integers.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndices(int[] data, int n) {
        // Simple cases
        if (n < 3) {
            if (n == 2) {
                final int i0 = data[0];
                final int i1 = data[1];
                if (i0 > i1) {
                    data[0] = i1;
                    data[1] = i0;
                } else if (i0 == i1) {
                    return 1;
                }
            }
            // n=0,1,2 unique values
            return n;
        }

        // Strategy: Must be fast on already ascending data.
        // Note: The recommended way to generate a lot of partition indices from
        // many quantiles for interpolation is to generate in sequence.

        // n <= small:
        //   Modified insertion sort (naturally finds ascending data)
        // n > small:
        //   Look for ascending sequence and compact
        // else:
        //   Remove duplicates using an order(1) data structure and sort

        if (n <= UNIQUE_INSERTION_SORT) {
            return sortIndicesInsertionSort(data, n);
        }

        if (isAscending(data, n)) {
            return compressDuplicates(data, n);
        }

        // At least 20 indices that are partially unordered.
        // Find min/max
        int min = data[0];
        int max = min;
        for (int i = 0; ++i < n;) {
            min = Math.min(min, data[i]);
            max = Math.max(max, data[i]);
        }

        // Benchmarking shows the IndexSet is very fast when the long[] efficiently
        // resides in cache memory. If the indices are very well separated the
        // distribution is sparse and it is faster to use a HashIndexSet despite
        // having to perform a sort after making keys unique.
        // Both structures have Order(1) detection of unique keys (the HashIndexSet
        // is configured with a load factor that should see low collision rates).
        // IndexSet     sort Order(n)        (data is stored sorted and must be read)
        // HashIndexSet sort Order(n log n)  (unique data is sorted separately)

        // For now base the choice on memory consumption alone which is a fair
        // approximation when n < 1000.
        // Above 1000 indices we assume that sorting the indices is a small cost
        // compared to sorting/partitioning the data that requires so many indices.
        // If the input data is small upstream code should detect this, e.g.
        // indices.length >> data.length, and choose to sort the data rather than
        // partitioning so many indices.

        // If the HashIndexSet uses < 50% memory of IndexSet then prefer that.
        // This detects obvious cases of sparse keys where the IndexSet is
        // outperformed by the HashIndexSet. Otherwise we can assume the
        // memory consumption of the IndexSet is small compared to the data to be
        // partitioned at these target indices (max 1/64 for double[] data); any
        // time taken here for sorting indices should be less than partitioning time.

        // TODO:
        // Requires more analysis of performance crossover.
        // Note: Expected behaviour under extreme use-cases should be documented.

        if (HashIndexSet.memoryFootprint(n) < (IndexSet.memoryFootprint(min, max) >>> 1)) {
            return sortIndicesHashIndexSet(data, n);
        }

        // Repeat code from sortIndicesIndexSet as we have the min/max
        final IndexSet set = IndexSet.ofRange(min, max);
        for (int i = -1; ++i < n;) {
            set.set(data[i]);
        }
        return set.toArray(data);
    }

    /**
     * Test the data is in ascending order: {@code data[i] <= data[i+1]}  for all {@code i}.
     * Data is assumed to be at least length 1.
     *
     * @param data Data.
     * @param n Length of data.
     * @return true if ascending
     */
    private static boolean isAscending(int[] data, int n) {
        int v = data[0];
        for (int i = 0; ++i < n;) {
            if (data[i] < v) {
                // descending
                return false;
            }
            v = data[i];
        }
        return true;
    }

    // The following methods all perform the same function and are present
    // for performance testing.

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <p>Uses an insertion sort modified to ignore duplicates.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndicesInsertionSort(int[] data, int n) {
        // Insert min at start to act as a sentinal
        int min = data[0];
        int mini = 0;
        for (int i = 0; ++i < n;) {
            final int v = data[i];
            if (v < min) {
                min = v;
                mini = i;
            }
        }
        data[mini] = data[0];
        data[0] = min;

        int unique = 1;
        int j;
        // Do an insertion sort but only compare the current set of unique values.
        for (int i = 0; ++i < n;) {
            final int v = data[i];
            j = unique - 1;
            if (v > data[j]) {
                // Insert at end
                data[j + 1] = v;
                unique++;
            } else if (v < data[j]) {
                // Find insertion point in the unique indices
                // Cannot move past the sentinal at data[0]
                do {
                    --j;
                } while (v < data[j]);
                // Only insert non-duplicate
                if (v != data[j]) {
                    // Update j so it is the insertion position
                    j++;
                    // Process the delayed moves
                    // Move from [j, unique) to [j+1, unique+1)
                    // System.arraycopy(data, j, data, j + 1, unique - j)
                    for (int k = unique; k-- > j;) {
                        data[k + 1] = data[k];
                    }
                    data[j] = v;
                    unique++;
                }
            }
        }
        return unique;
    }

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <p>Uses a heap sort modified to ignore duplicates.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndicesHeapSort(int[] data, int n) {
        // Build the min heap using Floyd's heap-construction algorithm
        // Start at parent of the last element in the heap (n-1)
        final int offset = n - 1;
        for (int start = offset >> 1; start >= 0; start--) {
            minHeapSiftDown(data, offset, start, n);
        }

        // The min heap has been constructed in-place so a[n-1] is the min.
        // To sort we have to move elements from the top of the
        // heap to the position immediately before the end of the heap
        // (which is below right), reducing the heap size each step:
        //                             root
        // |--------------|k|-min-heap-|r|
        //                 |  <-swap->  |

        // Move top of heap to the sorted end and move the end
        // to the top.
        int previous = data[offset];
        data[offset] = data[0];
        data[0] = previous;
        int s = n - 1;
        minHeapSiftDown(data, offset, 0, s);

        // Min heap is now 1 smaller
        // Proceed with the remaining elements but do not write them
        // to the sorted data unless different from the previous value.
        int unique = 1;
        for (;;) {
            s--;
            // Move top of heap to the sorted end
            final int v = data[offset];
            data[offset] = data[offset - s];
            if (previous != v) {
                data[unique++] = v;
                previous = v;
            }
            if (s == 1) {
                // end of heap
                break;
            }
            minHeapSiftDown(data, offset, 0, s);
        }
        // Stopped sifting when the heap was size 1.
        // Move the last (max) value to the sorted data.
        if (previous != data[offset]) {
            data[unique++] = data[offset];
        }
        return unique;
    }

    /**
     * Sift the top element down the min heap.
     *
     * <p>Note this creates the min heap in descending sequence so the
     * heap is positioned below the root.
     *
     * @param a Heap data.
     * @param offset Offset of the heap in the data.
     * @param root Root of the heap.
     * @param n Size of the heap.
     */
    private static void minHeapSiftDown(int[] a, int offset, int root, int n) {
        // For node i:
        // left child: 2i + 1
        // right child: 2i + 2
        // parent: floor((i-1) / 2)

        // Value to sift
        int p = root;
        final int v = a[offset - p];
        // Left child of root: p * 2 + 1
        int c = (p << 1) + 1;
        while (c < n) {
            // Left child value
            int cv = a[offset - c];
            // Use the right child if less
            if (c + 1 < n && cv > a[offset - c - 1]) {
                cv = a[offset - c - 1];
                c++;
            }
            // Min heap requires parent <= child
            if (v <= cv) {
                // Less than smallest child - done
                break;
            }
            // Swap and descend
            a[offset - p] = cv;
            p = c;
            c = (p << 1) + 1;
        }
        a[offset - p] = v;
    }

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <p>Uses a full sort and a second-pass to ignore duplicates.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndicesSort(int[] data, int n) {
        java.util.Arrays.sort(data, 0, n);
        return compressDuplicates(data, n);
    }

    /**
     * Compress duplicates in the ascending data.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of unique indices
     */
    private static int compressDuplicates(int[] data, int n) {
        // Compress to remove duplicates
        int i = 0;
        int unique = 1;
        int previous = data[0];
        OUTER:
        while (++i < n) {
            while (data[i] == previous) {
                if (++i == n) {
                    break OUTER;
                }
            }
            previous = data[i];
            data[unique++] = previous;
        }
        return unique;
    }

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <p>Uses an {@link IndexSet} to ignore duplicates. The sorted array is
     * extracted from the {@link IndexSet} storage in order.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndicesIndexSet(int[] data, int n) {
        // Delegate to IndexSet
        // Storage (bytes) = 8 * ceil((max - min) / 64), irrespective of n.
        // This can be use a lot of memory when the indices are spread out.
        return IndexSet.of(data, n).toArray(data);
    }

    /**
     * Sort the unique indices in-place to the start of the array. The number of
     * indices is returned.
     *
     * <p>Uses a {@link HashIndexSet} to ignore duplicates and then performs
     * a full sort of the unique values.
     *
     * <p>Warning: Requires {@code n > 0}.
     *
     * @param data Indices.
     * @param n Number of indices.
     * @return the number of indices
     */
    static int sortIndicesHashIndexSet(int[] data, int n) {
        // Compress to remove duplicates.
        // Duplicates are checked using a HashIndexSet.
        // Storage (bytes) = 4 * next-power-of-2(n*2) => 2-4 times n
        final HashIndexSet set = new HashIndexSet(n);
        int i = 0;
        int unique = 1;
        set.add(data[0]);
        while (++i < n) {
            final int v = data[i];
            if (set.add(v)) {
                data[unique++] = v;
            }
        }
        // Sort unique data.
        // This can exploit the input already being sorted.
        Arrays.sort(data, 0, unique);
        return unique;
    }
}
