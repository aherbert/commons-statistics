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
     * @param i2 Index.
     * @param i1 Index.
     * @param i0 Index.
     */
    static void sort3(double[] data, int i0, int i1, int i2) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // Order pairs:
        //[(0,2)]
        //[(0,1)]
        //[(1,2)]
        double v;
        if (data[i2] < data[i0]) {
            v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }
        if (data[i2] < data[i1]) {
            v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        } else if (data[i1] < data[i0]) {
            v = data[i1];
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
     * @param i4 Index.
     * @param i3 Index.
     * @param i2 Index.
     * @param i1 Index.
     * @param i0 Index.
     */
    static void sort5(double[] data, int i0, int i1, int i2, int i3, int i4) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // Order pairs:
        //[(0,3),(1,4)]
        //[(0,2),(1,3)]
        //[(0,1),(2,4)]
        //[(1,2),(3,4)]
        //[(2,3)]
        double u;
        double v;
        if (data[i4] < data[i1]) {
            u = data[i4];
            data[i4] = data[i1];
            data[i1] = u;
        }
        if (data[i3] < data[i0]) {
            v = data[i3];
            data[i3] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i2]) {
            u = data[i4];
            data[i4] = data[i2];
            data[i2] = u;
        }
        if (data[i3] < data[i1]) {
            v = data[i3];
            data[i3] = data[i1];
            data[i1] = v;
        }

        if (data[i4] < data[i3]) {
            u = data[i4];
            data[i4] = data[i3];
            data[i3] = u;
        }
        if (data[i2] < data[i0]) {
            v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }

        if (data[i3] < data[i2]) {
            u = data[i3];
            data[i3] = data[i2];
            data[i2] = u;
        }
        if (data[i1] < data[i0]) {
            v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }

        if (data[i2] < data[i1]) {
            u = data[i2];
            data[i2] = data[i1];
            data[i1] = u;
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
     * @param i2 Index.
     * @param i1 Index.
     * @param i0 Index.
     */
    static void sort3(int[] data, int i0, int i1, int i2) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // Order pairs:
        //[(0,2)]
        //[(0,1)]
        //[(1,2)]
        int v;
        if (data[i2] < data[i0]) {
            v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }
        if (data[i2] < data[i1]) {
            v = data[i2];
            data[i2] = data[i1];
            data[i1] = v;
        } else if (data[i1] < data[i0]) {
            v = data[i1];
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
     * @param i4 Index.
     * @param i3 Index.
     * @param i2 Index.
     * @param i1 Index.
     * @param i0 Index.
     */
    static void sort5(int[] data, int i0, int i1, int i2, int i3, int i4) {
        // Uses an optimal sorting network from Knuth's Art of Computer Programming.
        // Order pairs:
        //[(0,3),(1,4)]
        //[(0,2),(1,3)]
        //[(0,1),(2,4)]
        //[(1,2),(3,4)]
        //[(2,3)]
        int u;
        int v;
        if (data[i4] < data[i1]) {
            u = data[i4];
            data[i4] = data[i1];
            data[i1] = u;
        }
        if (data[i3] < data[i0]) {
            v = data[i3];
            data[i3] = data[i0];
            data[i0] = v;
        }

        if (data[i4] < data[i2]) {
            u = data[i4];
            data[i4] = data[i2];
            data[i2] = u;
        }
        if (data[i3] < data[i1]) {
            v = data[i3];
            data[i3] = data[i1];
            data[i1] = v;
        }

        if (data[i4] < data[i3]) {
            u = data[i4];
            data[i4] = data[i3];
            data[i3] = u;
        }
        if (data[i2] < data[i0]) {
            v = data[i2];
            data[i2] = data[i0];
            data[i0] = v;
        }

        if (data[i3] < data[i2]) {
            u = data[i3];
            data[i3] = data[i2];
            data[i2] = u;
        }
        if (data[i1] < data[i0]) {
            v = data[i1];
            data[i1] = data[i0];
            data[i0] = v;
        }

        if (data[i2] < data[i1]) {
            u = data[i2];
            data[i2] = data[i1];
            data[i1] = u;
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
}
