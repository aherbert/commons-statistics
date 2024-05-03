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

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.AdaptMode;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.EdgeSelectStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.ExpandStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.KeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.LinearStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.PairedKeyStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.SPStrategy;
import org.apache.commons.statistics.examples.jmh.descriptive.Partition.StopperStrategy;

/**
 * Create instances of partition algorithms.
 *
 * @see Partition
 * @see KthSelector
 */
final class PartitionFactory {
    /** Pattern for the minimum quickselect size. */
    private static final Pattern QS_PATTERN = Pattern.compile("QS(\\d+)");
    /** Pattern for the edgeselect constant. */
    private static final Pattern EC_PATTERN = Pattern.compile("EC(\\d+)");
    /** Pattern for the edgeselect constant for linear select. */
    private static final Pattern LC_PATTERN = Pattern.compile("LC(\\d+)");
    /** Pattern for the sub-sampling size. */
    private static final Pattern SU_PATTERN = Pattern.compile("SU(\\d+)");
    /** Pattern for the recursion multiple (simple float format). */
    private static final Pattern RM_PATTERN = Pattern.compile("RM(\\d+\\.?\\d*)");
    /** Pattern for the recursion constant. */
    private static final Pattern RC_PATTERN = Pattern.compile("RC(\\d+)");
    /** Pattern for the compression level. */
    private static final Pattern CL_PATTERN = Pattern.compile("CL(\\d+)");
    /** Pattern for the control flags. Allow negative flags. */
    private static final Pattern CF_PATTERN = Pattern.compile("CF(-?\\d+)");

    /** No instances. */
    private PartitionFactory() {}

    /**
     * Creates the {@link KthSelector}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link KthSelector} instance
     */
    static KthSelector createKthSelector(String name, String prefix) {
        return createKthSelector(name, prefix, 0);
    }

    /**
     * Creates the {@link KthSelector}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @param qs Minimum quickselect size (if non-zero).
     * @return the {@link KthSelector} instance
     */
    static KthSelector createKthSelector(String name, String prefix, int qs) {
        final String[] s = {name};
        final int minQuickSelectSize = qs != 0 ? qs : getMinQuickSelectSize(s);
        final PivotingStrategy sp = getEnumOrElse(s, PivotingStrategy.class, Partition.PIVOTING_STRATEGY);
        // Check for unharvested parameters
        for (int i = prefix.length(); i < s[0].length(); i++) {
            if (s[0].charAt(i) != '_') {
                throw new IllegalStateException(
                    String.format("Unharvested KthSelector parameters: %s -> %s", name, s[0]));
            }
        }
        return new KthSelector(sp, minQuickSelectSize);
    }

    /**
     * Creates the {@link Partition}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @return the {@link Partition} instance
     */
    static Partition createPartition(String name, String prefix) {
        return createPartition(name, prefix, 0, 0);
    }

    /**
     * Creates the {@link Partition}. Parameters are derived from the {@code name}.
     *
     * <p>After parameters are harvested the only allowed characters are underscores,
     * otherwise an exception is thrown. This ensures the parameters in the name were
     * correct.
     *
     * @param name Name.
     * @param prefix Method prefix.
     * @param qs Minimum quickselect size (if non-zero).
     * @param ec Minimum edgeselect constant (if non-zero); also used for linear sort select size.
     * @return the {@link Partition} instance
     */
    static Partition createPartition(String name, String prefix, int qs, int ec) {
        if (!name.startsWith(prefix)) {
            throw new IllegalArgumentException("Invalid prefix: " + prefix + " for " + name);
        }
        final String[] s = {name.substring(prefix.length())};
        final PivotingStrategy sp = getEnumOrElse(s, PivotingStrategy.class, Partition.PIVOTING_STRATEGY);
        final DualPivotingStrategy dp = getEnumOrElse(s, DualPivotingStrategy.class, Partition.DUAL_PIVOTING_STRATEGY);
        final int minQuickSelectSize = qs != 0 ? qs : getMinQuickSelectSize(s);
        final int edgeSelectConstant = ec != 0 ? ec : getEdgeSelectConstant(s);
        final int linearSortSelectConstant = ec != 0 ? ec : getLinearSortSelectConstant(s);
        final int subSamplingSize = getSubSamplingSize(s);
        final KeyStrategy keyStartegy = getEnumOrElse(s, KeyStrategy.class, Partition.KEY_STRATEGY);
        final PairedKeyStrategy pairedKeyStartegy =
            getEnumOrElse(s, PairedKeyStrategy.class, Partition.PAIRED_KEY_STRATEGY);
        final double recursionMultiple = getRecursionMultiple(s);
        final int recursionConstant = getRecursionConstant(s);
        final int compressionLevel = getCompressionLevel(s);
        final int controlFlags = getControlFlags(s);
        final SPStrategy spStrategy = getEnumOrElse(s, SPStrategy.class, Partition.SP_STRATEGY);
        final ExpandStrategy expandStrategy = getEnumOrElse(s, ExpandStrategy.class, Partition.EXPAND_STRATEGY);
        final LinearStrategy linearStrategy = getEnumOrElse(s, LinearStrategy.class, Partition.LINEAR_STRATEGY);
        final EdgeSelectStrategy esStrategy = getEnumOrElse(s, EdgeSelectStrategy.class, Partition.EDGE_STRATEGY);
        final StopperStrategy stopStrategy = getEnumOrElse(s, StopperStrategy.class, Partition.STOPPER_STRATEGY);
        final AdaptMode adaptMode = getEnumOrElse(s, AdaptMode.class, Partition.ADAPT_MODE);
        // Check for unharvested parameters
        for (int i = s[0].length(); --i >= 0;) {
            if (s[0].charAt(i) != '_') {
                throw new IllegalStateException(
                    String.format("Unharvested Partition parameters: %s -> %s", name, prefix + s[0]));
            }
        }
        final Partition p = new Partition(sp, dp, minQuickSelectSize,
            edgeSelectConstant, subSamplingSize);
        // Some values do not have to be final as they are not used within optimised
        // partitioning code.
        p.setKeyStrategy(keyStartegy);
        p.setPairedKeyStrategy(pairedKeyStartegy);
        p.setRecursionMultiple(recursionMultiple);
        p.setRecursionConstant(recursionConstant);
        p.setCompression(compressionLevel);
        p.setControlFlags(controlFlags);
        p.setSPStrategy(spStrategy);
        p.setExpandStrategy(expandStrategy);
        p.setLinearStrategy(linearStrategy);
        p.setEdgeSelectStrategy(esStrategy);
        p.setStopperStrategy(stopStrategy);
        p.setLinearSortSelectSize(linearSortSelectConstant);
        p.setAdaptMode(adaptMode);

        return p;
    }

    /**
     * Gets the minimum size for the recursive quickselect partition algorithm.
     * Below this size the algorithm will change strategy for partitioning,
     * e.g. change to a full sort.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the minimum quickselect size
     */
    static int getMinQuickSelectSize(String[] name) {
        final Matcher m = QS_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.MIN_QUICKSELECT_SIZE;
    }

    /**
     * Gets the constant for the edgeselect distance-from-end computation.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the edgeselect constant
     */
    static int getEdgeSelectConstant(String[] name) {
        final Matcher m = EC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.EDGESELECT_CONSTANT;
    }

    /**
     * Gets the constant for the sortselect distance-from-end computation for linearselect.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the sortselect constant
     */
    static int getLinearSortSelectConstant(String[] name) {
        final Matcher m = LC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.LINEAR_SORTSELECT_SIZE;
    }

    /**
     * Gets the minimum size for single-pivot sub-sampling.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the sub-sampling size
     */
    static int getSubSamplingSize(String[] name) {
        final Matcher m = SU_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.SUBSAMPLING_SIZE;
    }

    /**
     * Gets the recursion multiplication factor.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the recursion multiple
     */
    static double getRecursionMultiple(String[] name) {
        final Matcher m = RM_PATTERN.matcher(name[0]);
        if (m.find()) {
            final double d = Double.parseDouble(m.group(1));
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return d;
        }
        return Partition.RECURSION_MULTIPLE;
    }

    /**
     * Gets the recursion constant.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the recursion constant
     */
    static int getRecursionConstant(String[] name) {
        final Matcher m = RC_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.RECURSION_CONSTANT;
    }

    /**
     * Gets the compression level for {@link CompressedIndexSet}.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the compression
     */
    static int getCompressionLevel(String[] name) {
        final Matcher m = CL_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return Partition.COMPRESSION_LEVEL;
    }

    /**
     * Gets the control flags. These are used to enable additional features, for example
     * random sampling in the Floyd-Rivest algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @return the control flags
     */
    static int getControlFlags(String[] name) {
        return getControlFlags(name, Partition.CONTROL_FLAGS);
    }

    /**
     * Gets the control flags. These are used to enable additional features, for example
     * random sampling in the Floyd-Rivest algorithm.
     *
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param defaultValue Default value.
     * @return the control flags
     */
    static int getControlFlags(String[] name, int defaultValue) {
        final Matcher m = CF_PATTERN.matcher(name[0]);
        if (m.find()) {
            final int i = Integer.parseInt(name[0], m.start(1), m.end(1), 10);
            name[0] = name[0].substring(0, m.start()) + name[0].substring(m.end(), name[0].length());
            return i;
        }
        return defaultValue;
    }

    /**
     * Gets the enum from the name. The enum name must be prefixed with an underscore.
     *
     * @param <E> Enum type.
     * @param name Algorithm name (updated in-place to remove the parameter).
     * @param cls Class of the enum.
     * @param defaultValue Default value.
     * @return the enum value
     */
    static <E extends Enum<E>> E getEnumOrElse(String[] name, Class<E> cls, E defaultValue) {
        // Names can have partial matches. Match the longest name
        int index = -1;
        int len = 0;
        E result = defaultValue;
        for (final E s : cls.getEnumConstants()) {
            // Use the index so we can mandate that the enum is prefixed by underscore
            final int i = name[0].indexOf(s.name());
            if ((i > 0 && name[0].charAt(i - 1) == '_' || i == 0) && s.name().length() > len) {
                index = i;
                len = s.name().length();
                result = s;
            }
        }
        if (index >= 0) {
            name[0] = name[0].substring(0, index) + name[0].substring(index + len);
        }
        return result;
    }
}
