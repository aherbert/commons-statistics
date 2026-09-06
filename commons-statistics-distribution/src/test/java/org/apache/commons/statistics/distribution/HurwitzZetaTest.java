/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.statistics.distribution;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.Stream;
import org.apache.commons.numbers.core.DD;
import org.apache.commons.numbers.fraction.BigFraction;
import org.apache.commons.statistics.distribution.ExtendedPrecisionTest.RMS;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvFileSource;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test the {@link HurwitzZeta} function.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class HurwitzZetaTest {
    private static final int[] M = new int[53];

    /**
     * Numerators of the even Bernoulli numbers {@code B_{2k}}.
     * Taken from:
     * A000367 Numerators of Bernoulli numbers B_2n.
     * https://oeis.org/A164020/b164020.txt
     *
     * Contains the sequence up to 2k = 106 required for M=53 in the zeta implementation.
     * Johansson (2015) suggests N ~ M ~ P for P-bits of precision.
     */
    private static final String[] NUM = {
        "0 1",
        "1 1",
        "2 -1",
        "3 1",
        "4 -1",
        "5 5",
        "6 -691",
        "7 7",
        "8 -3617",
        "9 43867",
        "10 -174611",
        "11 854513",
        "12 -236364091",
        "13 8553103",
        "14 -23749461029",
        "15 8615841276005",
        "16 -7709321041217",
        "17 2577687858367",
        "18 -26315271553053477373",
        "19 2929993913841559",
        "20 -261082718496449122051",
        "21 1520097643918070802691",
        "22 -27833269579301024235023",
        "23 596451111593912163277961",
        "24 -5609403368997817686249127547",
        "25 495057205241079648212477525",
        "26 -801165718135489957347924991853",
        "27 29149963634884862421418123812691",
        "28 -2479392929313226753685415739663229",
        "29 84483613348880041862046775994036021",
        "30 -1215233140483755572040304994079820246041491",
        "31 12300585434086858541953039857403386151",
        "32 -106783830147866529886385444979142647942017",
        "33 1472600022126335654051619428551932342241899101",
        "34 -78773130858718728141909149208474606244347001",
        "35 1505381347333367003803076567377857208511438160235",
        "36 -5827954961669944110438277244641067365282488301844260429",
        "37 34152417289221168014330073731472635186688307783087",
        "38 -24655088825935372707687196040585199904365267828865801",
        "39 414846365575400828295179035549542073492199375372400483487",
        "40 -4603784299479457646935574969019046849794257872751288919656867",
        "41 1677014149185145836823154509786269900207736027570253414881613",
        "42 -2024576195935290360231131160111731009989917391198090877281083932477",
        "43 660714619417678653573847847426261496277830686653388931761996983",
        "44 -1311426488674017507995511424019311843345750275572028644296919890574047",
        "45 1179057279021082799884123351249215083775254949669647116231545215727922535",
        "46 -1295585948207537527989427828538576749659341483719435143023316326829946247",
        "47 1220813806579744469607301679413201203958508415202696621436215105284649447",
        "48 -211600449597266513097597728109824233673043954389060234150638733420050668349987259",
        "49 67908260672905495624051117546403605607342195728504487509073961249992947058239",
        "50 -94598037819122125295227433069493721872702841533066936133385696204311395415197247711",
        "51 3204019410860907078243020782116241775491817197152717450679002501086861530836678158791",
        "52 -319533631363830011287103352796174274671189606078272738327103470162849568365549721224053",
        "53 36373903172617414408151820151593427169231298640581690038930816378281879873386202346572901",
    };

    /**
     * Denomintators of the even Bernoulli numbers {@code B_{2k}}.
     * Taken from:
     * A002445 Denominators of Bernoulli numbers B_{2n}.
     * https://oeis.org/A002445/b002445.txt
     */
    private static final String[] DENOM = {
        "0 1",
        "1 6",
        "2 30",
        "3 42",
        "4 30",
        "5 66",
        "6 2730",
        "7 6",
        "8 510",
        "9 798",
        "10 330",
        "11 138",
        "12 2730",
        "13 6",
        "14 870",
        "15 14322",
        "16 510",
        "17 6",
        "18 1919190",
        "19 6",
        "20 13530",
        "21 1806",
        "22 690",
        "23 282",
        "24 46410",
        "25 66",
        "26 1590",
        "27 798",
        "28 870",
        "29 354",
        "30 56786730",
        "31 6",
        "32 510",
        "33 64722",
        "34 30",
        "35 4686",
        "36 140100870",
        "37 6",
        "38 30",
        "39 3318",
        "40 230010",
        "41 498",
        "42 3404310",
        "43 6",
        "44 61410",
        "45 272118",
        "46 1410",
        "47 6",
        "48 4501770",
        "49 6",
        "50 33330",
        "51 4326",
        "52 1590",
        "53 642",
    };

    /**
     * Precomputed factors for {@code k}-th element of the tail function {@code T}.
     * Uses {@code 2k!} divided by Bernoulli number {@code B_2k}.
     * The table size is suitable for N ~ M ~ P for P-bits of precision (53 entries)
     * as stated in Johansson (2015) section 3.1. In practice the result in double
     * precision requires lower N & M values.
     */
    private static final double[] F = {
        12.0, // 2! / (1 / 6)
        -720.0, // 4! / (-1 / 30)
        30240.0, // 6! / (1 / 42)
        -1209600.0, // 8! / (-1 / 30)
        4.790016E7, // 10! / (5 / 66)
        -1.8924375803183792E9, // 12! / (-691 / 2730)
        7.47242496E10, // 14! / (7 / 6)
        -2.950130727918164E12, // 16! / (-3617 / 510)
        1.1646782814350067E14, // 18! / (43867 / 798)
        -4.597978722407473E15, // 20! / (-174611 / 330)
        1.81521054019435456E17, // 22! / (854513 / 138)
        -7.1661652561756672E18, // 24! / (-236364091 / 2730)
        2.82908877253043E20, // 26! / (8553103 / 6)
        -1.1168794925000445E22, // 28! / (-23749461029 / 870)
        4.4092635141854666E23, // 30! / (8615841276005 / 14322)
        -1.7407074646225822E25, // 32! / (-7709321041217 / 510)
        6.872037622739274E26, // 34! / (2577687858367 / 6)
        -2.7129717107520044E28, // 36! / (-26315271553053477373 / 1919190)
        1.0710383014704457E30, // 38! / (2929993913841559 / 6)
        -4.228289733582729E31, // 40! / (-261082718496449122051 / 13530)
        1.669261878547101E33, // 42! / (1520097643918070802691 / 1806)
        -6.589981753232787E34, // 44! / (-27833269579301024235023 / 690)
        2.6016205165923062E36, // 46! / (596451111593912163277961 / 282)
        -1.0270786120209628E38, // 48! / (-5609403368997817686249127547 / 46410)
        4.054743835786749E39, // 50! / (495057205241079648212477525 / 66)
        -1.6007487042788347E41, // 52! / (-801165718135489957347924991853 / 1590)
        6.319502582715391E42, // 54! / (29149963634884862421418123812691 / 798)
        -2.4948396201225357E44, // 56! / (-2479392929313226753685415739663229 / 870)
        9.849232037909393E45, // 58! / (84483613348880041862046775994036021 / 354)
        -3.888320954748035E47, // 60! / (-1215233140483755572040304994079820246041491 / 56786730)
        1.535047584313167E49, // 62! / (12300585434086858541953039857403386151 / 6)
        -6.060124957607529E50, // 64! / (-106783830147866529886385444979142647942017 / 510)
        2.3924414381101893E52, // 66! / (1472600022126335654051619428551932342241899101 / 64722)
        -9.444980218768351E53, // 68! / (-78773130858718728141909149208474606244347001 / 30)
        3.728728733414322E55, // 70! / (1505381347333367003803076567377857208511438160235 / 4686)
        -1.4720431007109737E57, // 72! / (-5827954961669944110438277244641067365282488301844260429 / 140100870)
        5.811393226148101E58, // 74! / (34152417289221168014330073731472635186688307783087 / 6)
        -2.2942460864500874E60, // 76! / (-24655088825935372707687196040585199904365267828865801 / 30)
        9.057320508803927E61, // 78! / (414846365575400828295179035549542073492199375372400483487 / 3318)
        -3.575686814230726E63, // 80! / (-4603784299479457646935574969019046849794257872751288919656867 / 230010)
        1.4116245727459505E65, // 82! / (1677014149185145836823154509786269900207736027570253414881613 / 498)
        -5.5728704383437275E66, // 84! / (-2024576195935290360231131160111731009989917391198090877281083932477 / 3404310)
        2.2000810641991213E68, // 86! / (660714619417678653573847847426261496277830686653388931761996983 / 6)
        -8.685571901589203E69, // 88! / (-1311426488674017507995511424019311843345750275572028644296919890574047 / 61410)
        3.428926346636115E71, // 90! / (1179057279021082799884123351249215083775254949669647116231545215727922535 / 272118)
        -1.3536858624708422E73, // 92! / (-1295585948207537527989427828538576749659341483719435143023316326829946247 / 1410)
        5.344137578373867E74, // 94! / (1220813806579744469607301679413201203958508415202696621436215105284649447 / 6)
        -2.10978095054183E76, // 96! / (-211600449597266513097597728109824233673043954389060234150638733420050668349987259 / 4501770)
        8.329081341920855E77, // 98! / (67908260672905495624051117546403605607342195728504487509073961249992947058239 / 6)
        -3.288189514770133E79, // 100! / (-94598037819122125295227433069493721872702841533066936133385696204311395415197247711 / 33330)
        1.2981251882636475E81, // 102! / (3204019410860907078243020782116241775491817197152717450679002501086861530836678158791 / 4326)
        -5.124792828500739E82, // 104! / (-319533631363830011287103352796174274671189606078272738327103470162849568365549721224053 / 1590)
        2.023187114193683E84, // 106! / (36373903172617414408151820151593427169231298640581690038930816378281879873386202346572901 / 642)
    };

    private static final RMS RMS_ZETA1 = new RMS();
    private static final RMS RMS_ZETA2 = new RMS();
    private static final RMS RMS_ZETA3 = new RMS();
    private static final RMS RMS_ZETA4 = new RMS();
    private static final RMS RMS_ZETAC = new RMS();

    // Test implementations
    // Note: The method is sensitive to the initial loop over N to create S.
    // Under certain conditions the N cannot be too high if using an ascending
    // sum of k as the sum does not converge and later terms are added with
    // low precision.
    // Better results are obtained using descending k. However this prevents
    // an early exit if the series is rapidly converging and the term (a+k)^-s
    // drops below machine epsilon of the sum.

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(s, a)}.
     * See {@link HurwitzZeta} for the formula details.
     *
     * <p><strong>Warning</strong>: No parameter validation is performed.
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    static double zeta1(double s, double a, int n, int m) {
        double sum = 0;
        // S : k in [0, n-1]
        for (int k = n; --k >= 0;) {
            // Descending k sums in order of magnitude for increased precision.
            // Prevents early exit for large s when the term (a+k)^-s is below
            // machine epsilon of the ascending series sum.
            sum += Math.pow(a + k, -s);
        }
        final double apn = a + n;
        // I
        sum += Math.pow(apn, 1 - s) / (s - 1);
        // T
        sum += 0.5 * Math.pow(apn, -s);
        // Rising factorial term
        double f = s;
        // 2k - 1
        double k2 = 1;
        double tsum = 0;
        for (int i = 0; i < m; i++) {
            final double t = f * Math.pow(apn, -(k2 + s)) / F[i];
            tsum += t;
            if (tsum + t == tsum) {
                // additional terms too small
                break;
            }
            // f = s * (s+1) * (s+2) * ... * (s+2k-2)
            f *= s + k2;
            k2 += 1.0;
            f *= s + k2;
            k2 += 1.0;
        }
        return sum + tsum;
    }

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(s, a)}.
     * See {@link HurwitzZeta} for the formula details.
     *
     * <p><strong>Warning</strong>: No parameter validation is performed.
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    static double zeta2(double s, double a, int n, int m) {
        double sum = 0;
        // S : k in [0, n-1]
        for (int k = 0; k < n; k++) {
            // Allow early convergence
            final double t = Math.pow(a + k, -s);
            sum += t;
            if (sum + t == sum) {
                return sum;
            }
        }
        final double apn = a + n;
        // I
        sum += Math.pow(apn, 1 - s) / (s - 1);
        // T
        sum += 0.5 * Math.pow(apn, -s);
        // Rising factorial term
        double f = s;
        // 2k - 1
        double k2 = 1;
        double tsum = 0;
        for (int i = 0; i < m; i++) {
            final double t = f * Math.pow(apn, -(k2 + s)) / F[i];
            tsum += t;
            if (tsum + t == tsum) {
                // additional terms too small
                break;
            }
            // f = s * (s+1) * (s+2) * ... * (s+2k-2)
            f *= s + k2;
            k2 += 1.0;
            f *= s + k2;
            k2 += 1.0;
        }
        return sum + tsum;
    }

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(s, a)}.
     * See {@link HurwitzZeta} for the formula details.
     *
     * <p><strong>Warning</strong>: No parameter validation is performed.
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    static double zeta3(double s, double a, int n, int m) {
        final double apn = a + n;
        double p = Math.pow(apn, -s);

        // Initialise sum with the first tail term
        double sum = 0.5 * p;
        // S : k in [0, n-1]
        for (int k = n; --k >= 0;) {
            // Descending k sums in order of magnitude for increased precision.
            // Prevents early exit for large s when the term (a+k)^-s is below
            // machine epsilon of the ascending series sum.
            sum += Math.pow(a + k, -s);
        }

        // I
        sum += Math.pow(apn, 1 - s) / (s - 1);

        // T
        // The following recycles the power term p: (a+n)^-(2k-1+s).
        // This incorporates the factor for T into the sum terms.
        // This sets the first power as (a+n)^-(1+s) not (a+n)^-1.
        // When s is large the loop exits before the rising factorial overflows.

        // Rising factorial term.
        double f = s;
        // 2k - 1
        double k2 = 1;
        // Sum of an alternating series as each F changes sign.
        // Sum until terms will not impact the result (at least 10-bit non-overlap).
        double tsum = 0;
        final double stop = sum * 0x1p-63;
        int i;
        for (i = 0; i < m; i++) {
            // p = (a+n)^-(2k-1+s)
            p /= apn;
            // TODO - try this with the reciprocal of F
            final double t = f * p / F[i];
            tsum += t;
            if (Math.abs(t) <= stop) {
                break;
            }
            p /= apn;
            // f = s * (s+1) * (s+2) * ... * (s+2k-2)
            f *= s + k2;
            k2 += 1.0;
            f *= s + k2;
            k2 += 1.0;
        }
        //M[i]++;
        return sum + tsum;
    }

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(s, a)}.
     * See {@link HurwitzZeta} for the formula details.
     *
     * <p><strong>Warning</strong>: No parameter validation is performed.
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    // TODO - fix this or find out why it fails
    // Output the intermediates vs zeta3 for a failing case
    static double zeta4(double s, double a, int n, int m) {
        // S : k in [0, n-1]
        DD sum = DD.of(Math.pow(a, -s));
        for (int k = 1; k < n; k++) {
            // Allow early convergence
            final double t = Math.pow(a + k, -s);
            sum = sum.add(t);
            if (t * 0x1p-16 < sum.lo()) {
                // Rapidly converging
                return sum.doubleValue();
            }
        }
        final double apn = a + n;
        // I : (a+n)^(1-s) / (s-1)
        sum = sum.add(Math.pow(apn, 1 - s) / (s - 1));
        // T
        double p = Math.pow(apn, -s);
        //sum = sum.add(0.5 * p);

        // The following recycles the power term p: (a+n)^-(2k-1+s).
        // This incorporates the factor for T into the sum terms.
        // This sets the first power as (a+n)^-(1+s) not (a+n)^-1.
        // When s is large the loop exits before the rising factorial overflows.

        // Rising factorial term.
        double f = s;
        // 2k - 1
        double k2 = 1;
        double tsum = 0.5 * p;
        for (int i = 0; i < m; i++) {
            // p = (a+n)^-(2k-1+s)
            p /= apn;
            final double t = f * p / F[i];
            tsum += t;
            if (Math.abs(t) <= sum.hi() * 0x1p-63) {
                // additional terms too small
                break;
            }
            p /= apn;
            // f = s * (s+1) * (s+2) * ... * (s+2k-2)
            f *= s + k2;
            k2 += 1.0;
            f *= s + k2;
            k2 += 1.0;
        }
        return sum.add(tsum).doubleValue();
    }

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(x, q)}.
     *
     * @param x Argument {@code x > 1}
     * @param q Argument {@code q}
     * @return zeta(x, q)
     */
    // TODO - remove this
    static double zetaCephes(double x, double q) {
        int i;
        double a, b, k, s, t, w;

        /* Asymptotic expansion
         * https://dlmf.nist.gov/25.11#E43
         */
        if (q > 1e8) {
            return (1 / (x - 1) + 1 / (2 * q)) * Math.pow(q, 1 - x);
        }

        /* Euler-Maclaurin summation formula */

        /* Permit negative q but continue sum until n+q > +9 .
         * This case should be handled by a reflection formula.
         * If q<0 and x is an integer, there is a relation to
         * the polyGamma function.
         */
        s = Math.pow(q, -x);
        a = q;
        i = 0;
        b = 0.0;
        while ((i < 9) || (a <= 9.0)) {
            i += 1;
            a += 1.0;
            b = Math.pow(a, -x);
            s += b;
            // abs required for convergence of negative q ???
            if (Math.abs(b / s) < 0x1.0p-53)
                return s;
        }

        // w = q + n
        w = a;
        s += b * w / (x - 1.0);
        s -= 0.5 * b;
        a = 1.0;
        k = 0.0;
        for (i = 0; i < 12; i++) {
            a *= x + k;
            b /= w;
            t = a * b / F[i];
            s = s + t;
            t = Math.abs(t / s);
            if (t < 0x1.0p-53)
                return s;
            k += 1.0;
            a *= x + k;
            b /= w;
            k += 1.0;
        }
        return (s);
    }

    @Test
    void testFactors() {
        // Factorial of 2k. Initialise at k=0.
        BigInteger factorial = BigInteger.ONE;
        double sum1 = 0;
        double sum2 = 0;
        final int scale = 200;
        // Check factors
        for (int k = 1; k < NUM.length; k++) {
            factorial = factorial.multiply(BigInteger.valueOf(2 * k - 1)).multiply(BigInteger.valueOf(2 * k));
            final BigInteger num = new BigInteger(NUM[k].substring(NUM[k].indexOf(' ') + 1));
            final BigInteger denom = new BigInteger(DENOM[k].substring(DENOM[k].indexOf(' ') + 1));

            // 2k! / B_2k
            final BigFraction factor1 = BigFraction.of(factorial.multiply(denom), num);
            final double d1 = factor1.doubleValue();
            // Cross verify BigFraction vs BigDecimal
            BigDecimal v = factor1.bigDecimalValue(scale, RoundingMode.HALF_EVEN);
            Assertions.assertEquals(v.doubleValue(), d1);
            // Find ULP precision
            final double e1 = new BigDecimal(d1).subtract(v)
                .divide(new BigDecimal(Math.ulp(d1)), scale, RoundingMode.HALF_EVEN).doubleValue();
            sum1 += Math.abs(e1);

            Assertions.assertEquals(d1, F[k - 1]);

            // B_2k / 2k!
            final BigFraction factor2 = BigFraction.of(num, factorial.multiply(denom));
            final double d2 = factor2.doubleValue();
            // Cross verify BigFraction vs BigDecimal
            v = factor2.bigDecimalValue(scale, RoundingMode.HALF_EVEN);
            Assertions.assertEquals(v.doubleValue(), d2);
            // Find ULP precision
            final double e2 = new BigDecimal(d2).subtract(v)
                .divide(new BigDecimal(Math.ulp(d2)), scale, RoundingMode.HALF_EVEN).doubleValue();
            sum2 += Math.abs(e2);

            // Print the table:
            // "%s, // %s! / (%s / %s)%n", d1, 2 * k, num, denom
        }
        Assertions.assertTrue(sum1 < sum2, "2k! / B_2k does not have lower combined error");
    }

    @ParameterizedTest
    @MethodSource(value="testZetaSpot")
    void testZetaSpot(double s, double a, double z, int ulp) {
        final double v = zeta3(s, a, 9, 15);
        TestUtils.assertEquals(z, v, DoubleTolerances.ulps(ulp));
    }

    static Stream<Arguments> testZetaSpot() {
        return Stream.of(
            // Spot checks using maxima within its precision limit range for (s, a)
            // load("bffac");
            // bfhzeta(bfloat(s), bfloat(a), 30);
            // TODO: Find another implementation. Maxima's does not agree with Matlab
            // at large a.
            // Re-implement Cephes with 128-bit precision ???

            Arguments.of(1.001, 3, 9.99077634929516400548962369402e2, 0),

            // Reference values using Matlab R2026a Symbolic Math Toolbox
            //   vpa(hurwitzZeta(sym(s, 'f'), sym(a, 'f')))
            // Note: The use of 'f' uses the floating-point conversion as N * 2^e
            // where N is the mantissa and e is the exponent.

            Arguments.of(1.5, 4789, 0.0289021565574206125831622859402, 0),
            Arguments.of(2.345, 12.789, 0.0254390524135780630410689989495, 1),
            Arguments.of(1.345, 12.789, 1.2196695365743183226009917322, 1),
            Arguments.of(1.345, 1278.9, 0.245686258677206272154248922088, 1),
            Arguments.of(4.345, 28697.9, 0.000000000000000366539298049937530342298075842, 1),
            Arguments.of(1.345, 1278562927.9, 0.00209103729002248575358360674936, 0),
            // large s
            Arguments.of(23.45, 12.789, 0.0000000000000000000000000134520611728481431909082787144, 0),
            Arguments.of(23.45, 1278.9, 8.01879331040682555147782226582e-72, 1),
            Arguments.of(23.45, 1278562927.9, 1.59541010013538002585127908428e-206, 1),
            Arguments.of(234.5, 12.789, 2.7978232305220904641253847499e-260, 0),
            Arguments.of(234.5, 17.89, 1.83179298527375942363255194085e-294, 0),
            // small s
            Arguments.of(1.00001, 1.789, 99999.7231466483928005870213366, 1),
            Arguments.of(1.00001, 17.89, 99997.1440070978817356297446315, 1),
            Arguments.of(1.00001, 1278562927.89, 99979.0331951153772621341875866, 1),
            Arguments.of(1.0000000000000002, 1.789, 4503599627370495.72314713725233, 0),
            Arguments.of(1.0000000000000002, 1278562927.89, 4503599627370475.03099742881301, 0)
        );
    }

    @ParameterizedTest
    @Order(1)
    @CsvFileSource(resources = "hurwitzzeta.csv")
    void testZeta1(double s, double a, BigDecimal expected) {
        assertZeta(s, a, expected, (x, p) -> zeta1(x, p, 9, 15), 3, RMS_ZETA1);
    }

    @Test
    void testZetaPrecision1() {
//        System.out.printf("1 max   %s  rms  %s%n", RMS_ZETA1.getMax(), RMS_ZETA1.getRMS());
        // 3.060261764961949  rms  0.6644821802253738
        ExtendedPrecisionTest.assertPrecision(RMS_ZETA1, 3.2, 0.7);
    }

    @ParameterizedTest
    @Order(1)
    @CsvFileSource(resources = "hurwitzzeta.csv")
    void testZeta2(double s, double a, BigDecimal expected) {
        assertZeta(s, a, expected, (x, p) -> zeta2(x, p, 9, 15), 4, RMS_ZETA2);
    }

    @Test
    void testZetaPrecision2() {
//        System.out.printf("2 max   %s  rms  %s%n", RMS_ZETA2.getMax(), RMS_ZETA2.getRMS());
        // 4.064843257412612  rms  0.7655762322011738
        ExtendedPrecisionTest.assertPrecision(RMS_ZETA2, 4.2, 0.8);
    }

    @ParameterizedTest
    @Order(1)
    @CsvFileSource(resources = "hurwitzzeta.csv")
    void testZeta3(double s, double a, BigDecimal expected) {
        // TODO: Output the RMS with varying N
        assertZeta(s, a, expected, (x, p) -> zeta3(x, p, 9, 15), 2, RMS_ZETA3);
    }

    @Test
    void testZetaPrecision3() {
//        System.out.printf("3 max   %s  rms  %s%n", RMS_ZETA3.getMax(), RMS_ZETA3.getRMS());
        // max   2.442495123560756  rms  0.5968708245878805
        ExtendedPrecisionTest.assertPrecision(RMS_ZETA3, 2.5, 0.62);
    }

    // Broken
    //@ParameterizedTest
    @Order(1)
    @CsvFileSource(resources = "hurwitzzeta.csv")
    void testZeta4(double s, double a, BigDecimal expected) {
        assertZeta(s, a, expected, (x, p) -> zeta4(x, p, 25, 25), 5, RMS_ZETA4);
    }

    //@Test
    void testZetaPrecision4() {
//        System.out.printf("4 max   %s  rms  %s%n", RMS_ZETA4.getMax(), RMS_ZETA4.getRMS());
        ExtendedPrecisionTest.assertPrecision(RMS_ZETA4, 0, 0);
    }

    //@ParameterizedTest
    @Order(1)
    @CsvFileSource(resources = "hurwitzzeta.csv")
    void testZetaC(double s, double a, BigDecimal expected) {
        assertZeta(s, a, expected, HurwitzZetaTest::zetaCephes, 10, RMS_ZETAC);
    }

    //@Test
    void testZetaPrecisionC() {
//        System.out.printf("c max   %s  rms  %s%n", RMS_ZETAC.getMax(), RMS_ZETAC.getRMS());
        // max   9.776665608322459  rms  1.094288510174287
        ExtendedPrecisionTest.assertPrecision(RMS_ZETAC, 10, 1.2);
    }

    // TODO:
    // Copy zeta3 to the main class.
    // Add tests for this + spot checks.

    private static void assertZeta(double s, double a, BigDecimal expected, 
            DoubleBinaryOperator f, int ulp, RMS rms) {
        final double e = expected.doubleValue();
        final double x = f.applyAsDouble(s, a);
        TestUtils.assertEquals(e, x, DoubleTolerances.ulps(ulp));
        ExtendedPrecisionTest.addError(x, expected, e, rms);
    }
}
