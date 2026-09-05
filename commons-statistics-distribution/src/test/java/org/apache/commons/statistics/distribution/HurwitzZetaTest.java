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
import java.util.stream.Stream;
import org.apache.commons.numbers.fraction.BigFraction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Test the {@link HurwitzZeta} function.
 */
class HurwitzZetaTest {
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
     * as stated in Johansson (2015) section 3.1.
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

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(s, a)}.
     *
     * <pre>
     *                 oo    1
     * zeta(s, a) = sum    ------
     *                 k=0      s
     *                     (k+a)
     * </pre>
     *
     * <p><strong>Warning</strong>: No parameter validation is performed.
     * 
     * <p>This implementation allows the parameters {@code N} and {@code M}
     * in the Euler-Maclaurin approximation to be varied to test the function
     * precision.
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    static double zeta(double s, double a, int n, int m) {
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
     *
     * @param s Argument {@code s > 1}
     * @param a Argument {@code a >= 1}
     * @param n Argument {@code N}
     * @param m Argument {@code M}
     * @return zeta(s, a)
     */
    static double zeta3(double s, double a, int n, int m) {
        // TODO: The method is very sensitive to the initial loop over N to create S
        // The N cannot be too high if using an ascending sum.

        double sum = 0;
        // S : k in [0, n-1]
        for (int k = n; --k >= 0;) {
            // Descending k sums in order of magnitude for increased precision.
            // Prevents early exit for large s when the term (a+k)^-s is below
            // machine epsilon of the ascending series sum.
            sum += Math.pow(a + k, -s);
        }
//        // S : k in [0, n-1]
//        double sum = Math.pow(a, -s);
//        for (int k = 1; k < n; k++) {
//            // Allow early convergence
//            final double t = Math.pow(a + k, -s);
//            sum += t;
//            if (sum + t == sum) {
//                return sum;
//            }
//        }
        final double apn = a + n;
        // I : (a+n)^(1-s) / (s-1)
        sum += Math.pow(apn, 1 - s) / (s - 1);
        // T
        double p = Math.pow(apn, -s);
        sum += 0.5 * p;

        // The following recycles the power term p: (a+n)^-(2k-1+s).
        // This incorporates the factor for T into the sum terms.
        // This sets the first power as (a+n)^-(1+s) not (a+n)^-1.
        // When s is large the loop exits before the rising factorial overflows.

        // Rising factorial term.
        double f = s;
        // 2k - 1
        double k2 = 1;
        double tsum = 0;
        for (int i = 0; i < m; i++) {
            // p = (a+n)^-(2k-1+s)
            p /= apn;
            final double t = f * p / F[i];
            tsum += t;
            if (tsum + t == tsum) {
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
        return sum + tsum;
    }

    /**
     * Compute the value of the Hurwitz zeta function {@code zeta(x, q)}.
     *
     * @param x Argument {@code x > 1}
     * @param q Argument {@code q}
     * @return zeta(x, q)
     */
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
    @MethodSource
    void testSumConvergence(double s, double a, int n) {
        double sum = 0;
        double t = 0;
        int k = 0;
        for (; k < n; k++) {
            t = Math.pow(a + k, -s);
            final double updated = sum + t;
            if (updated == sum) {
                break;
            }
            sum = updated;
        }
        final double S = sum;
        final double last = t;
        //System.out.printf("Arguments.of(%s, %d, %d), // %s%n", s, a, k + 1, sum);
        Assertions.assertTrue(k < n, () -> S + " Ulp error: " + (last / Math.ulp(S)));
    }

    static Stream<Arguments> testSumConvergence() {
        // As a -> inf the sum requires more iterations as the terms are similar in magnitude
        return Stream.of(
            // Convergence from small a
            Arguments.of(20.0, 1, 7), // 1.0000009539620338
            Arguments.of(15.0, 1, 12), // 1.000030588236307
            Arguments.of(10.0, 1, 40), // 1.0009945751278182
            Arguments.of(5.0, 1, 1553), // 1.036927755143338
            Arguments.of(20.0, 3, 18), // 2.8771746654611316E-10
            Arguments.of(15.0, 3, 34), // 7.065818202049353E-8
            Arguments.of(10.0, 3, 118), // 1.8012627818085313E-5
            // Convergence from large a
            Arguments.of(100.0, 1000, 420), // 1.0609342003874935E-299
            Arguments.of(50.0, 1000, 966), // 2.091232974781865E-149
            // Large s with large a -> 0
            Arguments.of(70.0, 10000, 5749) // 1.4542811956471796E-278
       );
    }

    @ParameterizedTest
    @MethodSource
    void testTailConvergence(double s, double a, int n, int m) {
        Assumptions.assumeTrue(m > 1, "No early exit for large s");
        double sum = 0;
        double t = 0;
        int k = 0;
        double num = s;
        double factor = Math.pow(a + n, -s);
        // k is offset by -1
        for (; k < m; k++) {
            t = num * Math.pow(a + n, -(2 * k + 1)) / F[k];
//            System.out.printf("  %s%n", t * factor);
            final double updated = sum + t;
            if (updated == sum) {
                break;
            }
            sum = updated;
            // Potential for overflow
            // Pochhammer(x, n) = gamma(n + x) / gamma(x) = exp(logGamma(n + x) - logGamma(x))
            // gamma(x) overflows at x>171
            num = num * (s + (2 * k + 1)) * (s + (2 * k + 2));
        }
        final double T = sum;
        final double last = t;
        System.out.printf("Arguments.of(%s, %d, %d, %d), // %s (%s)%n", s, a, n, k + 1, sum, sum * factor);
        Assertions.assertTrue(k < m, () -> T + " Ulp error: " + (last / Math.ulp(T)));
    }

    @ParameterizedTest
    @MethodSource(value="testTailConvergence")
    void testTailConvergence2(double s, double a, int n, int m) {
        double sum = 0;
        double t = 0;
        double apn = a + n;
        // Power term: (a+n)^-(2k-1+s)
        // Incorporates the factor for T into the sum terms.
        // This sets the first power as (a+n)^-(1+s) not (a+n)^-1.
        // When s is large the loop exits before rising factorial overflows.
        double p = Math.pow(apn, -s);
        // Rising factorial term
        double f = s;
        // k is offset by -1
        int k = 0;
        for (; k < m; k++) {
            // p = (a+n)^-(2k-1+s)
            p /= apn;
            //System.out.printf("%s %s%n", g, Math.pow(apn, -(2*k+1 +s)));
            t = f * p / F[k];
//            System.out.printf("  %s%n", t);
            final double updated = sum + t;
            if (updated == sum) {
                break;
            }
            sum = updated;
            p /= apn;
            f = f * (s + (2 * k + 1)) * (s + (2 * k + 2));
        }
        final double T = sum;
        final double last = t;
        System.out.printf("Arguments.of(%s, %d, %d, %d), // %s%n", s, a, n, k + 1, sum);
        Assertions.assertTrue(k < m, () -> T + " Ulp error: " + (last / Math.ulp(T)));
    }

    static Stream<Arguments> testTailConvergence() {
        return Stream.of(
            // Value is T * (a+p)^-s
            Arguments.of(20.0, 1, 25, 12), // 3.180740271816808E-30
            Arguments.of(15.0, 1, 25, 11), // 2.8474033173332883E-23
            Arguments.of(10.0, 1, 25, 10), // 2.2631078280898945E-16
            Arguments.of(5.0, 1, 25, 8), // 1.3474102948521524E-9
            Arguments.of(20.0, 3, 25, 12), // 6.719150386148452E-31
            Arguments.of(15.0, 3, 25, 11), // 8.707407789332541E-24
            Arguments.of(10.0, 3, 25, 10), // 1.0019976805851467E-16
            Arguments.of(100.0, 1000, 25, 6), // 6.880775456873249E-304
            Arguments.of(50.0, 1000, 25, 5), // 1.182642149530307E-153
            Arguments.of(70.0, 10000, 25, 4), // 4.885683776445016E-284
            Arguments.of(5.0, 1000000, 25, 3), // 4.166041721347605E-37
            Arguments.of(3.0, 1000000, 25, 3), // 2.4997500156233853E-25
            Arguments.of(2.0, 1000000, 25, 3), // 1.6665416729160733E-19
            // Pochhammer overflow with large s requires detection
            Arguments.of(1000000.0, 10000, 25, 1), // 0.0
            Arguments.of(1.0E9, 10000, 25, 1), // 0.0
            Arguments.of(1.0E200, 10000, 25, 1) // 0.0
        );
    }

    @ParameterizedTest
    @MethodSource(value="testZeta")
    void testZeta(double s, double a, int n, int m, double z, int ulp) {
//        final double v = zeta(s, a, n, m);
//        final double v = zeta2(s, a, n, m);
      final double v = zeta3(s, a, n, m);
//        final double v = zetaCephes(s, a);
        TestUtils.assertEquals(z, v, DoubleTolerances.ulps(ulp));
    }

    // TODO:
    // More reference data
    // Run each method and compute the mean and max ULP error
    // Try with different n and m

    static Stream<Arguments> testZeta() {
        // Reference values using Matlab R2026a Symbolic Math Toolbox
        //   vpa(hurwitzZeta(sym(s, 'f'), sym(a, 'f')))
        // Note: The use of 'f' uses the floating-point conversion as N * 2^e
        // where N is the mantissa and e is the exponent.

        return Stream.of(
            // High N & M do not increase precision
            Arguments.of(2.345, 12.789, 53, 53, 0.0254390524135780630410689989495, 2),
            Arguments.of(2.345, 12.789, 25, 25, 0.0254390524135780630410689989495, 2),
            Arguments.of(2.345, 12.789, 15, 15, 0.0254390524135780630410689989495, 1),
            Arguments.of(2.345, 12.789, 8, 8, 0.0254390524135780630410689989495, 1),
            Arguments.of(1.345, 12.789, 53, 53, 1.2196695365743183226009917322, 0),
            Arguments.of(1.345, 12.789, 25, 25, 1.2196695365743183226009917322, 1),
            Arguments.of(1.345, 12.789, 15, 15, 1.2196695365743183226009917322, 0),
            Arguments.of(1.345, 12.789, 8, 8, 1.2196695365743183226009917322, 0),
            Arguments.of(1.345, 1278.9, 53, 53, 0.245686258677206272154248922088, 2),
            Arguments.of(1.345, 1278.9, 25, 25, 0.245686258677206272154248922088, 1),
            Arguments.of(1.345, 1278.9, 15, 15, 0.245686258677206272154248922088, 0),
            Arguments.of(1.345, 1278.9, 8, 8, 0.245686258677206272154248922088, 0),
            Arguments.of(4.345, 28697.9, 53, 53, 0.000000000000000366539298049937530342298075842, 1),
            Arguments.of(4.345, 28697.9, 25, 25, 0.000000000000000366539298049937530342298075842, 1),
            Arguments.of(4.345, 28697.9, 15, 15, 0.000000000000000366539298049937530342298075842, 1),
            Arguments.of(4.345, 28697.9, 8, 8, 0.000000000000000366539298049937530342298075842, 0),
            Arguments.of(1.345, 1278562927.9, 53, 53, 0.00209103729002248575358360674936, 1),
            Arguments.of(1.345, 1278562927.9, 25, 25, 0.00209103729002248575358360674936, 1),
            Arguments.of(1.345, 1278562927.9, 15, 15, 0.00209103729002248575358360674936, 0),
            Arguments.of(1.345, 1278562927.9, 8, 8, 0.00209103729002248575358360674936, 0),
            // large s
            Arguments.of(23.45, 12.789, 25, 25, 0.0000000000000000000000000134520611728481431909082787144, 0),
            Arguments.of(23.45, 12.789, 15, 15, 0.0000000000000000000000000134520611728481431909082787144, 0),
            Arguments.of(23.45, 1278.9, 25, 25, 8.01879331040682555147782226582e-72, 0),
            Arguments.of(23.45, 1278.9, 10, 10, 8.01879331040682555147782226582e-72, 0),
            Arguments.of(23.45, 1278562927.9, 25, 25, 1.59541010013538002585127908428e-206, 0),
            Arguments.of(23.45, 1278562927.9, 10, 10, 1.59541010013538002585127908428e-206, 1),
            Arguments.of(234.5, 12.789, 25, 25, 2.7978232305220904641253847499e-260, 0),
            Arguments.of(234.5, 12.789, 10, 10, 2.7978232305220904641253847499e-260, 0),
            Arguments.of(234.5, 17.89, 25, 25, 1.83179298527375942363255194085e-294, 0),
            Arguments.of(234.5, 17.89, 10, 10, 1.83179298527375942363255194085e-294, 0),
            // small s
            Arguments.of(1.00001, 1.789, 25, 25, 99999.7231466483928005870213366, 1),
            Arguments.of(1.00001, 1.789, 10, 10, 99999.7231466483928005870213366, 0),
            Arguments.of(1.00001, 17.89, 25, 25, 99997.1440070978817356297446315, 1),
            Arguments.of(1.00001, 17.89, 10, 10, 99997.1440070978817356297446315, 1),
            Arguments.of(1.00001, 1278562927.89, 25, 25, 99979.0331951153772621341875866, 0),
            Arguments.of(1.00001, 1278562927.89, 15, 15, 99979.0331951153772621341875866, 0),
            Arguments.of(1.0000000000000002, 1.789, 25, 25, 4503599627370495.72314713725233, 0),
            Arguments.of(1.0000000000000002, 1.789, 10, 10, 4503599627370495.72314713725233, 0),
            Arguments.of(1.0000000000000002, 17.89, 25, 25, 4503599627370493.14396697014348, 0),
            Arguments.of(1.0000000000000002, 17.89, 10, 10, 4503599627370493.14396697014348, 0),
            Arguments.of(1.0000000000000002, 1278562927.89, 25, 25, 4503599627370475.03099742881301, 0),
            Arguments.of(1.0000000000000002, 1278562927.89, 15, 15, 4503599627370475.03099742881301, 0)
        );
    }
}
