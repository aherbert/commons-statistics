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
package org.apache.commons.statistics.inference;

import java.util.OptionalInt;
import org.apache.commons.statistics.inference.OptionsB.Option;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Test cases for {@link OptionsB}.
 */
class OptionsTest {

    private enum A implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum B implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum C implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum D implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum E implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum F implements Option {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private class AA {}
    private class BB extends AA {}
    private class CC extends BB {}
    private class DD extends CC {}
    private class EE extends DD {}
    private class FF extends EE {}

    @Test
    void testNone() {
        final Options opt = Options.none();
        OptionalInt i;
        Assertions.assertNull(opt.instanceOf(A.class));
        Assertions.assertEquals(A.TWO, opt.orElse(A.TWO));
        Assertions.assertEquals(B.ONE, opt.orElse(B.ONE));
    }

    @Test
    void test5() {
        final Options opt = Options.of(A.ONE, B.TWO, C.THREE, D.FOUR, E.FIVE);
        Assertions.assertSame(A.ONE, opt.instanceOf(A.TWO));
        Assertions.assertSame(B.TWO, opt.instanceOf(B.ONE));
        Assertions.assertSame(C.THREE, opt.instanceOf(C.ONE));
        Assertions.assertSame(D.FOUR, opt.instanceOf(D.ONE));
        Assertions.assertSame(E.FIVE, opt.instanceOf(E.ONE));
        Assertions.assertNull(opt.instanceOf(F.ONE));
        Assertions.assertSame(A.ONE, opt.orElse(A.TWO));
        Assertions.assertSame(B.TWO, opt.orElse(B.ONE));
        Assertions.assertSame(C.THREE, opt.orElse(C.ONE));
        Assertions.assertSame(D.FOUR, opt.orElse(D.ONE));
        Assertions.assertSame(E.FIVE, opt.orElse(E.ONE));
        Assertions.assertSame(F.ONE, opt.orElse(F.ONE));
        // Test null
        Assertions.assertThrows(NullPointerException.class, () -> opt.instanceOf(null));
        Assertions.assertThrows(NullPointerException.class, () -> opt.orElse(null));
        // First option of the class is returned
        final Options opt2 = Options.of(A.ONE, A.TWO, A.THREE, A.FOUR, A.FIVE);
        Assertions.assertSame(A.ONE, opt2.instanceOf(A.THREE));
        AA aa = new AA();
        BB bb = new BB();
        CC cc = new CC();
        DD dd = new DD();
        EE ee = new EE();
        final Options opt3 = Options.of(ee, dd, cc, bb, aa);
        Assertions.assertSame(ee, opt3.instanceOf(new AA()));
        Assertions.assertSame(ee, opt3.instanceOf(new BB()));
        Assertions.assertSame(ee, opt3.instanceOf(new CC()));
        Assertions.assertSame(ee, opt3.instanceOf(new DD()));
        Assertions.assertSame(ee, opt3.instanceOf(new EE()));
        final Options opt4 = Options.of(aa, bb, cc, dd, ee);
        Assertions.assertSame(aa, opt4.instanceOf(new AA()));
        Assertions.assertSame(bb, opt4.instanceOf(new BB()));
        Assertions.assertSame(cc, opt4.instanceOf(new CC()));
        Assertions.assertSame(dd, opt4.instanceOf(new DD()));
        Assertions.assertSame(ee, opt4.instanceOf(new EE()));
    }

    @Test
    void testNoneB() {
        final OptionsB opt = OptionsB.none();
        Assertions.assertNull(opt.get(A.class));
        Assertions.assertEquals(A.TWO, opt.getOrElse(A.TWO));
        Assertions.assertEquals(B.ONE, opt.getOrElse(B.ONE));
    }

    @Test
    void testB5() {
        final OptionsB opt = OptionsB.of(A.ONE, B.TWO, C.THREE, D.FOUR, E.FIVE);
        Assertions.assertEquals(A.ONE, opt.get(A.class));
        Assertions.assertEquals(B.TWO, opt.get(B.class));
        Assertions.assertEquals(C.THREE, opt.get(C.class));
        Assertions.assertEquals(D.FOUR, opt.get(D.class));
        Assertions.assertEquals(E.FIVE, opt.get(E.class));
        Assertions.assertNull(opt.get(F.class));
        Assertions.assertEquals(A.ONE, opt.getOrElse(A.TWO));
        Assertions.assertEquals(B.TWO, opt.getOrElse(B.ONE));
        Assertions.assertEquals(C.THREE, opt.getOrElse(C.ONE));
        Assertions.assertEquals(D.FOUR, opt.getOrElse(D.ONE));
        Assertions.assertEquals(E.FIVE, opt.getOrElse(E.ONE));
        Assertions.assertEquals(F.ONE, opt.getOrElse(F.ONE));
        // Test null
        Assertions.assertNull(opt.get(null));
        Assertions.assertThrows(NullPointerException.class, () -> opt.getOrElse(null));
        // Last option of the class is returned
        final OptionsB opt2 = OptionsB.of(A.ONE, A.TWO, A.THREE, A.FOUR, A.FIVE);
        Assertions.assertEquals(A.FIVE, opt2.get(A.class));
        Assertions.assertNull(opt2.get(F.class));
        Assertions.assertEquals(A.FIVE, opt2.getOrElse(A.TWO));
        Assertions.assertEquals(B.ONE, opt2.getOrElse(B.ONE));
        // Test erasure
        @SuppressWarnings("rawtypes")
        Class c = F.class;
        @SuppressWarnings("unchecked")
        Class<A> c2 = (Class<A>) c;
        Assertions.assertNull(opt.get(c2));
        Option o = F.TWO;
        Assertions.assertEquals(F.TWO, opt.getOrElse(o));
    }
}
