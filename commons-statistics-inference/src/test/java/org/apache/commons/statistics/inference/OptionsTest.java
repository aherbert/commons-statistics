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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Test cases for {@link Options}.
 */
class OptionsTest {

    private enum A {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum B {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum C {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum D {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum E {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private enum F {
        ONE, TWO, THREE, FOUR, FIVE
    }

    private class AA {}
    private class BB extends AA {}
    private class CC extends BB {}
    private class DD extends CC {}
    private class EE extends DD {}
    private class FF extends EE {}

    @Test
    void testOrElse() {
        final Object[] options = {A.ONE, B.TWO, C.THREE, D.FOUR, E.FIVE};
        Assertions.assertSame(A.ONE, Options.orElse(A.TWO, options));
        Assertions.assertSame(B.TWO, Options.orElse(B.ONE, options));
        Assertions.assertSame(C.THREE, Options.orElse(C.ONE, options));
        Assertions.assertSame(D.FOUR, Options.orElse(D.ONE, options));
        Assertions.assertSame(E.FIVE, Options.orElse(E.ONE, options));
        Assertions.assertSame(F.ONE, Options.orElse(F.ONE, options));
        Assertions.assertSame(F.ONE, Options.orElse(F.ONE));
    }

    @Test
    void testOrElse2() {
        final AA a = new AA();
        final BB b = new BB();
        final CC c = new CC();
        final DD d = new DD();
        final EE e = new EE();
        final FF f = new FF();
        final Object[] options = {a, b, c, d, e};
        Assertions.assertSame(a, Options.orElse(new AA(), options));
        Assertions.assertSame(b, Options.orElse(new BB(), options));
        Assertions.assertSame(c, Options.orElse(new CC(), options));
        Assertions.assertSame(d, Options.orElse(new DD(), options));
        Assertions.assertSame(e, Options.orElse(new EE(), options));
        Assertions.assertSame(f, Options.orElse(f, options));
        Assertions.assertSame(f, Options.orElse(f));
    }
}
