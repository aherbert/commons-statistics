#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

EXEC='java -jar ../../../target/examples-distribution.jar'
GNUPLOT='/usr/local/bin/gnuplot'

#$EXEC norm pdf --steps 1000 -m 0,0,0,-2 -s 0.4472135954999579,1.0,2.23606797749979,0.7071067811865476 --min -5 --max 5 --out norm.pdf.txt

$GNUPLOT -e 'distFile="norm.pdf.txt"; distTitle="Normal distribution PDF"; xMin=-5; xMax=5; yMax=1.0; t1="{/Symbol m}=0, {/Symbol s}^2=0.2"' dist_density.gnuplot
