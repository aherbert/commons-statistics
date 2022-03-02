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

# Mandatory variables:
#  distFile = data file name
#  distTitle = title of the plot
#  xMin = start of plot range
#  xMax = end of plot range
#  yMax = end of plot range
# Optional variables:
#  logX = Whether to plot the x-axis in log scale
#  logY = Whether to plot the y-axis in log scale

set terminal png enhanced transparent
# font arial 11
set output distFile . ".png"

set title distTitle
set key autotitle columnhead
#unset key

# Missing columns are silently ignored
plot [xMin:xMax][0:yMax] \
  distFile u 1:2 w l title t1,\
  distFile u 1:3 w l,\
  distFile u 1:4 w l,\
  distFile u 1:5 w l,\
  distFile u 1:6 w l
