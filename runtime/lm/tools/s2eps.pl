#!/usr/bin/perl
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script replaces <s> and </s> with <eps> (on both input and output sides),
# for the G.fst acceptor.

while(<>){
    @A = split(" ", $_);
    if ( @A >= 4 ) {
        if ($A[2] eq "<s>" || $A[2] eq "</s>") { $A[2] = "<eps>"; }
        if ($A[3] eq "<s>" || $A[3] eq "</s>") { $A[3] = "<eps>"; }
    }
    print join("\t", @A) . "\n";
}
