#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright Chao Weng 

# normalizations for hkust trascript
# see the docs/trans-guidelines.pdf for details

while (<STDIN>) {
  @A = split(" ", $_);
  print "$A[0] ";
  for ($n = 1; $n < @A; $n++) { 
    $tmp = $A[$n];
    if ($tmp =~ /<sil>/) {$tmp =~ s:<sil>::g;}
    if ($tmp =~ /<%>/) {$tmp =~ s:<%>::g;}
    if ($tmp =~ /<->/) {$tmp =~ s:<->::g;}
    if ($tmp =~ /<\$>/) {$tmp =~ s:<\$>::g;}
    if ($tmp =~ /<#>/) {$tmp =~ s:<#>::g;}
    if ($tmp =~ /<_>/) {$tmp =~ s:<_>::g;}
    if ($tmp =~ /<space>/) {$tmp =~ s:<space>::g;}
    if ($tmp =~ /`/) {$tmp =~ s:`::g;}
    if ($tmp =~ /&/) {$tmp =~ s:&::g;}
    if ($tmp =~ /,/) {$tmp =~ s:,::g;}
    if ($tmp =~ /[a-zA-Z]/) {$tmp=uc($tmp);} 
    if ($tmp =~ /Ａ/) {$tmp =~ s:Ａ:A:g;}
    if ($tmp =~ /ａ/) {$tmp =~ s:ａ:A:g;}
    if ($tmp =~ /ｂ/) {$tmp =~ s:ｂ:B:g;}
    if ($tmp =~ /ｃ/) {$tmp =~ s:ｃ:C:g;}
    if ($tmp =~ /ｋ/) {$tmp =~ s:ｋ:K:g;}
    if ($tmp =~ /ｔ/) {$tmp =~ s:ｔ:T:g;}
    if ($tmp =~ /，/) {$tmp =~ s:，::g;}
    if ($tmp =~ /丶/) {$tmp =~ s:丶::g;}
    if ($tmp =~ /。/) {$tmp =~ s:。::g;}
    if ($tmp =~ /、/) {$tmp =~ s:、::g;}
    if ($tmp =~ /？/) {$tmp =~ s:？::g;}
    print "$tmp "; 
  }
  print "\n"; 
}
