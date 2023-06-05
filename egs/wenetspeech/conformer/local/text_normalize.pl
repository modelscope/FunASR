#!/usr/bin/env perl
use utf8;
use open qw(:std :utf8);
use warnings;

while (<STDIN>) {
    chomp;
    # remove non UTF-8 whitespace character
    if ($_ =~ /　/) {$_ =~ s:　::g;}
    if ($_ =~ / /) {$_ =~ s: ::g;}
    # upper letters
    if ($_ =~ /[a-zA-Z]/) {$_ =~ uc $_;}
    # add "_" before and after each English word
    if ($_ =~ /([A-Z]+)\s+([A-Z]+)/) {$_ =~ s/([A-Z]+)\s+([A-Z]+)/$1\_$2/g;}
    if ($_ =~ /([A-Z]+)\s+([A-Z]+)/) {$_ =~ s/([A-Z]+)\s+([A-Z]+)/$1\_$2/g;}
    if ($_ =~ m/([A-Z]+)(\p{Han}+)/) {$_ =~ s/([A-Z]+)(\p{Han}+)/$1\_$2/g;}
    if ($_ =~ m/(\p{Han}+)([A-Z]+)/) {$_ =~ s/(\p{Han}+)([A-Z]+)/$1\_$2/g;}
    # remove UTF-8 whitespace charcter
    if ($_ =~ /\s+/) {$_ =~ s:\s+::g;}
    # replace "_" with a normal whitespace
    if ($_ =~ /\_/) {$_ =~ s:\_: :g;}

    print "$_\n";
}
