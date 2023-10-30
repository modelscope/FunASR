#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey);
#                 Arnab Ghoshal, Karel Vesely

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


# Parse command-line options.
# To be sourced by another script (as in ". parse_options.sh").
# Option format is: --option-name arg
# and shell variable "option_name" gets set to value "arg."
# The exception is --help, which takes no arguments, but prints the 
# $help_message variable (if defined).


###
### The --config file options have lower priority to command line 
### options, so we need to import them first...
###

# Now import all the configs specified by command-line, in left-to-right order
for ((argpos=1; argpos<$#; argpos++)); do
  if [ "${!argpos}" == "--config" ]; then
    argpos_plus1=$((argpos+1))
    config=${!argpos_plus1}
    [ ! -r $config ] && echo "$0: missing config '$config'" && exit 1
    . $config  # source the config file.
  fi
done


###
### No we process the command line options
###
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    # If the enclosing script is called with --help option, print the help 
    # message and exit.  Scripts should put help messages in $help_message
  --help|-h) if [ -z "$help_message" ]; then echo "No help found." 1>&2;
	  else printf "$help_message\n" 1>&2 ; fi; 
	  exit 0 ;; 
  --*=*) echo "$0: options to scripts must be of the form --name value, got '$1'"
       exit 1 ;;
    # If the first command-line argument begins with "--" (e.g. --foo-bar), 
    # then work out the variable name as $name, which will equal "foo_bar".
  --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`; 
    # Next we test whether the variable in question is undefned-- if so it's 
    # an invalid option and we die.  Note: $0 evaluates to the name of the 
    # enclosing script.
    # The test [ -z ${foo_bar+xxx} ] will return true if the variable foo_bar
    # is undefined.  We then have to wrap this test inside "eval" because 
    # foo_bar is itself inside a variable ($name).
      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      
      oldval="`eval echo \\$$name`";
    # Work out whether we seem to be expecting a Boolean argument.
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then 
	was_bool=true;
      else 
	was_bool=false;
      fi

    # Set the variable to the right value-- the escaped quotes make it work if
    # the option had spaces, like --cmd "queue.pl -sync y"
      eval $name=\"$2\"; 
        
    # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;
  *) break;
  esac
done


# Check for an empty argument to the --cmd option, which can easily occur as a 
# result of scripting errors.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" 1>&2 && exit 1;


true; # so this script returns exit code 0.
