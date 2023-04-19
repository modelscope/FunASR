#!/bin/bash

default='\033[0m'
bold='\033[1m'
red='\033[31m'
green='\033[32m'

function ok() {
  printf "${bold}${green}[OK]${default} $1\n"
}

function error() {
  printf "${bold}${red}[FAILED]${default} $1\n"
}

function abort() {
  printf "${bold}${red}[FAILED]${default} $1\n"
  exit 1
}
