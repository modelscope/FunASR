#define GFLAGS_DLL_DECLARE_FLAG

#include <iostream>
#include <gflags/gflags_declare.h>

DECLARE_string(message); // in gflags_delcare_test.cc

void print_message();
void print_message()
{
  std::cout << FLAGS_message << std::endl;
}
