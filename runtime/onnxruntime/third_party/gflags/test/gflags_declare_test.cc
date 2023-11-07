#include <gflags/gflags.h>

DEFINE_string(message, "", "The message to print");
void print_message(); // in gflags_declare_flags.cc

int main(int argc, char **argv)
{
  GFLAGS_NAMESPACE::SetUsageMessage("Test compilation and use of gflags_declare.h");
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  print_message();
  return 0;
}
