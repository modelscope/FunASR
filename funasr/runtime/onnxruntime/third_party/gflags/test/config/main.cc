#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(message, "Hello World!", "The message to print");

static bool ValidateMessage(const char* flagname, const std::string &message)
{
  return !message.empty();
}
DEFINE_validator(message, ValidateMessage);

int main(int argc, char **argv)
{
  gflags::SetUsageMessage("Test CMake configuration of gflags library (gflags-config.cmake)");
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << FLAGS_message << std::endl;
  gflags::ShutDownCommandLineFlags();
  return 0;
}
