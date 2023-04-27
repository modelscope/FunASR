cmake_minimum_required (VERSION 3.16)
project (glog_package_config LANGUAGES CXX)

find_package (glog REQUIRED NO_MODULE)

add_executable (glog_package_config glog_package_config.cc)

target_link_libraries (glog_package_config PRIVATE glog::glog)
