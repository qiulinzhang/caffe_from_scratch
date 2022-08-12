#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp> // boost库，shared_ptr 用于智能指针
#include <gflags/gflags.h> // google flags 库，用于处理命令行参数
#include <glog/glog.h> // google log 库，用于处理日志

#include <climits> // 指定了各种int char等的位宽和最大值和最小值
#include <cmath>
#include <fstream> // 文件流 file stream 字符串流，搭配`<<` or `>>` 使用
#include <iostream> // 输入输出流 in/output stream 字符串流，搭配`<<` or `>>` 使用
#include <map>
#include <set>
#include <sstream> // string stream 字符串流，搭配`<<` or `>>` 使用
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/device_altenate.hpp"