#include<Python.h> // apt-get install python-dev 获得的头文件

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<boost/make_shared.hpp> // make_shared 工厂函数代替 new 操作符