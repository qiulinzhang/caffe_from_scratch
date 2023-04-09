#include<Python.h> // apt-get install python-dev 获得的头文件 可以使用 C 或 C++ 扩展 Python

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<boost/make_shared.hpp> // make_shared 工厂函数代替 new 操作符
#include<boost/python.hpp> // a C++ library which enables seamless interoperability between C++ and the Python programming language.
#include<boost/python/raw_function.hpp>
#include<boost/python/suite/indexing/vector_indexing_suite.hpp>
#include<numpy/arrayobject.h> // Pyhon端numpy的头文件

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/sgd_solver.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

/* Fix to avoid registration warnings in pycaffe (#3960) */
#define BP_REGISTER_SHARED_PTR_TO_PYTHON(PTR) do { \
  const boost::python::type_info info = \
    boost::python::type_id<shared_ptr<PTR > >(); \
  const boost::python::converter::registration* reg = \
    boost::python::converter::registry::query(info); \
  if (reg == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } else if ((*reg).m_to_python == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } \
} while (0)

namespace bp = boost::python

namespace caffe {
// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;

// Selecting mode
void set_mode_cpu() {Caffe::set_mode(Caffe::CPU);}
void set_mode_gpu() {Caffe::set_mode(Caffe::GPU);}

void InitLog(){
    ::google::InitGoogleLogging("");
    ::google::InstallFailureSignalHandler();
}
void InitLogLevel(int level) {
    FLAGS_minloglevel = level;
    InitLog();
}
void InitLogLevelPipe(int level, bool stderr) {
    FLAGS_minloglevel = level;
    FLAGS_logtostderr = stderr;
    InitLog();
}
void Log(const string& s) {
    LOG(INFO) << s;
}

void set_random_seed(unsigned int seed) {Caffe::set_random_seed(seed);}


// For convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases).

static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if(!f.good()) {
        f.close();
        throw std::runtime_error("Could not open file "+filename);
    }
    f.close();
}

void CheckContiguousArray(PyArrayObject* arr, string name, int channels, int height, int width){
    // C 连续指对于多维数组，采取行优先的存储方式， 只有最后一个分量变化的元素在内存中连续排列。
    // Fortran MatLab 采取列优先的存储方式， 只有第一个分量变化的元素在内存中连续排列
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        throw std::runtime_error(name + " must be C continuous");
    }

    if (PyArray_NDIM(arr) !=4) {
        throw std::runtime_error(name+ " must be 4-d");
    }

    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
      throw std::runtime_error(name + " must be float32");
    }
    if (PyArray_DIMS(arr)[1] != channels) {
      throw std::runtime_error(name + " has wrong number of channels");
    }
    if (PyArray_DIMS(arr)[2] != height) {
      throw std::runtime_error(name + " has wrong height");
    }
    if (PyArray_DIMS(arr)[3] != width) {
      throw std::runtime_error(name + " has wrong width");
    }

}

// Net constructor
shared_ptr<Net<Dtype> > Net_Init(string network_file, int phase, const int level, 
                                 const bp::object& stages, const bp::object& weights) {
    CheckFile(network_file);

    // Convert stages from list to vector
    vector<string> stages_vector;
    if(!stages.is_none()){
      for(int i=0; i<len(stages); ++i) {
        stages_vector.push_back(bp::extract<string>(stages[i]));
      }
    }

    // Initialize Net
    shared_ptr<Net<Dtype> > net(new Net<Dtype>(network_file, static_cast<Phase>(phase), level, &stages_vector));

    // Load weights
    if(!weights.is_none()) {
      std::string weights_file_str = bp::extract<std::string>(weights);
      CheckFile(weights_file_str);
      net->CopyTrainedLayersFrom(weights_file_str);
    }
    return net;
    }

// Legacy Net construct-and-load convenience constructor
shared_ptr<Net<Dtype> > Net_Init_Load(string param_file, string pretrained_param_file, int phase) {
  LOG(WARNING) << "DEPRECATION WARNING - deprecated use of Python interface";
  LOG(WARNING) << "Use this instead (with the named \"weights\" parameter):";
  LOG(WARNING) << "Net('"<<param_file<<"', "<<phase<<", weights='"<<pretrained_param_file<<"')";
  CheckFile(param_file);
  CheckFile(pretrained_param_file);

  shared_ptr<Net<Dtype> > net(new Net<Dtype>(param_file, static_cast<Phase>(phase)));
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

void Net_Save(const Net<Dtype>& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinary(net_param, filename.c_str());
}

void Net_SaveHDF5(const Net<Dtype>& net, string filename) {
  net.ToHDF5(filename);
}

void Net_LoadHDF5(Net<Dtype>* net, string filename) {
  net->CopyTrainedLayersFromHDF5(filename.c_str());
}



}