/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

 #ifndef CAFFE_LAYER_FACTORY_H_
 #define CAFFE_LAYER_FACTORY_H_

 #include<map>
 #include<string>
 #include<vector>

 #include "caffe/common.hpp"
 #include "caffe/layer.hpp"
 #include "caffe/proto/caffe.pb.h"

 namespace caffe {

    template <typename Dtype>
    class Layer;

    template <typename Dtype>
    class LayerRegistry {
        public:
            typedef shared_ptr<Layer<Dtype> > (*Creator) (const LayerParameter&);
            typedef std::map<string, Creator> CreatorRegistry;

            // 静态成员函数，可以通过类::函数名直接调用，只允许访问静态成员变量，静态成员变量会一直保留
            static CreatorRegistry& Registry() {
                static CreatorRegistry* g_registry_ = new CreatorRegistry();
                return *g_registry_;
            }

            // Add a creatror
            static void AddCreator(const string& type, Creator creator) {
                // 即便每次都是重新定义，但是静态成员变量的值会一直保留
                CreatorRegistry& registry = Registry();
                CHECK_EQ(registry.count(type), 0) << "Layer type " << "already registered.";
                registry[type] = creator;
            }

            // Get a layer using a LayerParameter
            // 通过给定的 LayerParameter 来实时创建一个 Layer
            // 其中会判断param给定的type是否已经注册到 registry 中，如果没有提前注册，就不能创建
            static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
                if (Caffe::root_solver()) { // 如果是并行训练的主机上
                    LOG(INFO) << "Creating layer "<<param.name();
                }
                const string& type = param.type();
                CreatorRegistry& registry = Registry();
                CHECK_EQ(registry.count(type), 1) << "Unkown layer type: "<< type <<
                    " (known types: "<<LayerTypeListString() << ")";
                return registry[type](param);
            }

            static vector<string> LayerTypeList() {
                CreatorRegistry& registry Registry();
                vector<string> layer_types;
                for (typename CreatorRegistry::iterator iter=registry.begin(); iter!=registry.end(); ++iter) {
                    layer_types.push_back(iter->first);
                }
                return layer_types;
            }

        private:
            // Layer registry should never be instantiated - everything is done with its
            // static variable
            LayerRegistry() {}

            static string LayerTypeListString() {
                vector<string> layer_types = LayerTypeList();
                string layer_types_str;

                for (vector<string>::iterator iter = layer_types.begin(); 
                        iter != layer_types.end(); ++iter) {
                            if (iter != layer_types.begin()) {
                                layer_types_str += ", ";
                            }
                            layer_types_str += *iter;
                        }
                return layer_types_str;
            }
    };


    template <typename Dtype>
    class LayerRegister {
        // 直接调用 LayerRegistry 的AddCreator函数来注册层
        public:
            LayerRegister(const string& type, 
                          shared_ptr<Layer<Dtype> > (*creator) (const LayerParameter&)) {
                LayerRegistry<Dtype>::AddCreator(type, creator);
        }
    };


    #define REGISTER_LAYER_CREATOR(type, creator)
        static LayerRegister<float> g_creator_f_##type(#type, creator<float>);
        static LayerRegister<double> g_creator_d_##type(#type, creator<double>);

    #define REGISTER_LAYER_CLASS(type)
        template <typename Dtype>
        shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) {
            return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));
        }
        REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

 } // namespace caffe

 #endif // CAFFE_LAYER_FACTORY_H_