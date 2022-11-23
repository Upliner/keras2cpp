#include "model.h"
#include "layers/conv1d.h"
#include "layers/conv2d.h"
#include "layers/dense.h"
#include "layers/elu.h"
#include "layers/embedding.h"
#include "layers/flatten.h"
#include "layers/locally1d.h"
#include "layers/locally2d.h"
#include "layers/lstm.h"
#include "layers/maxPooling2d.h"
#include "layers/upSampling2d.h"
#include "layers/batchNormalization.h"
#include "layers/binary.h"

namespace keras2cpp {
    std::unique_ptr<Layer> Model::make_layer(Stream& file) {
        switch (static_cast<unsigned>(file)) {
            case Dense:
                return layers::Dense::make(file);
            case Conv1D:
                return layers::Conv1D::make(file);
            case Conv2D:
                return layers::Conv2D::make(file);
            case LocallyConnected1D:
                return layers::LocallyConnected1D::make(file);
            case LocallyConnected2D:
                return layers::LocallyConnected2D::make(file);
            case Flatten:
                return layers::Flatten::make(file);
            case ELU:
                return layers::ELU::make(file);
            case Activation:
                return layers::Activation::make(file);
            case MaxPooling2D:
                return layers::MaxPooling2D::make(file);
            case LSTM:
                return layers::LSTM::make(file);
            case Embedding:
                return layers::Embedding::make(file);
            case BatchNormalization:
                return layers::BatchNormalization::make(file);
            case UpSampling2D:
                return layers::UpSampling2D::make(file);

            case Add:
                return layers::Add::make(file);
            case Multiply:
                return layers::Multiply::make(file);
            case Concatenate:
                return layers::Concatenate::make(file);

            case InputLayerType:
                return std::make_unique<InputLayer>();
        }
        return nullptr;
    }

    Model::Model(Stream& file) {
        auto count = static_cast<unsigned>(file);
        if (count > file.size()/sizeof(unsigned))
            throw KerasException("Model file is corrupt");
        layers_.reserve(count);
        for (size_t i = 0; i != count; ++i)
            layers_.push_back(make_layer(file));
    }

    Model Model::load(const std::string& filename) {
        Stream file(filename);
        return Model(file);
    }

    void Model::run(MachineState &state) const {
        for (auto& layer : layers_)
            layer->run(state);
    }

    std::unique_ptr<Tensor> Model::calc(const std::vector<const Tensor*> &input) const {
        MachineState state;
        state.inputs = input;
        state.tensors.reserve(layers_.size());
        state.calculated_tensors.reserve(layers_.size());
        run(state);
        return std::move(state.calculated_tensors.back());
    }

    Tensor Model::operator()(const Tensor &input) const {
        return *((*this).calc(std::vector<const Tensor*>{&input}).get());
    }
}
