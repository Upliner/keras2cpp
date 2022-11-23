#pragma once
#include "baseLayer.h"
namespace keras2cpp {
    class Model : public Layer {
        enum _LayerType : unsigned {
            Dense = 1,
            Conv1D = 2,
            Conv2D = 3,
            LocallyConnected1D = 4,
            LocallyConnected2D = 5,
            Flatten = 6,
            ELU = 7,
            Activation = 8,
            MaxPooling2D = 9,
            LSTM = 10,
            Embedding = 11,
            BatchNormalization = 12,
            UpSampling2D = 13,

            Add = 101,
            Multiply = 102,
            Concatenate = 103,

            InputLayerType = 1000,
        };
        std::vector<std::unique_ptr<Layer>> layers_;

        static std::unique_ptr<Layer> make_layer(Stream&);

    public:
        static Model load(const std::string& filename);
        Model(Stream& file);
        void run(MachineState &state) const;
        std::unique_ptr<Tensor> calc(const std::vector<const Tensor*> &input) const;
        Tensor operator()(const Tensor &input) const;
    };
}
