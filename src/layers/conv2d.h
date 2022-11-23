#pragma once
#include "activation.h"
namespace keras2cpp{
    namespace layers{
        class Conv2D final : public UnaryLayer<Conv2D> {
            unsigned padding, stride_y, stride_x;
            Tensor weights_;
            Tensor biases_;
            Activation activation_;
        public:
            Conv2D(Stream& file);
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}