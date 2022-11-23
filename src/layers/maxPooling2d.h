#pragma once
#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class MaxPooling2D final : public UnaryLayer<MaxPooling2D> {
            unsigned pool_size_y_{0};
            unsigned pool_size_x_{0};

        public:
            MaxPooling2D(Stream& file);
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}