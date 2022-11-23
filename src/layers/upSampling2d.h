#pragma once
#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class UpSampling2D final : public UnaryLayer<UpSampling2D> {
            unsigned y_{0};
            unsigned x_{0};

        public:
            UpSampling2D(Stream& file);
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}
