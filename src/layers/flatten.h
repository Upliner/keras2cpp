#pragma once
#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class Flatten final : public UnaryLayer<Flatten> {
        public:
            using UnaryLayer<Flatten>::UnaryLayer;
            Tensor operator()(const Tensor& in) const noexcept override;
        };
    }
}
