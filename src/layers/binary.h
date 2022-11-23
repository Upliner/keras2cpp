#pragma once
#include "../baseLayer.h"
namespace keras2cpp{
    namespace layers{
        class Add final : public BinaryLayer<Add> {
        public:
            using BinaryLayer<Add>::BinaryLayer;
            Tensor operator()(const Tensor& a, const Tensor& b) const noexcept override;
        };

        class Multiply final : public BinaryLayer<Multiply> {
        public:
            using BinaryLayer<Multiply>::BinaryLayer;
            Tensor operator()(const Tensor& a, const Tensor& b) const noexcept override;
        };

        class Concatenate final : public BinaryLayer<Concatenate> {
        public:
            using BinaryLayer<Concatenate>::BinaryLayer;
            Tensor operator()(const Tensor& a, const Tensor& b) const noexcept override;
        };
    }
}
