#pragma once

#include "tensor.h"
#include <memory>

namespace keras2cpp {
    struct MachineState {
        std::vector<const Tensor*> inputs;
        std::vector<const Tensor*> tensors;
        std::vector<std::unique_ptr<Tensor>> calculated_tensors;
    };
    class Layer {
    public:
        Layer() = default;
        Layer(Stream&) : Layer() {}
        Layer(Layer&&) = default;
        Layer& operator=(Layer&&) = default;
        virtual void run(MachineState &state) const = 0;
        virtual ~Layer() = default;
    };

    class InputLayer : public Layer {
    public:
        virtual void run(MachineState &values) const;
    };

    class BaseUnaryLayer : public Layer {
    protected:
        unsigned in_id;
    public:
        using Layer::Layer;
        virtual void run(MachineState &state) const;
        virtual Tensor operator()(const Tensor& in) const noexcept = 0;
    };

    class BaseBinaryLayer : public Layer {
    protected:
        unsigned a_id, b_id;
    public:
        using Layer::Layer;
        virtual void run(MachineState &state) const;
        virtual Tensor operator()(const Tensor& a, const Tensor& b) const noexcept = 0;
    };

    template <typename Derived>
    class UnaryLayer : public BaseUnaryLayer {
    public:
        using BaseUnaryLayer::BaseUnaryLayer;
        static std::unique_ptr<Layer> make(Stream& file) {
            auto result = std::make_unique<Derived>(file);
            result->in_id = static_cast<unsigned>(file);
            return result;
        }
    };

    template <typename Derived>
    class BinaryLayer : public BaseBinaryLayer {
    public:
        using BaseBinaryLayer::BaseBinaryLayer;
        static std::unique_ptr<Layer> make(Stream& file) {
            auto result = std::make_unique<Derived>(file);
            result->a_id = static_cast<unsigned>(file);
            result->b_id = static_cast<unsigned>(file);
            return result;
        }
    };
}
