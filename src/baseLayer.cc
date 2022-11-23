#include "baseLayer.h"
namespace keras2cpp {
    void InputLayer::run(MachineState &state) const {
        if (state.inputs.empty())
            throw KerasException("Corrupt keras model: invalid input layer");
        state.tensors.push_back(state.inputs[0]);
        state.inputs.erase(state.inputs.begin());
    }
    void BaseUnaryLayer::run(MachineState &state) const {
        if (in_id >= state.tensors.size())
            throw KerasException("Corrupt keras model: invalid input id");
        auto uptr = std::make_unique<Tensor>((*this)(*state.tensors[in_id]));
        state.tensors.push_back(uptr.get());
        state.calculated_tensors.push_back(std::move(uptr));
    }
    void BaseBinaryLayer::run(MachineState &state) const {
        if (a_id >= state.tensors.size() || b_id >= state.tensors.size())
            throw KerasException("Corrupt keras model: invalid input id");
        auto uptr = std::make_unique<Tensor>((*this)(*state.tensors[a_id], *state.tensors[b_id]));
        state.tensors.push_back(uptr.get());
        state.calculated_tensors.push_back(std::move(uptr));
    }
}
