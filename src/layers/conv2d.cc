﻿#include "conv2d.h"
namespace keras2cpp{
    static Tensor pad_tensor(const Tensor &in, unsigned x, unsigned y) noexcept {
        Tensor out(in.dims_[0] + y, in.dims_[1] + x, in.dims_[2]);
        auto orow_size = out.dims_[1] * out.dims_[2];
        auto ptr = out.begin() + orow_size * (y/2) + out.dims_[2] * (x/2);
        if (!x) {
            std::copy(in.begin(), in.end(), out.begin());
            return out;
        }
        const auto irow_size = in.dims_[1] * in.dims_[2];
        const auto i_end = in.end();
        for (auto i_ptr = in.begin(); i_ptr < i_end;) {
            const auto row_end = i_ptr + irow_size;
            std::copy(i_ptr, row_end, ptr);
            ptr += orow_size;
            i_ptr = row_end;
        }
        return out;
    }

    namespace layers{
        Conv2D::Conv2D(Stream& file)
        : padding(file), stride_y(file), stride_x(file), weights_(file, 4), biases_(file), activation_(file) {}

        Tensor Conv2D::operator()(const Tensor& in) const noexcept {
            kassert(in.dims_.size() == 3);
            kassert(in.dims_[2] == weights_.dims_[3]);

            auto& ww = weights_.dims_;

            size_t offset_y = ww[1] - 1;
            size_t offset_x = ww[2] - 1;

            const Tensor *in2 = &in;
            Tensor padded_tensor;

            if (padding && (offset_x || offset_y)) {
                padded_tensor = pad_tensor(in, offset_x, offset_y);
                in2 = &padded_tensor;
            }

            auto tmp
                = Tensor::empty(
                    (in2->dims_[0] - offset_y) / stride_y,
                    (in2->dims_[1] - offset_x) / stride_x,
                    ww[0]);

            auto ws_ = cast(ww[3] * ww[2] * ww[1] * ww[0]);
            auto ws0 = cast(ww[3] * ww[2] * ww[1]);
            auto ws1 = cast(ww[3] * ww[2]);
            auto ws2 = cast(ww[3]);
            auto is0 = cast(ww[3] * in2->dims_[1]);

            auto ty = cast(tmp.dims_[0]);
            auto tx = cast(tmp.dims_[1]);

            auto w_ptr = weights_.begin();
            auto b_ptr = biases_.begin();
            auto t_ptr = std::back_inserter(tmp.data_);
            auto i_ptr = in2->begin();

            for (ptrdiff_t y = 0; y < ty; ++y)
                for (ptrdiff_t x = 0; x < tx; ++x) {
                    auto b_ = b_ptr;
                    auto i_ = i_ptr + y * is0 * stride_y + x * ws2 * stride_x;
                    for (auto w0 = w_ptr; w0 < w_ptr + ws_; w0 += ws0) {
                        auto tmp_ = 0.f;
                        auto i0 = i_;
                        for (auto w1 = w0; w1 < w0 + ws0; w1 += ws1, i0 += is0)
                            tmp_ = std::inner_product(w1, w1 + ws1, i0, tmp_);
                        *(++t_ptr) = *(b_++) + tmp_;
                    }
                }
            return activation_(tmp);
        }
    }
}