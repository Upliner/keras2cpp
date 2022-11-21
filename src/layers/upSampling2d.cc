#include "upSampling2d.h"
namespace keras2cpp{
    namespace layers{
        UpSampling2D::UpSampling2D(Stream& file)
        : y_(file), x_(file) {}

        Tensor UpSampling2D::operator()(const Tensor& in) const noexcept {
            kassert(in.ndim() == 3);

            const auto& iw = in.dims_;

            Tensor out {iw[0] * y_, iw[1] * x_, iw[2]};

            const auto channels = iw[2];
            const auto i_row = iw[1] * channels;
            const auto o_row = i_row * x_;

            auto o_ptr = out.begin();
            auto i_ptr = in.begin();

            for (auto i_end = i_ptr + (iw[0] * iw[1] * iw[2]); i_ptr < i_end;) {
                const auto row_start = o_ptr;
                for (auto row_end = i_ptr + i_row; i_ptr < row_end; i_ptr += channels) {
                    for (unsigned i = 0; i < x_; i++) {
                        const auto pix_end = i_ptr + channels;
                        std::copy(i_ptr, pix_end, o_ptr);
                        o_ptr += channels;
                    }
                }
                const auto row_end = o_ptr;
                for (unsigned i = 1; i < y_; i++) {
                    std::copy(row_start, row_end, o_ptr);
                    o_ptr += o_row;
                }
            }
            return out;
        }
    }
}
