#include "binary.h"
namespace keras2cpp{
    namespace layers{
        Tensor Add::operator()(const Tensor& a, const Tensor& b) const noexcept {
            kassert(a.dims_ == b.dims_)
            Tensor out;
            out.dims_ = a.dims_;
            out.data_.resize(out.size());

            auto a_ptr = a.begin();
            auto b_ptr = b.begin();
            auto o_end = out.end();
            for (auto ptr = out.begin(); ptr < o_end; ++ptr) {
                *ptr = *a_ptr + *b_ptr;
                ++a_ptr;
                ++b_ptr;
            }
            return out;
        }

        Tensor Multiply::operator()(const Tensor& a, const Tensor& b) const noexcept {
            kassert(a.dims_ == b.dims_);
            Tensor out;
            out.dims_ = a.dims_;
            out.data_.resize(out.size());

            auto a_ptr = a.begin();
            auto b_ptr = b.begin();
            auto o_end = out.end();
            for (auto ptr = out.begin(); ptr < o_end; ++ptr) {
                *ptr = *a_ptr * *b_ptr;
                ++a_ptr;
                ++b_ptr;
            }
            return out;
        }

        Tensor Concatenate::operator()(const Tensor& a, const Tensor& b) const noexcept {
            kassert(a.dims_.size() == b.dims_.size());

            Tensor out;
            if (a.dims_.empty())
                return out;

            const auto dimlen = a.dims_.size() - 1;
            for (size_t i = 0; i < dimlen; i++) {
                kassert(a.dims_[i] == b.dims_[i]);
                out.dims_.push_back(a.dims_[i]);
            }
            const auto a_pix = a.dims_[dimlen];
            const auto b_pix = b.dims_[dimlen];
            out.dims_.push_back(a_pix + b_pix);
            out.data_.resize(out.size());

            const auto end = out.end();
            auto a_ptr = a.begin();
            auto b_ptr = b.begin();
            for (auto ptr = out.begin(); ptr < end;) {
                std::copy(a_ptr, a_ptr + a_pix, ptr);
                a_ptr += a_pix;
                ptr += a_pix;
                std::copy(b_ptr, b_ptr + b_pix, ptr);
                b_ptr += b_pix;
                ptr += b_pix;
            }
            return out;
        }
    }
}
