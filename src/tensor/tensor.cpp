#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace NN {
  size_t Tensor::checked_num_el(const std::vector<size_t> &s) {
    size_t n = 1;
    for (size_t x : s) {
      if (x == 0)
        throw std::invalid_argument("Tensor dimensions must be positive.");
      if (n > std::numeric_limits<size_t>::max() / x)
        throw std::overflow_error("Tensor size overflow");
      n *= x;
    }
    return n;
  }

  void Tensor::assign_shape(std::vector<size_t> &&dims, const bool clear) {
    const size_t n = checked_num_el(dims);
    shape = std::move(dims);
    finalize_shape(n, clear);
  }

  void Tensor::finalize_shape(const size_t n, const bool clear) {
    num_el = n;
  void Tensor::assign_shape(std::vector<size_t> &&dims, const bool clear) {
    const size_t n = checked_num_el(dims);
    shape = std::move(dims);
    finalize_shape(n, clear);
    data.resize(num_el);
    if (clear)
      data.clear();
  }

    data.resize(num_el);
  }

  void Tensor::valid_dim_pair(const Tensor &b) const {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes.");
    if (num_el == 0 or data.size() != num_el or b.data.size() != b.num_el)

    if (num_el == 0 or b.num_el == 0 or data.size() == 0 or b.data.size() == 0)
      throw std::invalid_argument("Tensor data is not initialized.");

    if (num_el != b.num_el or data.size() != num_el or b.data.size() != b.num_el)
      throw std::logic_error("Malformed num_el or data.");
  }

  Tensor &Tensor::set_data(std::vector<float> &&v) {
    if (num_el != v.size())
      throw std::invalid_argument("Data vectors are of different size.");
    data = std::move(v);
    return *this;
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    assign_shape(std::move(v), clear);
    return *this;
  }

  Tensor &Tensor::add(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::plus<>());
    return *this;
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.add(b);
  }

  Tensor Tensor::cadd(const Tensor &b) const {
    auto c = *this;
    return c.add(b);
  }

  Tensor &Tensor::sub(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::minus<>());
    return *this;
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.sub(b);
  }

  Tensor Tensor::csub(const Tensor &b) const {
    auto c = *this;
    return c.sub(b);
  }

  Tensor &Tensor::mul(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::multiplies<>());
    return *this;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.mul(b);
  }

  Tensor Tensor::cmul(const Tensor &b) const {
    auto c = *this;
    return c.mul(b);
  }

  Tensor Tensor::operator+(const Tensor &b) const {
    auto c = *this;
    return c.add(b);
  }

  Tensor &Tensor::operator+=(const Tensor &b) { return add(b); }

  Tensor Tensor::operator-(const Tensor &b) const {
    auto c = *this;
    return c.sub(b);
  }

  Tensor &Tensor::operator-=(const Tensor &b) { return sub(b); }
} // namespace NN
