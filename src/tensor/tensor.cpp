#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace NN {
  Tensor &Tensor::operator=(const Tensor &b) {
    if (this != &b) {
      data = b.data;
      shape = b.shape;
      strides = b.strides;
      num_el = b.num_el;
    }
    return *this;
  }

  Tensor &Tensor::operator=(Tensor &&b) noexcept {
    if (this != &b) {
      data = std::move(b.data);
      shape = std::move(b.shape);
      strides = std::move(b.strides);
      num_el = b.num_el;
      b.num_el = 0;
    }
    return *this;
  }

  std::vector<size_t> Tensor::compute_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    size_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      strides[i] = acc;
      acc *= shape[i];
    }
    return strides;
  }

  void Tensor::assign_shape(std::vector<size_t> &&dims, const bool clear) {
    const size_t n = checked_num_el(dims);
    shape = std::move(dims);
    finalize_shape(n, clear);
  }

  void Tensor::finalize_shape(const size_t n, const bool clear) {
    num_el = n;
    strides.resize(shape.size());
    for (size_t i = 0, tmp = num_el; i < shape.size(); i++) {
      tmp /= shape[i];
      strides[i] = tmp;
    }
    data.resize(num_el);
    if (clear)
      data.clear();
  }

  bool Tensor::valid_strides() const {
    // NOTE: contiguous row-major; relax once non-contiguous views exist.
    return strides == compute_strides(shape);
  }

  void Tensor::valid_dim_pair(const Tensor &b) const {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes.");

    if (not valid_strides() or not b.valid_strides())
      throw std::invalid_argument("Malformed strides.");

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
