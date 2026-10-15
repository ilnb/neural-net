#include "tensor/tensor.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace NN {
  Tensor &Tensor::set_data(std::vector<float> &&v) {
    if (num_el != v.size())
      throw std::invalid_argument("Data vectors are of different size.");
    data = std::move(v);
    return *this;
  }

  Tensor &Tensor::set_dims(std::vector<size_t> &&v, const bool clear) {
    if (shape.size() != v.size())
      throw std::invalid_argument("Dimension vectors are of different size.");
    shape = std::move(v);
    num_el = 1;
    for (size_t x : shape)
      num_el *= x;
    if (clear)
      data.clear();
    return *this;
  }

  Tensor &Tensor::add(const Tensor &b) {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes can't be added.");

    if (data.size() != b.data.size())
      throw std::invalid_argument("Data vectors are of different size.");

    std::ranges::transform(data, b.data, data.begin(), std::plus<>());
    return *this;
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.add(b);
  }

  Tensor &Tensor::sub(const Tensor &b) {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes can't be substracted.");

    if (data.size() != b.data.size())
      throw std::invalid_argument("Data vectors are of different size.");

    std::ranges::transform(data, b.data, data.begin(), std::minus<>());
    return *this;
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.sub(b);
  }

  Tensor &Tensor::mul(const Tensor &b) {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes can't be substracted.");

    if (data.size() != b.data.size())
      throw std::invalid_argument("Data vectors are of different size.");

    std::ranges::transform(data, b.data, data.begin(), std::multiplies<>());
    return *this;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    auto c = a;
    return c.mul(b);
  }

  Tensor Tensor::operator+(const Tensor &b) {
    auto c = *this;
    return c.add(b);
  }

  Tensor &Tensor::operator+=(const Tensor &b) { return add(b); }

  Tensor Tensor::operator-(const Tensor &b) {
    auto c = *this;
    return c.sub(b);
  }

  Tensor &Tensor::operator-=(const Tensor &b) { return sub(b); }
} // namespace NN
