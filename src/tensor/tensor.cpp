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
    if (strides.size() != shape.size())
      return false;
    for (size_t i = shape.size(), acc = 1; i-- > 0;) {
      if (strides[i] != acc)
        return false;
      acc *= shape[i];
    }
    return true;
  }

  bool Tensor::empty() const { return num_el == 0 or data.size() == 0; }

  void Tensor::valid_dim_pair(const Tensor &b) const {
    if (shape != b.shape)
      throw std::invalid_argument("Tensors of different shapes.");

    if (not valid_strides() or not b.valid_strides())
      throw std::invalid_argument("Malformed strides.");

    if (empty() or b.empty())
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

  Tensor &Tensor::transpose() {
    if (shape.size() != 2)
      throw std::invalid_argument("Transpose requires 2D tensor.");
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    if (data.size() != num_el)
      throw std::invalid_argument("Malformed tensor data.");
    const size_t r = shape[0], c = shape[1];
    std::vector<float> out(num_el);
    for (size_t i = 0; i < r; i++)
      for (size_t j = 0; j < c; j++)
        out[j * r + i] = data[i * c + j];
    data = std::move(out);
    std::ranges::reverse(shape);
    strides = compute_strides(shape);
    return *this;
  }

  Tensor Tensor::ctranspose() const {
    auto c = *this;
    c.transpose();
    return c;
  }

  Tensor &Tensor::add(const float x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](float &v) { v += x; });
    return *this;
  }

  Tensor &Tensor::add(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::plus<>());
    return *this;
  }

  Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.add(b);
    return c;
  }

  Tensor Tensor::cadd(const float x) const {
    auto c = *this;
    c.add(x);
    return c;
  }

  Tensor Tensor::cadd(const Tensor &b) const {
    auto c = *this;
    c.add(b);
    return c;
  }

  Tensor &Tensor::sub(const float x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](float &v) { v -= x; });
    return *this;
  }

  Tensor &Tensor::sub(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::minus<>());
    return *this;
  }

  Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.sub(b);
    return c;
  }

  Tensor Tensor::csub(const float x) const {
    auto c = *this;
    c.sub(x);
    return c;
  }

  Tensor Tensor::csub(const Tensor &b) const {
    auto c = *this;
    c.sub(b);
    return c;
  }

  Tensor &Tensor::mul(const Tensor &b) {
    valid_dim_pair(b);
    std::ranges::transform(data, b.data, data.begin(), std::multiplies<>());
    return *this;
  }

  Tensor &Tensor::mul(const float x) {
    if (empty())
      throw std::invalid_argument("Empty tensor.");
    std::ranges::for_each(data, [x](float &v) { v *= x; });
    return *this;
  }

  Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.mul(b);
    return c;
  }

  Tensor Tensor::cmul(const float x) const {
    auto c = *this;
    c.mul(x);
    return c;
  }

  Tensor Tensor::cmul(const Tensor &b) const {
    auto c = *this;
    c.mul(b);
    return c;
  }

  Tensor &Tensor::matmul(const Tensor &b) {
    if (shape.size() != 2 or b.shape.size() != 2)
      throw std::invalid_argument("Tensor should be rank 2 for matmul");
    if (shape[1] != b.shape[0])
      throw std::invalid_argument("Incompatible tensors for matmul.");

    Tensor ret(shape[0], b.shape[1]);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < b.shape[1]; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < shape[1]; k++)
          sum += at(i, k) * b.at(k, j);
        ret.at(i, j) = sum;
      }
    }
    *this = std::move(ret);
    return *this;
  }

  Tensor Tensor::cmatmul(const Tensor &b) const {
    auto c = *this;
    c.matmul(b);
    return c;
  }

  Tensor Tensor::matmul(const Tensor &a, const Tensor &b) {
    auto c = a;
    c.matmul(b);
    return c;
  }

  Tensor Tensor::operator-() const {
    auto c = *this;
    std::ranges::for_each(c.data, [](float &x) { x = -x; });
    return c;
  }

  Tensor Tensor::operator+(const float x) const { return cadd(x); }

  Tensor operator+(const float x, const Tensor &a) { return a.cadd(x); }

  Tensor &Tensor::operator+=(const float x) { return add(x); }

  Tensor Tensor::operator+(const Tensor &b) const {
    auto c = *this;
    c.add(b);
    return c;
  }

  Tensor &Tensor::operator+=(const Tensor &b) { return add(b); }

  Tensor Tensor::operator-(const float x) const { return csub(x); }

  Tensor operator-(const float x, const Tensor &a) {
    auto c = -a;
    c.add(x);
    return c;
  }

  Tensor &Tensor::operator-=(const float x) { return sub(x); }

  Tensor &Tensor::operator-=(const Tensor &b) { return sub(b); }

  Tensor Tensor::operator-(const Tensor &b) const {
    auto c = *this;
    c.sub(b);
    return c;
  }

  Tensor Tensor::operator*(const float x) const { return cmul(x); }

  Tensor operator*(const float x, const Tensor &a) { return a.cmul(x); }

  Tensor &Tensor::operator*=(const float x) { return mul(x); }

  Tensor Tensor::operator/(const float x) const {
    if (x == 0.0f)
      throw std::domain_error("Division by zero.");
    return cmul(1 / x);
  }

  Tensor &Tensor::operator/=(const float x) {
    if (x == 0.0f)
      throw std::domain_error("Division by zero.");
    return mul(1 / x);
  }

  float Tensor::sum() const {
    if (empty())
      return 0;
    return std::ranges::fold_left(data, 0.0f, [](float s, float x) { return s + x; });
  }

  float Tensor::mean() const {
    if (empty())
      return 0;
    return sum() / num_el;
  }

  float Tensor::min() const {
    if (empty())
      throw std::domain_error("Cant find minimum of empty tensor.");
    return *std::ranges::min_element(data);
  }

  float Tensor::max() const {
    if (empty())
      throw std::domain_error("Cant find maximum of empty tensor.");
    return *std::ranges::max_element(data);
  }
} // namespace NN
