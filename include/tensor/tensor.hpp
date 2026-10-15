#pragma once

#include <concepts>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace NN {
  template <typename R, typename T>
  concept ConvRangeOf = std::ranges::input_range<R> and std::convertible_to<std::ranges::range_reference_t<R>, T>;

  struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
    size_t num_el{};

    Tensor() = default;
    Tensor(const Tensor &b) = default;
    Tensor(Tensor &&b) noexcept = default;
    virtual ~Tensor() = default;

    template <typename R>
      requires ConvRangeOf<R, int>
    Tensor(R &&dims) : shape(std::ranges::begin(dims), std::ranges::end(dims)), num_el(1) {
      for (size_t x : shape) {
        if (x == 0)
          throw std::invalid_argument("Tensor dimensions must be positive.");
        if (num_el > std::numeric_limits<size_t>::max() / x)
          throw std::overflow_error("Tensor size overflow");
        num_el *= x;
      }
      data.resize(num_el);
    }

    Tensor(std::initializer_list<size_t> r) : shape(r), num_el(1) {
      for (size_t x : shape) {
        if (x == 0)
          throw std::invalid_argument("Tensor dimensions must be positive.");
        if (num_el > std::numeric_limits<size_t>::max() / x)
          throw std::overflow_error("Tensor size overflow");
        num_el *= x;
      }
      data.resize(num_el);
    }

    template <typename R>
      requires ConvRangeOf<R, float>
    Tensor &set_data(R &&r) {
      if (num_el != r.size())
        throw std::invalid_argument("Data ranges are of different size.");
      data.assign_range(std::forward<R>(r));
      return *this;
    }

    Tensor &set_data(std::vector<float> &&v);

    // Invalidates data vector if @clear is set
    template <typename R>
      requires ConvRangeOf<R, size_t>
    Tensor &set_dims(R &&r, const bool clear = true) {
      if (shape.size() != r.size())
        throw std::invalid_argument("Dimension ranges are of different size.");
      shape.assign_range(std::forward<R>(r));
      num_el = 1;
      for (size_t x : shape)
        num_el *= x;
      if (clear)
        data.clear();
      return *this;
    }

    // Invalidates data vector if @clear is set
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    Tensor &add(const Tensor &b);
    Tensor &sub(const Tensor &b);
    Tensor &mul(const Tensor &b);

    static Tensor add(const Tensor &a, const Tensor &b);
    static Tensor sub(const Tensor &a, const Tensor &b);
    static Tensor mul(const Tensor &a, const Tensor &b);

    Tensor operator+(const Tensor &b);
    Tensor &operator+=(const Tensor &b);
    Tensor operator-(const Tensor &b);
    Tensor &operator-=(const Tensor &b);
  };
} // namespace NN
