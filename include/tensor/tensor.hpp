#pragma once

#include <concepts>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace NN {
  template <typename R, typename T>
  concept ConvIntervalOf = std::ranges::random_access_range<R> and std::convertible_to<std::ranges::range_reference_t<R>, T>;

  struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
    size_t num_el{};

    Tensor() = default;

    Tensor(const Tensor &b) = default;

    Tensor(Tensor &&b) noexcept : data(std::move(b.data)), shape(std::move(b.shape)), num_el(b.num_el) { b.num_el = 0; }

    Tensor &operator=(const Tensor &b) {
      if (this != &b) {
        data = b.data;
        shape = b.shape;
        num_el = b.num_el;
      }
      return *this;
    }

    Tensor &operator=(Tensor &&b) noexcept {
      if (this != &b) {
        data = std::move(b.data);
        shape = std::move(b.shape);
        num_el = b.num_el;
        b.num_el = 0;
      }
      return *this;
    }

    template <typename R>
      requires ConvIntervalOf<R, size_t>
    explicit Tensor(R &&r) : shape(std::ranges::begin(r), std::ranges::end(r)) {
      const size_t n = checked_num_el(std::forward<R>(r));
      finalize_shape(n, false);
    }

    explicit Tensor(std::initializer_list<size_t> r) : shape(r) {
      const size_t n = checked_num_el(std::forward<std::initializer_list<size_t>>(r));
      finalize_shape(n, false);
    }

    template <typename R>
      requires ConvIntervalOf<R, float> and std::ranges::sized_range<R>
    Tensor &set_data(R &&r) {
      if (std::ranges::size(r) != num_el)
        throw std::invalid_argument("Data ranges are of different size.");
      data.assign_range(r);
      return *this;
    }

    Tensor &set_data(std::vector<float> &&v);

    // Clears data vector if @clear is set
    template <typename R>
      requires ConvIntervalOf<R, size_t>
    Tensor &set_dims(R &&r, const bool clear = true) {
      assign_shape(std::forward<R>(r), clear);
      return *this;
    }

    // Clears data vector if @clear is set
    Tensor &set_dims(std::vector<size_t> &&v, const bool clear = true);

    Tensor &add(const Tensor &b);
    // Creates another tensor
    Tensor cadd(const Tensor &b) const;
    Tensor &sub(const Tensor &b);
    // Creates another tensor
    Tensor csub(const Tensor &b) const;
    Tensor &mul(const Tensor &b);
    // Creates another tensor
    Tensor cmul(const Tensor &b) const;

    static Tensor add(const Tensor &a, const Tensor &b);
    static Tensor sub(const Tensor &a, const Tensor &b);
    static Tensor mul(const Tensor &a, const Tensor &b);

    Tensor operator+(const Tensor &b) const;
    Tensor &operator+=(const Tensor &b);
    Tensor operator-(const Tensor &b) const;
    Tensor &operator-=(const Tensor &b);

  private:
    template <typename R>
      requires ConvIntervalOf<R, size_t>
    void assign_shape(R &&dims, const bool clear) {
      const size_t n = checked_num_el(dims);
      shape.assign_range(dims);
      finalize_shape(n, clear);
    }

    template <typename R>
      requires ConvIntervalOf<R, size_t>
    static size_t checked_num_el(R &&s) {
      if (std::ranges::empty(s))
        throw std::invalid_argument("Empty size vector.");
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

    void assign_shape(std::vector<size_t> &&dims, const bool clear);
    void valid_dim_pair(const Tensor &b) const;
    static size_t checked_num_el(const std::vector<size_t> &s);
    void finalize_shape(size_t n, bool clear);
  };
} // namespace NN
