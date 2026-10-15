#pragma once

#include <concepts>
#include <initializer_list>
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
      requires ConvRangeOf<R, size_t>
    explicit Tensor(R &&dims) : shape(std::ranges::begin(dims), std::ranges::end(dims)) {
      update_to_shape();
    }

    explicit Tensor(std::initializer_list<size_t> r) : shape(r) { update_to_shape(); }

    template <typename R>
      requires ConvRangeOf<R, float>
    Tensor &set_data(R &&r) {
      std::vector<float> v;
      v.assign_range(std::forward<R>(r));
      if (v.size() != num_el)
        throw std::invalid_argument("Data ranges are of different size.");
      data = std::move(v);
      return *this;
    }

    Tensor &set_data(std::vector<float> &&v);

    // Invalidates data vector if @clear is set
    template <typename R>
      requires ConvRangeOf<R, size_t>
    Tensor &set_dims(R &&r, const bool clear = true) {
      std::vector<size_t> s(std::ranges::begin(r), std::ranges::end(r));
      const size_t n = checked_num_el(s);
      shape = std::move(s);
      num_el = n;
      data.resize(num_el);
      if (clear)
        data.clear();
      return *this;
    }

    // Invalidates data vector if @clear is set
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
    void update_to_shape();
    void valid_dim_pair(const Tensor &b) const;
    size_t checked_num_el(const std::vector<size_t> &s);
  };
} // namespace NN
