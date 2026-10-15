#include "tensor/tensor.hpp"
#include <algorithm>
#include <print>

int main() {
  NN::Tensor a({2, 3});
  a.data = {3, 2, 10, 4, 5, 6};

  NN::Tensor b = a;
  std::ranges::for_each(b.data, [](auto &x) { x++; });

  auto c = a + b;
  std::println("{}", (NN::Tensor({1, 2}).set_data({2, 3}) + NN::Tensor({1, 2}).set_data({3, 4})).data);

  std::println("{} {} {}", c.data, a.data, NN::Tensor::mul(c, a).data);

  std::vector v = {1, 2};
  a.set_dims(v);
  std::println("{}", a.data.size());
  return 0;
}
