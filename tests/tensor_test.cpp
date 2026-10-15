#include "tensor/tensor.hpp"
#include <algorithm>
#include <print>

int main() {
  std::vector v = {2, 3};
  NN::Tensor a;
  a.set_dims(v);
  if (a.data.size() != 0)
    std::println(stderr, "Test for set_dims failed: expected size(data) = 0, got {}", a.data.size());
  a.set_data({3, 2, 10, 4, 5, 6});

  NN::Tensor b = a;
  std::ranges::for_each(b.data, [](auto &x) { x++; });

  auto c = a + b;
  std::println("        a : {}", a.data);
  std::println("        b : {}", b.data);
  std::println("c = a + b : {}", c.data);
  std::println("    c * a : {}", NN::Tensor::mul(c, a).data);

  NN::Tensor{}.set_dims({1, 2}).set_data({2, 3}) + NN::Tensor{}.set_dims({1, 2}).set_data({3, 4});
  return 0;
}
