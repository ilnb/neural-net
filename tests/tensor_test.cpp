#include "tensor/tensor.hpp"
#include <algorithm>
#include <print>

int main() {
  NN::Tensor a(2, 3);
  std::vector v = {2, 3};
  a.set_dims(v);
  if (a.data.size() != 0)
    std::println(stderr, "Test for set_dims failed: expected size(data) = 0, got {}", a.data.size());
  a.set_data({3, 2, 10, 4, 5, 6});

  NN::Tensor b = a;
  std::ranges::for_each(b.data, [](auto &x) { x++; });

  auto c = a + b;
  auto d = NN::Tensor::mul(c, a);
  std::println("        a : {}", a.data);
  std::println("        b : {}", b.data);
  std::println("c = a + b : {}", c.data);
  std::println("d = c * a : {}", d.data);
  std::println("with utils: {}", a.cadd(b).mul(a).data);
  a.at(1, 2) = 7;
  std::println("a[1,2] = 7: {}", a.data);
  a[0, 1] = 8;
  std::println("a[0,1] = 8: {}", a.data);

  NN::Tensor{}.set_dims({1, 2}).set_data({2, 3}) + NN::Tensor{}.set_dims({1, 2}).set_data({3, 4});
  NN::Tensor(1, 2).at(0, 1);
  return 0;
}
