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
  std::println();

  a.at(1, 2) = 7;
  std::println("a[1,2] = 7: {}", a.data);
  a[0, 1] = 8;
  std::println("a[0,1] = 8: {}", a.data);
  std::println();

  NN::Tensor t(2, 3);
  t.set_data({1, 2, 3, 4, 5, 6});

  t.transpose();
  std::println("transpose: {}", t.data);
  std::println("shape: {}", t.shape);
  std::println("strides: {}", t.strides);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      std::println("t[{},{}] = {}", i, j, t[i, j]);

  const NN::Tensor ct = t;
  std::println("const t[1,0] = {}", ct[1, 0]);
  std::println();

  try {
    NN::Tensor x(2, 3, 4);
    x.transpose();
  } catch (const std::invalid_argument &e) {
    std::println("transpose rank test: {}", e.what());
  }
  std::println();

  NN::Tensor r(2, 3);
  r.set_data({1, 2, 3, 4, 5, 6});
  r.transpose().transpose();
  std::println("double transpose: {}", r.data);
  std::println("double transpose shape: {}", r.shape);
  std::println("double transpose strides: {}", r.strides);
  std::println();

  t = NN::Tensor(2, 3, 2);
  t.set_data(3, 2, 5, 1, 6, 12, 9, 8, 10, 12, 14, 7);
  std::println("reduce test");
  std::println("data : {}", t.data);
  std::println("shape : {}", t.shape);
  std::println();
  for (size_t i = 0; i < 3; i++) {
    std::println("sum axis = {} : {}", i, t.csum(i).data);
    std::println("mean axis = {} : {}", i, t.cmean(i).data);
    std::println("max axis = {} : {}", i, t.cmax(i).data);
    std::println("min axis = {} : {}", i, t.cmin(i).data);
  }
  std::println();
  return 0;
}
