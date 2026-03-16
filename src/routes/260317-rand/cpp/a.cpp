#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
  std::cout << std::setprecision(17);
  double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
  for (uint64_t i = 0; i < 2000000000; i += 4) {
#ifdef JULIA
    uint64_t k0 = (i       & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    uint64_t k1 = ((i + 1) & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    uint64_t k2 = ((i + 2) & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    uint64_t k3 = ((i + 3) & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000;
    d0 += std::bit_cast<double>(k0) - 1;
    d1 += std::bit_cast<double>(k1) - 1;
    d2 += std::bit_cast<double>(k2) - 1;
    d3 += std::bit_cast<double>(k3) - 1;
#else
#ifdef PYTHON
    constexpr double scale = 1.0 / (1ULL << 52);
    d0 += static_cast<double>(i)     * scale;
    d1 += static_cast<double>(i + 1) * scale;
    d2 += static_cast<double>(i + 2) * scale;
    d3 += static_cast<double>(i + 3) * scale;
#endif
#endif
  }
  std::cout << (d0 + d1 + d2 + d3) << std::endl;
}
