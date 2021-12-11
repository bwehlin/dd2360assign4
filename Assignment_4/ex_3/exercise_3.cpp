#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <functional>

void time_call(const char* name, std::function<void()> fn)
{
  using clock = std::chrono::steady_clock;
  auto start = clock::now();
  fn();
  auto end = clock::now();
  using fdur = std::chrono::duration<float>;
  fdur fsec = end - start;
  std::cout << name << " took " << fsec.count() * 1e3 << "ms\n";
}

std::vector<float> create_vec(std::size_t n)
{
  std::vector<float> vec(n);
  std::generate(vec.begin(), vec.end(), []{ return rand(); });
  return vec;
}

void cpu_saxpy(const std::vector<float>& x, std::vector<float>& y, float a)
{
  auto n = x.size();
  for (auto i = 0ul; i < n; ++i)
  {
    y[i] += a * x[i];
  }
}

void gpu_saxpy(const std::vector<float>& x, std::vector<float>& y, float a)
{
  auto sz = x.size();
  auto* xx = x.data();
  auto* yy = y.data();

#pragma acc parallel loop copyin(xx[0:sz]) copy(yy[0:sz])
  for (auto i = 0ul; i < sz; ++i)
  {
    yy[i] += a * xx[i];
  }
}

bool compare(const std::vector<float>& a, const std::vector<float>& b)
{
  for (auto i = 0ul; i < a.size(); ++i)
  {
    if (std::fabs(a[i]-b[i]) > std::fabs(a[i]) / 1e6)
    {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " n_items\n";
    return EXIT_FAILURE;
  }

  std::size_t n_items = std::stol(argv[1]);
  std::cout << "Running with " << n_items << " items...\n";

  auto x = create_vec(n_items);
  auto ycpu = create_vec(n_items);
  auto ygpu = ycpu;

  auto a = .5f;

  time_call("CPU", [&]{ cpu_saxpy(x, ycpu, a); });
  time_call("GPU", [&]{ gpu_saxpy(x, ygpu, a); });

  std::cout << "Comparing CPU and GPU... ";
  auto ok = compare(ycpu, ygpu);
  std::cout << (ok ? "Correct!" : "Incorrect!") << '\n';

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

