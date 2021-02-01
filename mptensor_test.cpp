#include <chrono>
#include <iostream>

#include "mptensor/mptensor.hpp"


// mpicxx test.cpp -o test -L./mptensor/build/src -I./mptensor/include -lmptensor -lscalapack -llapack -lblas -fopenmp
// g++ -pthread -O3 test2.cpp -o test -I./taskflow/ -std=c++17

using namespace mptensor;
typedef Tensor<scalapack::Matrix, double> ptensor;


int main(int argc, char *argv[])
{

  ptensor pt1(Shape({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2}));
  for (int i = 0; i < pt1.local_size(); i++){
    pt1[i] = i;
  }

  ptensor pt2(Shape({2,2,2,2,2,2,2,2,2,2}));
  for (int i = 0; i < pt2.local_size(); i++){
    pt2[i] = i;
  }
  
  auto ti1 = std::chrono::high_resolution_clock::now();
  auto pt3 = tensordot(pt1, pt2,{0,2,5},{0,2,5});
  auto ti2 = std::chrono::high_resolution_clock::now();

  auto time_normal = std::chrono::duration_cast<std::chrono::milliseconds>( ti2 - ti1 ).count();
  std::cout << "time mptensor = " << time_normal << std::endl;
  
  return 0;
}
