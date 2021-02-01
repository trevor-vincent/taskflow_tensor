#include "SerialTensor.hpp"
#include "TaskflowTensor.hpp"

#include <chrono>
#include <iostream>


int main(int argc, char *argv[])
{
  
  SerialTensor<double> st1({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  std::cout << "st1.LocalSize()*8/(1.e9) = " << st1.LocalSize()*8/(1.e9) << std::endl;
  std::cout << "log2(st1.LocalSize()) = " << log2(st1.LocalSize()) << std::endl;
  for (int i = 0; i < st1.LocalSize(); i++){
    st1[i] = i;
  }

  SerialTensor<double> st2({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
  for (int i = 0; i < st2.LocalSize(); i++){
    st2[i] = i;
  }
  
  TaskflowTensorContractor<double> contractor;
  contractor.AddContractionTask(st1,st2,{0,2,5},{0,2,5});
  auto ti3 = std::chrono::high_resolution_clock::now();
  auto & c = contractor.Contract();
  auto ti4 = std::chrono::high_resolution_clock::now();

  auto time_task = std::chrono::duration_cast<std::chrono::milliseconds>( ti4 - ti3 ).count();
  std::cout << "time task = " << time_task << std::endl;

  
  return 0;

}
