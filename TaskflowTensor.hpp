#include "SerialTensor.hpp"


#include <memory>
#include <vector>
#include <taskflow/taskflow.hpp> 

extern "C" {
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
            double a[], int *lda, double b[], int *ldb, double *beta,
            double c[], int *ldc);

}

template <typename T>
class TaskflowTensorContractor {

private:
  tf::Taskflow taskflow_;
  std::vector<SerialTensor<T>> tensors_;
  std::vector<std::vector<size_t>> d_offset_;
  std::vector<std::vector<size_t>> axes_trans_;
  std::vector<size_t> rows_;
  std::vector<size_t> cols_;
  std::vector<std::vector<size_t>> dim_new_;
  std::vector<size_t> shape_c_;
  std::vector<bool> no_transpose_;

  size_t tensor_trans_a_;
  size_t tensor_trans_b_;
  size_t tensor_c_;
  
  size_t rank_row_c_;
  
public:
  void AddTransposeTask
  (
   tf::Task & task_init_c,
   size_t input_id,
   const std::vector<size_t> &axes,
   size_t urank_new,
   size_t output_id
   )
  {
    auto & input =  tensors_[input_id];    
    const size_t rank = input.GetShape().size();
    no_transpose_.push_back(false);
    
    //Check if a transpose is actually needed
    if (urank_new == input.GetUpperRank()) {
      bool no_transpose = true;
      auto & input_axes_map = input.GetAxesMap();
      for (size_t i = 0; i < rank; ++i) {
	if (axes[i] != i) {no_transpose=false; break;}
	if (input_axes_map[i] != i) {no_transpose=false; break;}
      }
      if (no_transpose){
	no_transpose_[no_transpose_.size() - 1] = true;
	return;
      }
    }

    auto & dim_old = input.GetShape();
    std::vector<size_t> dim_new(rank);
    for (size_t r = 0; r < rank; ++r){
      dim_new[r] = dim_old[axes[r]];
    }

    const size_t local_size = input.LocalSize();

    std::vector<size_t> d_offset(rank);
    size_t d_row(1), d_col(1);
    for (size_t r = 0; r < urank_new; ++r) {
      d_offset[r] = d_row;
      d_row *= dim_new[r];
    }
    for (size_t r = urank_new; r < rank; ++r) {
      d_offset[r] = d_col;
      d_col *= dim_new[r];
    }

    std::vector<size_t> axes_map(rank);
    for (size_t i = 0; i < rank; i++) axes_map[i] = i;
    
    // auto & axes_map = output.GetAxesMap();
    std::vector<size_t> axes_inv(rank);
    std::vector<size_t> axes_trans(rank);
    for (size_t r = 0; r < rank; ++r) {
      axes_inv[axes[r]] = r;
    }
    for (size_t r = 0; r < rank; ++r) {
      axes_trans[r] = axes_inv[axes_map[r]];
    }
    
    size_t output_rows = 1;
    for (size_t i = 0; i < urank_new; ++i) output_rows *= dim_new[i];
    size_t output_cols = 1;
    for (size_t i = urank_new; i < rank; ++i) output_cols *= dim_new[i];

    rows_.push_back(output_rows);
    cols_.push_back(output_cols);
    
    axes_trans_.push_back(axes_trans);
    d_offset_.push_back(d_offset);
    dim_new_.push_back(dim_new);

    auto task_output_init = taskflow_.emplace
      (
       [this, urank_new, input_id, output_id]
       {
	 tensors_[output_id] = SerialTensor<T>(dim_new_[input_id], urank_new) ;
       }
       );
    task_output_init.precede(task_init_c);
    
    for (size_t i = 0; i < local_size; ++i) {
      auto task_transpose = taskflow_.emplace
	(
	 [this, i, input_id, output_id, output_rows, urank_new, rank]
	 {
	   auto & input = tensors_[input_id];
	   auto & output = tensors_[output_id];
	   auto & axes_trans = axes_trans_[input_id];
	   auto & d_offset = d_offset_[input_id];
	   auto && index_new = input.GetTransposeIndex(i, axes_trans);
	   
	   size_t g_row(0), g_col(0); 
	   for (size_t r = 0; r < urank_new; ++r) {
	     g_row += index_new[r] * d_offset[r];
	   }
	   for (size_t r = urank_new; r < rank; ++r) {
	     g_col += index_new[r] * d_offset[r];
	   }
	   output[g_row + g_col * output_rows] = input[i];
	 }
	 );
      
      task_transpose.succeed(task_output_init);
      task_transpose.precede(task_init_c);
    }
    
    return;
  }

  
  void AddContractionTask
  (
   const SerialTensor<T> & a,
   const SerialTensor<T> & b,
   const std::vector<size_t> & axes_a,
   const std::vector<size_t> & axes_b
   )
  {
    auto & shape_a = a.GetShape();
    auto & shape_b = b.GetShape();

    auto a_rank = shape_a.size();
    auto b_rank = shape_b.size();
  
    std::vector<size_t> shape_c;
    const size_t rank_row_c = a_rank - axes_a.size();
    const size_t rank_col_c = b_rank - axes_b.size();
    shape_c.resize(rank_row_c + rank_col_c);

    std::vector<size_t> trans_axes_a;
    std::vector<size_t> trans_axes_b;
    size_t urank_a;
    size_t urank_b;

    {
      const size_t rank = a_rank;
      const size_t rank_row = rank - axes_a.size();
      const size_t rank_col = axes_a.size();
      size_t v[rank];
      for (size_t i = 0; i < rank; ++i) v[i] = i;
      for (size_t i = 0; i < rank_col; ++i) v[axes_a[i]] = rank;
      std::sort(v, v + rank);
      for (size_t i = 0; i < rank_col; ++i) v[rank_row + i] = axes_a[i];

      // trans_axes_a.assign(rank, v);
      trans_axes_a.resize(rank);
      trans_axes_a.assign(v,v+rank);
    
      urank_a = rank_row;

      for (size_t i = 0; i < rank_row; ++i) shape_c[i] = shape_a[v[i]];
    }

    {
      const size_t rank = b_rank;
      const size_t rank_row = axes_b.size();
      const size_t rank_col = rank - axes_b.size();
      size_t v[rank];
      for (size_t i = 0; i < rank; ++i) v[i] = i;
      for (size_t i = 0; i < rank_row; ++i) v[axes_b[i]] = 0;
      std::sort(v, v + rank);
      for (size_t i = 0; i < rank_row; ++i) v[i] = axes_b[i];

      // trans_axes_b.assign(rank, v);
      trans_axes_b.resize(rank);
      trans_axes_b.assign(v,v+rank);
      urank_b = rank_row;

      for (size_t i = 0; i < rank_col; ++i)
	shape_c[i + rank_row_c] = shape_b[v[i + rank_row]];
    }

    tensors_.resize(5);
    tensors_[0] = a;
    tensors_[1] = b;

    shape_c_ = shape_c;
    rank_row_c_ = rank_row_c;




    auto task_init_c = taskflow_.emplace(
					 [this] {
					   tensors_[tensor_c_] =
					     SerialTensor<T>(shape_c_,
							     rank_row_c_ );
					 });
  

    AddTransposeTask
      (
       task_init_c,
       0,
       trans_axes_a,
       urank_a,
       2
       );

    AddTransposeTask
      (
       task_init_c,
       1,
       trans_axes_b,
       urank_b,
       3
       );

    if(no_transpose_[0]) {tensor_trans_a_ = 0;}
    else {tensor_trans_a_ = 2;}
    if(no_transpose_[1]) {tensor_trans_b_ = 1;}
    else {tensor_trans_b_ = 3;}
    tensor_c_ = 4;
    size_t M = rows_[0];
    size_t K = cols_[0];
    size_t N = cols_[1];

    for (size_t m = 0; m < M; m++){
      auto task_contract = taskflow_.emplace([this, m, M, K, N]{
	auto & A = tensors_[tensor_trans_a_];
	auto & B = tensors_[tensor_trans_b_];
	auto & C = tensors_[tensor_c_];
	for(size_t n=0; n < N; n++) {
	  // C[m][n] = 0;
	  C[m + n * M] = 0;
	  for(size_t k=0; k < K; k++) {
	    // C[m][n] += A[m][k] * B[k][n];
	    C[m + n * M] += A[m + k * M] * B[k + n * K];
	  }
	}
      }
	);
      task_contract.succeed(task_init_c);
    }


    // auto task_contract = taskflow_.emplace([this]{

    // 	auto & A = tensors_[tensor_trans_a_];
    // 	auto & B = tensors_[tensor_trans_b_];
    // 	auto & C = tensors_[tensor_c_];
    // 	int m = rows_[0];
    // 	int n = cols_[1];
    // 	int k = cols_[0];

    // 	char transa = 'N';
    // 	char transb = 'N';
    // 	double alpha = 1.0;
    // 	double beta = 0.0;
    // 	int lda = rows_[0];
    // 	int ldb = rows_[1];
    // 	int ldc = rows_[0];

    // 	dgemm_(&transa, &transb, &m, &n, &k, &alpha, const_cast<double *>(&A.GetMatrix()[0]),
    // 	       &lda, const_cast<double *>(&B.GetMatrix()[0]), &ldb, &beta, &C.GetMatrix()[0], &ldc);
    // }
    //   );
    // task_contract.succeed(task_init_c);
    // if (!no_transpose_[0]){
    //   tf::Task trans_a_module_task = taskflow_.composed_of(trans_taskflow_a).name("trans_a_module");
    //   trans_a_module_task.precede(task_init_c);
    // }

    // if (!no_transpose_[1]){    
    //   tf::Task trans_b_module_task = taskflow_.composed_of(trans_taskflow_b).name("trans_b_module");
    //   trans_b_module_task.precede(task_init_c);
    // }
  }

  SerialTensor<T>& Contract(){
    tf::Executor executor;
    executor.run(taskflow_).wait();
    return tensors_[4];
  }

};
