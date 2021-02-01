#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

template<class T> std::ostream& operator<<(std::ostream& os, const std::vector<T> & v)
{
    os << '{';
    for (int i = 0; i<v.size(); ++i) {
        os << "  ";
        os << v[i];
    }
    os << '}';
    return os;
}




std::vector<size_t> UnravelIndex(unsigned long long int linear_index,
				 const std::vector<size_t> &multi_index_maxes) {

  //handle corner case
  if (multi_index_maxes.size() == 0)
    return {linear_index};
      
  std::vector<size_t> i(multi_index_maxes.size(), 0);
  unsigned int l = 0;
  int s0 = multi_index_maxes[0];
  i[l] = linear_index % s0;
  unsigned long long int gl = linear_index;

  while (l < i.size() - 1) {
    unsigned long long int glp1 = (gl - i[l]) / (multi_index_maxes[l]);
    i[l + 1] = glp1 % (multi_index_maxes[l + 1]);
    gl = glp1;
    l++;
  }
  return i;
}

template <typename T>
class SerialTensor {
  
private:
  std::vector<T> mat_;
  size_t n_row_;
  size_t n_col_;
  
  std::vector<size_t> shape_;
  std::vector<size_t> axes_map_;
  size_t upper_rank_;

public:

  SerialTensor(){}

  SerialTensor(const std::vector<size_t> & shape)
  {
    size_t rank = shape.size();
    size_t n_row = 1;
    size_t n_col = 1;
    upper_rank_ = rank / 2;
    axes_map_.resize(rank);
    for (size_t i = 0; i < rank; i++) axes_map_[i] = i;
    for (size_t i = 0; i < upper_rank_; ++i) n_row *= shape[i];
    for (size_t i = upper_rank_; i < rank; ++i) n_col *= shape[i];
    shape_ = shape;
    n_row_ = n_row;
    n_col_ = n_col;
    mat_.resize(n_row_ * n_col_);
  }

  SerialTensor(const std::vector<size_t> & shape,
	       size_t upper_rank)
  {
    size_t rank = shape.size();
    size_t n_row = 1;
    size_t n_col = 1;
    upper_rank_ = upper_rank;
    axes_map_.resize(rank);
    for (size_t i = 0; i < rank; i++) axes_map_[i] = i;
    for (size_t i = 0; i < upper_rank_; ++i) n_row *= shape[i];
    for (size_t i = upper_rank_; i < rank; ++i) n_col *= shape[i];
    shape_ = shape;
    n_row_ = n_row;
    n_col_ = n_col;
    mat_.resize(n_row_ * n_col_);
  }
  

  size_t LocalSize() const {
    return mat_.size();
  }
  
  const std::vector<size_t>& GetShape() const{
    return shape_;
  }

  std::vector<T>& GetMatrix() {
    return mat_;
  }

  const std::vector<T>& GetMatrix() const {
    return mat_;
  }

  const size_t& GetUpperRank() const{
    return upper_rank_;
  }

  size_t& GetUpperRank(){
    return upper_rank_;
  }

 size_t& GetRows(){
    return n_row_;
  }

 const size_t& GetRows() const{
    return n_row_;
  }


 size_t& GetCols(){
    return n_col_;
  }

 const size_t& GetCols() const{
    return n_col_;
  }
  
  
  std::vector<size_t>& GetShape(){
    return shape_;
  }

  T& operator[](size_t index){
    return mat_[index];
  }

  const T& operator[](size_t index) const {
    return mat_[index];
  }
  
  const std::vector<size_t>& GetAxesMap() const{
    return axes_map_;
  }
  
  std::vector<size_t>& GetAxesMap(){
    return axes_map_;
  }

  std::vector<size_t> GetTransposeIndex(size_t lindex,
					const std::vector<size_t> &axes_trans) const
  {

    const size_t rank = GetShape().size();
    const size_t rank0 = upper_rank_;
    const size_t rank1 = rank - rank0;  // lower_rank
    const size_t lindex_row = lindex % n_row_;
    const size_t lindex_col = lindex / n_row_;
    const size_t *axes_map0 = &(axes_map_[0]);
    const size_t *axes_map1 = &(axes_map_[rank0]);
    std::vector<size_t> index_new(rank);
    
    auto & Dim = GetShape();
    std::vector<size_t> dim0(rank0);
    std::vector<size_t> dim1(rank1);
    for (size_t i = 0; i < rank0; ++i) dim0[i] = (Dim[axes_map0[i]]);
    for (size_t i = 0; i < rank1; ++i) dim1[i] = (Dim[axes_map1[i]]);  

    std::div_t divresult;

    auto && map_row = UnravelIndex(lindex_row, dim0);
    auto && map_col = UnravelIndex(lindex_col, dim1);
    
    for (size_t i = 0; i < rank0; ++i) {
      index_new[axes_trans[i]] = map_row[i];
    }
    for (size_t i = 0; i < rank1; ++i) {
      index_new[axes_trans[i + rank0]] = map_col[i];
    }
    return index_new;
  }
};

template <typename T>
SerialTensor<T> Transpose
(
 const SerialTensor<T> &input,
 const std::vector<size_t> &axes,
 size_t urank_new
)
{
  const size_t rank = input.GetShape().size();

  //Check if a transpose is actually needed
  if (urank_new == input.GetUpperRank()) {
    bool no_transpose = true;
    auto & input_axes_map = input.GetAxesMap();
    for (size_t i = 0; i < rank; ++i) {
      if (axes[i] != i) {no_transpose=false; break;}
      if (input_axes_map[i] != i) {no_transpose=false; break;}
    }
    if (no_transpose) return input;
  }
  
  auto & dim_old = input.GetShape();
  std::vector<size_t> dim_new(rank);
  for (size_t r = 0; r < rank; ++r){
    dim_new[r] = dim_old[axes[r]];
  }

  // std::cout << "dim_new = " << dim_new << std::endl;
  SerialTensor<T> output(dim_new, urank_new);
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

  auto & axes_map = output.GetAxesMap();
  std::vector<size_t> axes_inv(rank);
  std::vector<size_t> axes_trans(rank);
  for (size_t r = 0; r < rank; ++r) {
    axes_inv[axes[r]] = r;
  }
  for (size_t r = 0; r < rank; ++r) {
    axes_trans[r] = axes_inv[axes_map[r]];
  }

  const size_t output_rows = output.GetRows();
  for (size_t i = 0; i < local_size; ++i) {
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
    
  return output;
}

template <typename T>
void MatrixMultiply
(
 std::vector<T> &A,
 std::vector<T> &B,
 std::vector<T> &C,
 size_t M,
 size_t K,
 size_t N
)
{
  for(size_t m=0; m < M; m++) {
    for(size_t n=0; n < N; n++) {
      // C[m][n] = 0;
      C[m + n * M] = 0;
      for(size_t k=0; k < K; k++) {
	// C[m][n] += A[m][k] * B[k][n];
	C[m + n * M] += A[m + k * M] * B[k + n * K];
      }
    }
  }
}

template <typename T>
 SerialTensor<T> Contract
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

  SerialTensor<T> c(shape_c, rank_row_c);

  auto && a_trans = Transpose(a, trans_axes_a, urank_a);
  auto && b_trans = Transpose(b, trans_axes_b, urank_b);
  
  MatrixMultiply(
		 a_trans.GetMatrix(),
		 b_trans.GetMatrix(),
		 c.GetMatrix(),
		 a_trans.GetRows(),
		 a_trans.GetCols(),
		 b_trans.GetCols()
		 );
		  
  return c;
}
	 
