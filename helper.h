#ifndef HELPER_H
#define HELPER_H
#include "parse_gen.h"
#include <algorithm>
#include <math.h>
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <string>


using std::ifstream; using std::string;
using std::vector; using std::unordered_map; 
using std::cout; using std::endl;
using std::pair; using std::find;
void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, const double a, unsigned sz, int opt_llk);

void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, \
	const double a, int opt_llk) 
{
    for (size_t i = 0; i < dat.ref_ld_mat.size(); i++) 
	{
		size_t size = dat.boundary[i].second - dat.boundary[i].first;
		gsl_matrix *A = gsl_matrix_alloc(size, size);
		gsl_matrix *B = gsl_matrix_alloc(size, size);
		gsl_matrix *L = gsl_matrix_alloc(size, size);
		gsl_matrix_memcpy(A, dat.ref_ld_mat[i]);
		gsl_matrix_memcpy(B, dat.ref_ld_mat[i]);
		gsl_matrix_memcpy(L, dat.ref_ld_mat[i]);


		if (opt_llk == 1) 
		{
	    	// (R + aNI) / N A = R via cholesky decomp
	    	// Changed May 21 2021 to divide by N
	    	// replace aN with a ???
	    	gsl_vector_view diag = gsl_matrix_diagonal(B);
	    	gsl_vector_add_constant(&diag.vector, a);
		}
		else 
		{
	    	// R_ij N_s,ij / N_i N_j
	    	// Added May 24 2021
	    	for (size_t j=0; j<size ; j++) 
			{
				for (size_t k=0; k<size; k++) 
				{
		    		double tmp = gsl_matrix_get(B, j, k);
		    		// if genotyped on two different arrays, N_s = 0
		    		size_t idx1 = j + dat.boundary[i].first;
		    		size_t idx2 = k + dat.boundary[i].first;
		    		if ( (dat.array[idx1] == 1 && dat.array[idx2] == 2) || \
			    	(dat.array[idx1] == 2 && dat.array[idx2] == 1) ) 
					{
						tmp = 0;
		    		}
		    	else 
				{
					tmp *= std::min(dat.sz[idx1], dat.sz[idx2]) / \
			       		(1.1 * dat.sz[idx1] * dat.sz[idx2]);
		    	}
		    	gsl_matrix_set(B, j, k, tmp);
				}
	    	}

	    	// force positive definite
	    	// B = Q \Lambda Q^T
	    	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(size);
	    	gsl_matrix *evac = gsl_matrix_alloc(size, size);
	    	gsl_matrix *eval = gsl_matrix_calloc(size, size);
	    	gsl_vector_view eval_diag = gsl_matrix_diagonal(eval);
	    	gsl_eigen_symmv(B, &eval_diag.vector, evac, w);

	    	// get minium of eigen value
	    	double eval_min = gsl_matrix_get(eval, 0, 0);
	    	for (size_t k=1; k<size; k++) 
			{
				double eval_k = gsl_matrix_get(eval, k, k);
				if (eval_k <= eval_min) 
				{
		    		eval_min = eval_k;
				}	
	    	}

	    	// restore lower half of B
	    	for (size_t j=0; j<size; j++) 
			{
				for (size_t k=0; k<j; k++) 
				{
		    		double tmp = gsl_matrix_get(B, k, j);
		    		gsl_matrix_set(B, j ,k, tmp);
				}
	    	}

	    	// if min eigen value < 0, add -1.1 * eval to diagonal
	    	for (size_t j=0; j<size; j++) 
			{
				if (eval_min < 0) 
				{
		    	gsl_matrix_set(B, j, j, \
			    	1.0/dat.sz[j+dat.boundary[i].first] - 1.1*eval_min);
				}
				else 
				{
		    		gsl_matrix_set(B, j, j, \
			    		1.0/dat.sz[j+dat.boundary[i].first]);
				}
			}
	    	gsl_matrix_free(evac);
	    	gsl_matrix_free(eval);
	    	gsl_eigen_symmv_free(w);
		}

		gsl_linalg_cholesky_decomp1(B);
		gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, \
			CblasNonUnit, 1.0, B, A);
		gsl_blas_dtrsm(CblasLeft, CblasLower, CblasTrans, \
		                CblasNonUnit, 1.0, B, A);
	
		// This creates A = (R + aNI)-1 R
	
		// Changed May 21 2021 to divide by N 
		/*if (opt_llk == 1) 
		{
	   		gsl_matrix_scale(A, sz);
		}*/

		// B = RA
		// Changed May 21 2021 as A may not be symmetric
		//gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, L, A, 0, B);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, A, 0, B);
	
		// L = R %*% R;
		gsl_matrix_mul_elements(L, L);

		// memory allocation for A^T beta_mrg
		// Changed May 21 2021 from A to A^T
		gsl_vector *beta_mrg = gsl_vector_alloc(size);
		for (size_t j = 0; j < size; j++) 
		{
	    	gsl_vector_set(beta_mrg, j, dat.beta_mrg[j+dat.boundary[i].first]);
		}
		gsl_vector *b_tmp = gsl_vector_alloc(size);

		//gsl_blas_dsymv(CblasUpper, 1.0, A, beta_mrg, 0, b_tmp);
		// Changed May 21 2021 from A to A^T why??
		gsl_blas_dgemv(CblasTrans, 1.0, A, beta_mrg, 0, b_tmp);

		ldmat_dat.A.push_back(A);
		ldmat_dat.B.push_back(B);
		ldmat_dat.L.push_back(L);
		ldmat_dat.calc_b_tmp.push_back(b_tmp);
		ldmat_dat.beta_mrg.push_back(beta_mrg);
		ldmat_dat.denom.push_back(0);
		ldmat_dat.num.push_back(0);
    }
}

#endif