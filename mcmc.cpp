#include <iostream>
#include <vector>
#include <math.h>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"
#include "mcmc.h"
#include "helper.h"

using std::ref;

void MCMC_state::update_suffstats() 
{
    std::fill(suff_stats.begin(), suff_stats.end(), 0.0);
	for (size_t i = 0; i < cls_assgn.size(); i++) suff_stats[cls_assgn[i]]++;
	sumsq[0] = 0;
}

void MCMC_state::sample_sigma2()
{
	std::fill(sumsq.begin(), sumsq.end(), 0.0);
	for (size_t i = 0; i < cls_assgn.size(); i++)
	{
		double beta_1 = gsl_vector_get(beta1, i);
		double beta_2 = gsl_vector_get(beta2, i);
		double beta_3 = gsl_vector_get(beta3, i);

		double tmp = InvS.A11 * square(beta_1) + InvS.A22 * square(beta_2) + InvS.A33 * square(beta_3) + \
		2 * InvS.A12 * beta_1 * beta_2 + 2 * InvS.A13 * beta_1 * beta_3 + 2 * InvS.A23 * beta_2 * beta_3;
		
		sumsq[cls_assgn[i]] += tmp;
	}

	cluster_var[0] = 0;
    for (size_t i = 1; i < num_cluster; i++) 
	{
		double a = suff_stats[i] / 2.0 + para.a0k;
		double b = 1.0 / (sumsq[i] / 2.0 + para.b0k);
		cluster_var[i] = 1.0 / gsl_ran_gamma(r, a, b);
		if (isinf(cluster_var[i])) 
		{
	    	cluster_var[i] = 1e5;
	    	std::cerr << "Cluster variance is infintie." << std::endl;
		}
		else if (cluster_var[i] == 0) 
		{
	    	cluster_var[i] = 1e-10;
	    	std::cerr << "Cluster variance is zero." << std::endl;
		}
    }
}

void MCMC_state::calc_b(size_t j, const mcmc_data &dat1, const ldmat_data &ldmat_dat1, \
						const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3) 
{
    size_t start_i = dat1.boundary[j].first;
    size_t end_i = dat1.boundary[j].second;

    gsl_vector_view b_j1 = gsl_vector_subvector(b1, start_i, end_i - start_i);
    gsl_vector_view beta_j1 = gsl_vector_subvector(beta1, start_i, end_i - start_i);
	
	gsl_vector_view b_j2 = gsl_vector_subvector(b2, start_i, end_i - start_i);
    gsl_vector_view beta_j2 = gsl_vector_subvector(beta2, start_i, end_i - start_i);

	gsl_vector_view b_j3 = gsl_vector_subvector(b3, start_i, end_i - start_i);
    gsl_vector_view beta_j3 = gsl_vector_subvector(beta3, start_i, end_i - start_i);

    gsl_vector_const_view diag1 = gsl_matrix_const_diagonal(ldmat_dat1.B[j]);
	gsl_vector_const_view diag2 = gsl_matrix_const_diagonal(ldmat_dat2.B[j]);
	gsl_vector_const_view diag3 = gsl_matrix_const_diagonal(ldmat_dat3.B[j]);
    
    gsl_vector_memcpy(&b_j1.vector, &beta_j1.vector);
    gsl_vector_mul(&b_j1.vector, &diag1.vector);
    gsl_blas_dsymv(CblasUpper, -eta*eta, ldmat_dat1.B[j], &beta_j1.vector, \
	    eta*eta, &b_j1.vector);
    gsl_blas_daxpy(eta, ldmat_dat1.calc_b_tmp[j], &b_j1.vector);

	gsl_vector_memcpy(&b_j2.vector, &beta_j2.vector);
    gsl_vector_mul(&b_j2.vector, &diag2.vector);
    gsl_blas_dsymv(CblasUpper, -eta*eta, ldmat_dat2.B[j], &beta_j2.vector, \
	    eta*eta, &b_j2.vector);
    gsl_blas_daxpy(eta, ldmat_dat2.calc_b_tmp[j], &b_j2.vector);

	gsl_vector_memcpy(&b_j3.vector, &beta_j3.vector);
    gsl_vector_mul(&b_j3.vector, &diag3.vector);
    gsl_blas_dsymv(CblasUpper, -eta*eta, ldmat_dat3.B[j], &beta_j3.vector, \
	    eta*eta, &b_j3.vector);
    gsl_blas_daxpy(eta, ldmat_dat3.calc_b_tmp[j], &b_j3.vector);
}

void MCMC_state::sample_V() 
{
	vector<double> a(M - 1);
	a[M - 2] = suff_stats[M];
	for (int k = M - 3; k >= 0; k--) 
	{
		a[k] = suff_stats[k + 2] + a[k + 1];
    }
	double idx41 = 0;
    for (size_t j = 0; j < M - 1; j++) 
	{
		if (idx41 == 1)
		{
			V[j] = 0;
			continue;
		}
		V[j] = gsl_ran_beta(r, \
			1 + suff_stats[j + 1], \
			alpha + a[j]);
		if (V[j] == 1) idx41 = 1;
    }
    V[M - 1] = 1.0 - idx41;
}

void MCMC_state::sample_alpha() 
{
	double sum = 0;
    for (size_t j = 0; j < M; j++) 
	{
		if (V[j] == 1) break;
		sum += log(1 - V[j]);
	}
    alpha = gsl_ran_gamma(r, para.a0 + M - 1, 1.0/(para.b0 - sum));
}

void MCMC_state::update_p() 
{
	p[0] = theta;
	log_p[0] = logf(p[0] + 1e-40);

	vector<double> cumprod(M - 1);
	cumprod[0] = 1 - V[0];
	for (size_t i = 1; i < (M - 1); i++) 
	{
		cumprod[i] = cumprod[i - 1] * (1 - V[i]);
		if (V[i] == 1) 
		{
	    	std::fill(cumprod.begin() + i + 1, cumprod.end(), 0.0);
	    	break;
		}
    }
    p[1] = V[0]; 
	double sum = p[1];
    for (size_t i = 1; i < M - 1; i++) 
	{
		p[i + 1] = cumprod[i - 1] * V[i];
		sum = sum + p[i + 1];
    }
    if (1 - sum > 0) 
	{
		p[M] = 1 - sum;
    }
    else 
	{
		p[M] = 0;
    }
    for (size_t i = 0; i < M; i++) 
	{
		p[1 + i] = p[1 + i] * (1 - theta);
		log_p[1 + i] = logf(p[1 + i] + 1e-40); 
    }
}// Double check later

void MCMC_state::sample_theta()
{
	size_t M0 = 0;
	size_t M1 = 0;
	for (size_t i = 0; i < cls_assgn.size(); i++)
	{
		if (cls_assgn[i] == 0) M0++;
		if (cls_assgn[i] != 0) M1++;
	}
	theta = gsl_ran_beta(r, 1 + M0, 1 + M1);
}

void MCMC_state::compute_h2(const mcmc_data &dat1, const mcmc_data &dat2, const mcmc_data &dat3) 
{
    double h21_tmp = 0, h22_tmp = 0, h23_tmp = 0;
    h22 = 0; h21 = 0, h23 = 0;
    for (size_t j = 0; j < dat1.ref_ld_mat.size(); j++) 
	{
		size_t start_i1 = dat1.boundary[j].first;
		size_t end_i1 = dat1.boundary[j].second;

		size_t start_i2 = dat2.boundary[j].first;
		size_t end_i2 = dat2.boundary[j].second;

		size_t start_i3 = dat3.boundary[j].first;
		size_t end_i3 = dat3.boundary[j].second;

		gsl_vector *tmp1 = gsl_vector_alloc(end_i1 - start_i1);
		gsl_vector_view beta1_j = gsl_vector_subvector(beta1, \
	    	start_i1, end_i1 - start_i1);
		gsl_blas_dsymv(CblasUpper, 1.0, \
	    	dat1.ref_ld_mat[j], &beta1_j.vector, 0, tmp1);
		gsl_blas_ddot(tmp1, &beta1_j.vector, &h21_tmp);
		h21 += h21_tmp;

		gsl_vector *tmp2 = gsl_vector_alloc(end_i2 - start_i2);
		gsl_vector_view beta2_j = gsl_vector_subvector(beta2, \
	    	start_i2, end_i2 - start_i2);
		gsl_blas_dsymv(CblasUpper, 1.0, \
	    	dat2.ref_ld_mat[j], &beta2_j.vector, 0, tmp2);
		gsl_blas_ddot(tmp2, &beta2_j.vector, &h22_tmp);
		h22 += h22_tmp;

		gsl_vector *tmp3 = gsl_vector_alloc(end_i3 - start_i3);
		gsl_vector_view beta3_j = gsl_vector_subvector(beta3, \
	    	start_i3, end_i3 - start_i3);
		gsl_blas_dsymv(CblasUpper, 1.0, \
	    	dat3.ref_ld_mat[j], &beta3_j.vector, 0, tmp3);
		gsl_blas_ddot(tmp3, &beta3_j.vector, &h23_tmp);
		h23 += h23_tmp;

		gsl_vector_free(tmp1);
		gsl_vector_free(tmp2);
		gsl_vector_free(tmp3);
    }
}

void MCMC_state::sample_assignment(size_t j, const mcmc_data &dat1, const ldmat_data &ldmat_dat1,\
								   const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3) 
{
	size_t start_i = dat1.boundary[j].first;
    size_t end_i = dat1.boundary[j].second;

	size_t blk_size = end_i - start_i;

	vector<float> B1jj(blk_size);
    vector<float> b1j(blk_size); 
	vector<float> B2jj(blk_size);
    vector<float> b2j(blk_size); 
	vector<float> B3jj(blk_size);
    vector<float> b3j(blk_size); 
	float **prob = new float*[blk_size];
    float **tmp = new float*[blk_size];
	vector<float> rnd(blk_size);
    
    for (size_t i = 0; i < blk_size; i++) 
	{
		prob[i] = new float[num_cluster]; 
		tmp[i] = new float[num_cluster];

		B1jj[i] = gsl_matrix_get(ldmat_dat1.B[j], i, i);
		b1j[i] = gsl_vector_get(b1, start_i + i);

		B2jj[i] = gsl_matrix_get(ldmat_dat2.B[j], i, i);
		b2j[i] = gsl_vector_get(b2, start_i + i);

		B3jj[i] = gsl_matrix_get(ldmat_dat3.B[j], i, i);
		b3j[i] = gsl_vector_get(b3, start_i + i);
	
		prob[i][0] = log_p[0];
		tmp[i][0] = log_p[0];
		rnd[i] = gsl_rng_uniform(r);
    }

	double C1 = N1 * square(eta);
	double C2 = N2 * square(eta);
	double C3 = N3 * square(eta);

	for (size_t i = 0; i < blk_size; i++) 
	{
		double Nb1 = N1 * b1j[i];
		double Nb2 = N2 * b2j[i];
		double Nb3 = N3 * b3j[i];

		for (size_t k = 1; k < num_cluster; k++) 
		{
			double c1 = InvS.A12 / cluster_var[k];
			double c2 = InvS.A13 / cluster_var[k];
			double c3 = InvS.A23 / cluster_var[k];

			double a1 = C1 * B1jj[i] + InvS.A11 / cluster_var[k];
			double a2 = C2 * B2jj[i] + InvS.A22 / cluster_var[k];
			double a3 = C3 * B3jj[i] + InvS.A33 / cluster_var[k];

			double s1 = a1 * a2 * a3 + 2 * c1 * c2 * c3 - a1 * square(c3) - a3 * square(c1) - a2 * square(c2);

			double L11 = (a2 * a3 - square(c3)) / s1;
			double L22 = (a1 * a3 - square(c2)) / s1;
			double L33 = (a2 * a1 - square(c1)) / s1;
			double L12 = (c2 * c3 - c1 * a3) / s1;
			double L13 = (c1 * c3 - c2 * a2) / s1;
			double L23 = (c2 * c1 - c3 * a1) / s1;

	    	prob[i][k] = 0.5 * (L11 * square(Nb1) + L22 * square(Nb2) + L33 * square(Nb3) + \
				2 * L12 * Nb1 * Nb2 + 2 * L13 * Nb1 * Nb3 + 2 * L23 * Nb2 * Nb3) + log_p[k];
			tmp[i][k] = s1 * s0 * cluster_var[k] * cluster_var[k] * cluster_var[k];
		}
    }

	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = 1; k < num_cluster; k++) 
		{
	    	tmp[i][k] = logf(tmp[i][k]);
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = 1; k < num_cluster; k++) 
		{
	    	prob[i][k] = -0.5 * tmp[i][k] + prob[i][k];
		}
    }
	
	for (size_t i = 0; i < blk_size; i++) 
	{
		float max_elem = *std::max_element(&prob[i][0], &prob[i][num_cluster - 1]);
		float log_exp_sum = 0;
		for (size_t k = 0; k < num_cluster; k++) 
		{
	    	log_exp_sum += expf(prob[i][k] - max_elem);
		}

		log_exp_sum = max_elem + logf(log_exp_sum);
		cls_assgn[i + start_i] = num_cluster - 1;

		for (size_t k = 0; k < num_cluster - 1; k++) 
		{
	    	rnd[i] -= expf(prob[i][k] - log_exp_sum);
	    	if (rnd[i] < 0) 
			{
				cls_assgn[i + start_i] = k;
				break;
	    	}
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		delete[] prob[i]; delete tmp[i];
    }
	delete[] prob; delete[] tmp;
}

void MCMC_state::sample_beta(size_t j, const mcmc_data &dat1, ldmat_data &ldmat_dat1, \
							 ldmat_data &ldmat_dat2, ldmat_data &ldmat_dat3) 
{
    size_t start_i = dat1.boundary[j].first;
    size_t end_i = dat1.boundary[j].second;

    vector <size_t>causal_list;

    for (size_t i = start_i; i < end_i; i++) 
	{
		if (cls_assgn[i] > 0) 
		{
	    	causal_list.push_back(i);
		}
    }

    gsl_vector_view beta_1j = gsl_vector_subvector(beta1, \
	                            start_i, end_i - start_i);
	
	gsl_vector_view beta_2j = gsl_vector_subvector(beta2, \
	                            start_i, end_i - start_i);

	gsl_vector_view beta_3j = gsl_vector_subvector(beta3, \
	                            start_i, end_i - start_i);

    gsl_vector_set_zero(&beta_1j.vector);
	gsl_vector_set_zero(&beta_2j.vector);
	gsl_vector_set_zero(&beta_3j.vector);

	size_t causal_size = causal_list.size();
	size_t B_size = 3 * causal_size;

    if (B_size  == 0) 
	{
		ldmat_dat1.num[j] = 0;
		ldmat_dat1.denom[j] = 0;

		ldmat_dat2.num[j] = 0;
		ldmat_dat2.denom[j] = 0;

		ldmat_dat3.num[j] = 0;
		ldmat_dat3.denom[j] = 0;
		return;
    }

    gsl_vector *A_vec = gsl_vector_alloc(B_size);
    gsl_vector *A_vec2 = gsl_vector_alloc(B_size);

    double C1 = square(eta) * N1; 
	double C2 = square(eta) * N2; 
	double C3 = square(eta) * N3; 

    gsl_matrix* B = gsl_matrix_alloc(B_size, B_size);

	double value;
	double sigma_tmp;
	
	for (size_t i = 0; i < B_size; i++)
	{
		if (i < causal_size) 
		{
			gsl_vector_set(A_vec, i, N1 * eta * gsl_vector_get(ldmat_dat1.calc_b_tmp[j], \
		   		causal_list[i] - start_i));
		}
		if (i >= causal_size && i < 2 * causal_size) 
		{
			gsl_vector_set(A_vec, i, N2 * eta * gsl_vector_get(ldmat_dat2.calc_b_tmp[j], \
		   		causal_list[i - causal_size] - start_i));
		}
		if (i >= 2 * causal_size) 
		{
			gsl_vector_set(A_vec, i, N3 * eta * gsl_vector_get(ldmat_dat3.calc_b_tmp[j], \
		   		causal_list[i - 2 * causal_size] - start_i));
		}
		for (size_t k = 0; k < B_size; k++)
		{
			if (i < causal_size)
			{
				value = 0;
				sigma_tmp = cluster_var[cls_assgn[causal_list[i]]];
				if (k == i) value = InvS.A11 / sigma_tmp;
				if (k == (causal_size + i)) value = InvS.A12 / sigma_tmp;
				if (k == (2 * causal_size + i)) value = InvS.A13 / sigma_tmp;
				if (k < causal_size) 
				{
					value = value + C1 * ldmat_dat1.B[j] -> data[ldmat_dat1.B[j]-> \
					tda * (causal_list[i] - start_i) + causal_list[k] - start_i];
				}
				gsl_matrix_set(B, i, k, value);
				continue;
			}

			if (i >= causal_size && i < 2 * causal_size) 
			{
				value = 0;
				sigma_tmp = cluster_var[cls_assgn[causal_list[i - causal_size]]];
				if (k == i) value = InvS.A22 / sigma_tmp;
				if (k == (i - causal_size)) value = InvS.A12 / sigma_tmp;
				if (k == (causal_size + i)) value = InvS.A23 / sigma_tmp;
				
				if (k >= causal_size && k < 2 * causal_size)  
				{
					value = value + C2 * ldmat_dat2.B[j] -> data[ldmat_dat2.B[j]->\
					tda * (causal_list[i - causal_size] - start_i) + \
					causal_list[k - causal_size] - start_i];
				}
				gsl_matrix_set(B, i, k, value);
				continue;
			}
			if (i >= 2 * causal_size) 
			{
				value = 0;
				sigma_tmp = cluster_var[cls_assgn[causal_list[i - 2 * causal_size]]];
				if (k == i) value = InvS.A33 / sigma_tmp;
				if (k == (i - causal_size)) value = InvS.A23 / sigma_tmp;
				if (k == (i - 2 * causal_size)) value = InvS.A13 / sigma_tmp;

				if (k >= 2 * causal_size)   
				{
					value = value + C3 * ldmat_dat3.B[j] -> data[ldmat_dat3.B[j]-> \
					tda * (causal_list[i - 2 * causal_size] - start_i) + \
					causal_list[k - 2 * causal_size] - start_i];
				}
				gsl_matrix_set(B, i, k, value);
				continue;
			}
		}
	}

    gsl_vector_memcpy(A_vec2, A_vec);
    gsl_vector *beta_c = gsl_vector_alloc(B_size);
    
    for (size_t i = 0; i < B_size; i++) 
	{
		gsl_vector_set(beta_c, i, gsl_ran_ugaussian(r));
    }

    // (N B_gamma + \Sigma_0^-1) = L L^T
    gsl_linalg_cholesky_decomp1(B);

    // \mu = L^{-1} A_vec
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, \
	    CblasNonUnit, B, A_vec);

    // N(\mu, I)
    gsl_blas_daxpy(1.0, A_vec, beta_c);

    // X ~ N(\mu, I), L^{-T} X ~ N( L^{-T} \mu, (L L^T)^{-1} )
    gsl_blas_dtrsv(CblasLower, CblasTrans, \
	    CblasNonUnit, B, beta_c);

	gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat1.num[j]);
	gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat2.num[j]);
	gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat3.num[j]);
	
	gsl_matrix_view B1_view = gsl_matrix_submatrix(B, 0, 0, causal_size, causal_size);
    gsl_matrix *B1 = &B1_view.matrix;

	gsl_matrix_view B2_view = gsl_matrix_submatrix(B, causal_size, causal_size, causal_size, causal_size);
    gsl_matrix *B2 = &B2_view.matrix;

	gsl_matrix_view B3_view = gsl_matrix_submatrix(B, 2 * causal_size, 2 * causal_size, causal_size, causal_size);
    gsl_matrix *B3 = &B3_view.matrix;


    // compute eta related terms
    for (size_t i = 0; i < causal_size; i++) 
	{
		gsl_matrix_set(B1, i, i, \
		C1 * gsl_matrix_get(ldmat_dat1.B[j], 
	    	causal_list[i] - start_i, \
	    	causal_list[i] - start_i));
		
		gsl_matrix_set(B2, i, i, \
		C2 * gsl_matrix_get(ldmat_dat2.B[j], 
	    	causal_list[i] - start_i, \
	    	causal_list[i] - start_i));

		gsl_matrix_set(B3, i, i, \
		C3 * gsl_matrix_get(ldmat_dat3.B[j], 
	    	causal_list[i] - start_i, \
	    	causal_list[i] - start_i));
    }

	gsl_vector *beta_c1 = gsl_vector_alloc(causal_size);
	gsl_vector *A1_vec = gsl_vector_alloc(causal_size);

	gsl_vector *beta_c2 = gsl_vector_alloc(causal_size);
	gsl_vector *A2_vec = gsl_vector_alloc(causal_size);

	gsl_vector *beta_c3 = gsl_vector_alloc(causal_size);
	gsl_vector *A3_vec = gsl_vector_alloc(causal_size);

	for (size_t i = 0; i < causal_size; i++) 
	{
		gsl_vector_set(beta_c1, i, gsl_vector_get(beta_c, i));
    }

	for (size_t i = causal_size; i < 2 * causal_size; i++) 
	{
		gsl_vector_set(beta_c2, i - causal_size, gsl_vector_get(beta_c, i));
    }

	for (size_t i = 2 * causal_size; i < 3 * causal_size; i++) 
	{
		gsl_vector_set(beta_c3, i - 2 * causal_size, gsl_vector_get(beta_c, i));
    }

	double denom1, denom2, denom3;
    gsl_blas_dsymv(CblasUpper, 1.0, B1, beta_c1, 0, A1_vec);
    gsl_blas_ddot(beta_c1, A1_vec, &denom1);

	gsl_blas_dsymv(CblasUpper, 1.0, B2, beta_c2, 0, A2_vec);
    gsl_blas_ddot(beta_c2, A2_vec, &denom2);

	gsl_blas_dsymv(CblasUpper, 1.0, B3, beta_c3, 0, A3_vec);
    gsl_blas_ddot(beta_c3, A3_vec, &denom3);


	ldmat_dat1.denom[j] = (denom1 + denom2 + denom3) / square(eta);
	ldmat_dat2.denom[j] = (denom1 + denom2 + denom3) / square(eta);
	ldmat_dat3.denom[j] = (denom1 + denom2 + denom3) / square(eta);


    ldmat_dat1.num[j] /= eta;
	ldmat_dat2.num[j] /= eta;
	ldmat_dat3.num[j] /= eta;

    for (size_t i = 0; i < B_size; i++) 
	{
		if (i < causal_size)
		{
			gsl_vector_set(&beta_1j.vector, causal_list[i] - start_i, \
			gsl_vector_get(beta_c, i));
		}
		if (i >= causal_size && i < 2 * causal_size)
		{
			gsl_vector_set(&beta_2j.vector, causal_list[i - causal_size] - start_i, \
			gsl_vector_get(beta_c, i));
		}
		if (i >= 2 * causal_size)
		{
			gsl_vector_set(&beta_3j.vector, causal_list[i - 2 * causal_size] - start_i, \
			gsl_vector_get(beta_c, i));
		}
    }

    gsl_vector_free(A_vec);
	gsl_vector_free(A_vec2);
    gsl_vector_free(beta_c);
	gsl_vector_free(beta_c1);
	gsl_vector_free(beta_c2);
	gsl_vector_free(beta_c3);
	gsl_vector_free(A1_vec);
	gsl_vector_free(A2_vec);
	gsl_vector_free(A3_vec);
	gsl_matrix_free(B);
}

void MCMC_state::sample_eta(const ldmat_data &ldmat_dat) 
{
    double num_sum = std::accumulate(ldmat_dat.num.begin(), \
	    ldmat_dat.num.end(), 0.0);

    double denom_sum = std::accumulate(ldmat_dat.denom.begin(), \
	    ldmat_dat.denom.end(), 0.0);

    denom_sum += 1e-6;

    eta = gsl_ran_ugaussian(r) * sqrt(1.0/denom_sum) + \
	  num_sum / denom_sum;
}

void mcmc(const string &ref_path, const string &ss_path1, const string &ss_path2, \
		  const string &ss_path3, const string &valid_path, \
		  const string &ldmat_path1, const string &ldmat_path2, const string &ldmat_path3, \
		  unsigned N1, unsigned N2, unsigned N3, \
		  const string &out_path1, const string &out_path2, const string &out_path3, \
		  double a, double rho1, double rho2, double rho3, double a0k, double b0k, \
		  int iter, int burn, int thin, unsigned n_threads, int opt_llk, double c1, double c2, double c3)
{
	int n_pst = (iter - burn) / thin;
	cout << n_threads << endl;
    mcmc_data dat1;
	mcmc_data dat2;
	mcmc_data dat3;
	cout << "\nreading in population 1" << endl << endl;
    coord(ref_path, ss_path1, valid_path, ldmat_path1, dat1, N1, opt_llk);
	cout << "\nreading in population 2" << endl << endl;
	coord(ref_path, ss_path2, valid_path, ldmat_path2, dat2, N2, opt_llk);
	cout << "\nreading in population 3" << endl << endl;
	coord(ref_path, ss_path3, valid_path, ldmat_path3, dat3, N2, opt_llk);
	cout << endl << endl;

	if (dat1.beta_mrg.size() == 0 || dat2.beta_mrg.size() == 0) 
	{
		cout << "0 SNPs remained after coordination. Exit." << endl;
		return;
    }
	
	for (size_t i = 0; i < dat1.beta_mrg.size(); i++) 
	{
		dat1.beta_mrg[i] /= c1;
    }
	for (size_t i = 0; i < dat2.beta_mrg.size(); i++) 
	{
		dat2.beta_mrg[i] /= c2;
    }
	for (size_t i = 0; i < dat2.beta_mrg.size(); i++) 
	{
		dat3.beta_mrg[i] /= c3;
    }

	MCMC_samples samples = MCMC_samples(dat1.beta_mrg.size(), dat2.beta_mrg.size(), dat3.beta_mrg.size());
	int M = 1000;
	MCMC_state state = MCMC_state(dat1.beta_mrg.size(), M, a0k, b0k, N1, N2, N3, dat1, dat2, dat3, rho1, rho2, rho3);
	
	ldmat_data ldmat_dat1;
	ldmat_data ldmat_dat2;
	ldmat_data ldmat_dat3;
	solve_ldmat(dat1, ldmat_dat1, a, 1);
	solve_ldmat(dat2, ldmat_dat2, a, 1);
	solve_ldmat(dat3, ldmat_dat3, a, 1);
	size_t bd_size = dat1.boundary.size();
	
	state.update_suffstats();
	
	for (int j = 1; j < iter + 1; j++)
	{
		state.sample_sigma2();
		for (size_t i = 0; i < dat1.ref_ld_mat.size(); i++) 
		{
	    	state.calc_b(i, dat1, ldmat_dat1, ldmat_dat2, ldmat_dat3);
		}
		for (size_t i = 0; i < bd_size; i++)
		{
			state.sample_assignment(i, dat1, ldmat_dat1, ldmat_dat2, ldmat_dat3);
		}
		state.update_suffstats();
		state.sample_theta();
		state.sample_V();
		state.update_p();
		state.sample_alpha();
		for (size_t i = 0; i < bd_size; i++) 
		{
	    	state.sample_beta(i, dat1, ldmat_dat1, ldmat_dat2, ldmat_dat3);
		}
		state.sample_eta(ldmat_dat1);

		if ((j > burn) && (j % thin == 0)) 
		{
			state.compute_h2(dat1, dat2, dat3);
	    	samples.h21 += state.h21 * square(state.eta) / n_pst;
			samples.h22 += state.h22 * square(state.eta) / n_pst;
			samples.h23 += state.h23 * square(state.eta) / n_pst;
			
	    	gsl_blas_daxpy(state.eta/n_pst, state.beta1, \
		    	samples.beta1);
			gsl_blas_daxpy(state.eta/n_pst, state.beta2, \
		    	samples.beta2);
		    gsl_blas_daxpy(state.eta/n_pst, state.beta3, \
		    	samples.beta3);
		}
		if (j % 1 == 0)
		{
	    	state.compute_h2(dat1, dat2, dat3);
	    	cout << j << " iter. h21: " << state.h21 * square(state.eta) << \
			" max beta: " << gsl_vector_max(state.beta1) * state.eta;

			cout << ' ' << " iter. h22: " << state.h22 * square(state.eta) << \
			" max beta: " << gsl_vector_max(state.beta2) * state.eta;

			cout << ' ' << " iter. h23: " << state.h23 * square(state.eta) << \
			" max beta: " << gsl_vector_max(state.beta3) * state.eta \
		 	<< endl;
		}
	}

	cout << "h21: " << \
	samples.h21 << " max: " << \
	gsl_vector_max(samples.beta1) << endl;

	cout << "h22: " << \
	samples.h22 << " max: " << \
	gsl_vector_max(samples.beta2) << endl;

	cout << "h23: " << \
	samples.h23 << " max: " << \
	gsl_vector_max(samples.beta3) << endl;

	std::ofstream out1(out_path1);
    out1 << "SNP" << "\t" << \
	"A1" << "\t" << "beta1" << endl;

    for (size_t i = 0; i < dat1.beta_mrg.size(); i++) 
	{
	    double tmp = gsl_vector_get(samples.beta1, i);
		out1 << dat1.id[i] << "\t" << \
	    	dat1.A1[i] << "\t" <<  \
	    	tmp << endl; 
    }
    out1.close();

	std::ofstream out2(out_path2);
    out2 << "SNP" << "\t" << \
	"A1" << "\t" << "beta2" << endl;

    for (size_t i = 0; i < dat2.beta_mrg.size(); i++) 
	{
		double tmp = gsl_vector_get(samples.beta2, i);
		out2 << dat2.id[i] << "\t" << \
	    	dat2.A1[i] << "\t" <<  \
	    	tmp << endl; 
    }
    out2.close();

	std::ofstream out3(out_path3);
    out3 << "SNP" << "\t" << \
	"A1" << "\t" << "beta3" << endl;

    for (size_t i = 0; i < dat3.beta_mrg.size(); i++) 
	{
		double tmp = gsl_vector_get(samples.beta3, i);
		out3 << dat3.id[i] << "\t" << \
	    	dat3.A1[i] << "\t" <<  \
	    	tmp << endl; 
    }
    out3.close();

	for (size_t i = 0; i < dat1.ref_ld_mat.size(); i++) 
	{
		gsl_matrix_free(ldmat_dat1.A[i]);
		gsl_matrix_free(ldmat_dat1.B[i]);
		gsl_matrix_free(ldmat_dat1.L[i]);
		gsl_vector_free(ldmat_dat1.calc_b_tmp[i]);
		gsl_vector_free(ldmat_dat1.beta_mrg[i]);
		gsl_matrix_free(dat1.ref_ld_mat[i]);
    }
	for (size_t i = 0; i < dat2.ref_ld_mat.size(); i++) 
	{ 
		gsl_matrix_free(ldmat_dat2.A[i]);
		gsl_matrix_free(ldmat_dat2.B[i]);
		gsl_matrix_free(ldmat_dat2.L[i]);
		gsl_vector_free(ldmat_dat2.calc_b_tmp[i]);
		gsl_vector_free(ldmat_dat2.beta_mrg[i]);
		gsl_matrix_free(dat2.ref_ld_mat[i]);
    }
	for (size_t i = 0; i < dat3.ref_ld_mat.size(); i++) 
	{
		gsl_matrix_free(ldmat_dat3.A[i]);
		gsl_matrix_free(ldmat_dat3.B[i]);
		gsl_matrix_free(ldmat_dat3.L[i]);
		gsl_vector_free(ldmat_dat3.calc_b_tmp[i]);
		gsl_vector_free(ldmat_dat3.beta_mrg[i]);
		gsl_matrix_free(dat3.ref_ld_mat[i]);
    }
}