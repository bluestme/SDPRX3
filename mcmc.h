#ifndef MCMC_H
#define MCMC_H

#include <iostream>
#include <vector>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <math.h>
#include <unordered_map>
#include <string>
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"

#define square(x) ((x)*(x))
std::vector<double> selectElements(const std::vector<double>& inputVector, const std::vector<int>& indices);
using std::string;
using std::cout;
using std::endl;

typedef struct{
	double a0k;
	double b0k;
	double a0;
	double b0;
} hyper;

typedef struct{
	double A11;
	double A12;
	double A13;
	double A22;
	double A23;
	double A33;
} InvMat3;

typedef struct {
    std::vector<gsl_matrix*> A;
    std::vector<gsl_matrix*> B;
    std::vector<gsl_matrix*> L;
    std::vector<gsl_vector*> beta_mrg;
    std::vector<gsl_vector*> calc_b_tmp;
    std::vector<double> num;
    std::vector<double> denom;
} ldmat_data;

class MCMC_state {
    public:
	double alpha;
	double theta;

	double theta_k1;
	double theta_k2;
	double theta_k3;
	double sigma_k1;
	double sigma_k2;
	double sigma_k3;

	double eta;
	double h21, h22, h23;
	double rho1, rho2, rho3;
	double N1, N2, N3;
	double s0;
	size_t bd_size;

	size_t M;
	size_t num_cluster;

	gsl_vector *beta1;
	gsl_vector *beta2;
	gsl_vector *beta3;
	gsl_vector *b1;
	gsl_vector *b2;
	gsl_vector *b3;
	hyper para;
	InvMat3 InvS;
	std::vector<size_t> cls_assgn0;
	std::vector<size_t> cls_assgn1;
	std::vector<size_t> cls_assgn2;
	std::vector<size_t> cls_assgn3;
	std::vector<double> V;
	std::vector<double> p;
	std::vector<double> log_p;
	std::vector<double> cluster_var;
	std::vector<unsigned> suff_stats;
	std::vector<double> sumsq;
	std::vector<std::pair<size_t, size_t> > boundary1;
	std::vector<std::pair<size_t, size_t> > boundary2;
	std::vector<std::pair<size_t, size_t> > boundary3;

	std::vector<std::vector<size_t>> shared_idx1;
	std::vector<std::vector<size_t>> shared_idx2;
	std::vector<std::vector<size_t>> shared_idx3;
	std::vector<size_t> shared_idx_vec1;
	std::vector<size_t> shared_idx_vec2;
	std::vector<size_t> shared_idx_vec3;
	std::unordered_map<size_t, size_t> shared_map1;
	std::unordered_map<size_t, size_t> shared_map2;
	std::unordered_map<size_t, size_t> shared_map3;

	std::vector<std::vector<size_t>> pop_idx1;
	std::vector<std::vector<size_t>> pop_idx2;
	std::vector<std::vector<size_t>> pop_idx3;
	std::vector<size_t> pop_idx_vec1;
	std::vector<size_t> pop_idx_vec2;
	std::vector<size_t> pop_idx_vec3;
	std::unordered_map<size_t, size_t> pop_map1;
	std::unordered_map<size_t, size_t> pop_map2;
	std::unordered_map<size_t, size_t> pop_map3;

	MCMC_state(size_t num_snp1, size_t num_snp2, size_t num_snp3, int max_cluster, \
			   double a0, double b0, double sz1, double sz2, double sz3, \
			   mcmc_data &dat1, mcmc_data&dat2, mcmc_data&dat3,\
			   double rho_1, double rho_2, double rho_3) 
	{
		M = max_cluster;
		num_cluster = 1 + M;
		bd_size = dat1.boundary.size();
		for (size_t j = 0; j < bd_size; j++)
		{
			boundary1.push_back(std::make_pair(dat1.boundary[j].first, dat1.boundary[j].second));
		}
		for (size_t j = 0; j < bd_size; j++)
		{
			boundary2.push_back(std::make_pair(dat2.boundary[j].first, dat2.boundary[j].second));
		}
		for (size_t j = 0; j < bd_size; j++)
		{
			boundary3.push_back(std::make_pair(dat3.boundary[j].first, dat3.boundary[j].second));
		}

		N1 = sz1;
		N2 = sz2;
		N3 = sz3;
		para.a0k = a0;
		para.b0k = b0;
		para.a0 = 0.1;
		para.b0 = 0.1;

		rho1 = rho_1;
		rho2 = rho_2;
		rho3 = rho_3;
		s0 = - rho1 * rho1 - rho2 * rho2 - rho3 * rho3 + 2 * rho1 * rho2 * rho3 + 1;
		InvS.A11 = (1 - square(rho3)) / s0;
		InvS.A12 = (rho2 * rho3 - rho1) / s0;
		InvS.A13 = (rho1 * rho3 - rho2) / s0;
		InvS.A22 = (1 - square(rho2)) / s0;
		InvS.A23 = (rho1 * rho2 - rho3) / s0;
		InvS.A33 = (1 - square(rho1)) / s0;

		alpha = 1; theta = 0.5;
		theta_k1 = 0.5; theta_k2 = 0.5; theta_k3 = 0.5;
		sigma_k1 = 0; sigma_k2 = 0; sigma_k3 = 0; 
	    eta = 1; h21 = 0; h22 = 0; h23 = 0;
		r = gsl_rng_alloc(gsl_rng_default); // Note that variable r has been taken
		gsl_rng_set(r, 1760);

		set_idx(bd_size, dat1, dat2, dat3);

	    beta1 = gsl_vector_calloc(num_snp1);
		beta2 = gsl_vector_calloc(num_snp2);
		beta3 = gsl_vector_calloc(num_snp3);
	    b1 = gsl_vector_calloc(num_snp1);
		b2 = gsl_vector_calloc(num_snp2);
		b3 = gsl_vector_calloc(num_snp3);
	    p.assign(num_cluster, 1.0/num_cluster);
	    log_p.assign(num_cluster, 0);

	    for (size_t i = 0; i < num_cluster; i++) 
		{
			log_p[i] = logf(p[i] + 1e-40);
	    }

	    cluster_var.assign(num_cluster, 0.0);
	    suff_stats.assign(num_cluster, 0);
		sumsq.assign(num_cluster, 0);
		cls_assgn0.assign(shared_idx_vec1.size(), 0);
		cls_assgn1.assign(pop_idx_vec1.size(), 0);
		cls_assgn2.assign(pop_idx_vec2.size(), 0);
		cls_assgn3.assign(pop_idx_vec3.size(), 0);
		V.assign(M, 0);
	    
	    for (size_t i = 0; i < shared_idx_vec1.size(); i++) 
		{
			cls_assgn0[i] = gsl_rng_uniform_int(r, num_cluster);
		}

		for (size_t i = 0; i < pop_idx_vec1.size(); i++) 
		{
			cls_assgn1[i] = 1;
		}
		for (size_t i = 0; i < pop_idx_vec2.size(); i++) 
		{
			cls_assgn2[i] = 1;
		}
		for (size_t i = 0; i < pop_idx_vec3.size(); i++) 
		{
			cls_assgn3[i] = 1;
		}
	}

	~MCMC_state() {
	    gsl_vector_free(beta1);
		gsl_vector_free(beta2);
		gsl_vector_free(beta3);
	    gsl_vector_free(b1);
		gsl_vector_free(b2);
		gsl_vector_free(b3);
	    gsl_rng_free(r);
	}

	void sample_sigma2();
	void calc_b(size_t j, const mcmc_data &dat1, const mcmc_data &dat2,\
						const mcmc_data &dat3, const ldmat_data &ldmat_dat1, \
						const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3);

	void sample_assignment(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, \
						   const mcmc_data &dat3, const ldmat_data &ldmat_dat1, \
						   const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3);
						   
	void update_suffstats();
	void sample_V();
	void update_p();
	void sample_alpha();
	void sample_theta();
	void sample_beta(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, \
					const mcmc_data &dat3, ldmat_data &ldmat_dat1, \
					ldmat_data &ldmat_dat2, ldmat_data &ldmat_dat3);
	void compute_h2(const mcmc_data &dat1, const mcmc_data &dat2, const mcmc_data &dat3);
	void sample_eta(const ldmat_data &ldmat_dat);
	void set_idx(size_t bd_size, mcmc_data &dat1, mcmc_data &dat2, mcmc_data &dat3);

    private:
	gsl_rng *r;
	size_t find_index(std::vector<size_t>& vec, size_t num);
};

class MCMC_samples 
{
    public:
	gsl_vector *beta1;
	gsl_vector *beta2;
	gsl_vector *beta3;
	double h21;
	double h22;
	double h23;

	MCMC_samples(size_t num_snps1, size_t num_snps2, size_t num_snps3) 
	{
	    beta1 = gsl_vector_calloc(num_snps1);
		beta2 = gsl_vector_calloc(num_snps2);
		beta3 = gsl_vector_calloc(num_snps3);
	    h21 = 0;
		h22 = 0;
		h23 = 0;
	}

	~MCMC_samples() 
	{
	    gsl_vector_free(beta1);
		gsl_vector_free(beta2);
		gsl_vector_free(beta3);
	}
};

void mcmc(const string &ref_path1, const string &ref_path2, const string &ref_path3,\
          const string &ss_path1, const string &ss_path2, \
		  const string &ss_path3, const string &valid_path, \
		  const string &ldmat_path1, const string &ldmat_path2, const string &ldmat_path3, \
		  unsigned N1, unsigned N2, unsigned N3, \
		  const string &out_path1, const string &out_path2, const string &out_path3, \
		  double a, double rho1, double rho2, double rho3, double a0k, double b0k, \
		  int iter, int burn, int thin, unsigned n_threads, int opt_llk, double c1, \
		  double c2, double c3, int pop_ind);


# endif