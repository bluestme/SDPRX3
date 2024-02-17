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
	std::vector<size_t> cls_assgn;
	std::vector<double> V;
	std::vector<double> p;
	std::vector<double> log_p;
	std::vector<double> cluster_var;
	std::vector<unsigned> suff_stats;
	std::vector<double> sumsq;
	std::vector<std::pair<size_t, size_t> > boundary1;
	std::vector<std::pair<size_t, size_t> > boundary2;
	std::vector<std::pair<size_t, size_t> > boundary3;
	MCMC_state(size_t num_snp, int max_cluster, \
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
	    eta = 1; h21 = 0; h22 = 0; h23 = 0;
		r = gsl_rng_alloc(gsl_rng_default); // Note that variable r has been taken
		//gsl_rng_set(r, 1235);

	    beta1 = gsl_vector_calloc(num_snp);
		beta2 = gsl_vector_calloc(num_snp);
		beta3 = gsl_vector_calloc(num_snp);
	    b1 = gsl_vector_calloc(num_snp);
		b2 = gsl_vector_calloc(num_snp);
		b3 = gsl_vector_calloc(num_snp);
	    p.assign(num_cluster, 1.0/num_cluster);
	    log_p.assign(num_cluster, 0);

	    for (size_t i = 0; i < num_cluster; i++) 
		{
			log_p[i] = logf(p[i] + 1e-40);
	    }

	    cluster_var.assign(num_cluster, 0.0);
	    suff_stats.assign(num_cluster, 0);
		sumsq.assign(num_cluster, 0);
		cls_assgn.assign(num_snp, 0);
		V.assign(M, 0);
	    
	    for (size_t i = 0; i < num_snp; i++) 
		{
			cls_assgn[i] = gsl_rng_uniform_int(r, num_cluster);
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
	void calc_b(size_t j, const mcmc_data &dat1, \
				const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3);

	void sample_assignment(size_t j, size_t m, const mcmc_data &dat1,\
						   const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2, const ldmat_data &ldmat_dat3);
	
	void update_suffstats(size_t m);
	void sample_V(size_t m);
	void update_p(size_t m);
	void sample_alpha(size_t m);
	void sample_theta();
	void sample_beta(size_t j, size_t m, const mcmc_data &dat1, ldmat_data &ldmat_dat1, ldmat_data &ldmat_dat2, ldmat_data &ldmat_dat3);
	void compute_h2(const mcmc_data &dat1, const mcmc_data &dat2, const mcmc_data &dat3);
	void sample_eta(const ldmat_data &ldmat_dat);

    private:
	gsl_rng *r;
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

void mcmc(const string &ref_path, const string &ss_path1, const string &ss_path2, \
		  const string &ss_path3, const string &valid_path, \
		  const string &ldmat_path1, const string &ldmat_path2, const string &ldmat_path3, \
		  unsigned N1, unsigned N2, unsigned N3, \
		  const string &out_path1, const string &out_path2, const string &out_path3, \
		  double a, double rho1, double rho2, double rho3, double a0k, double b0k, \
		  int iter, int burn, int thin, unsigned n_threads, int opt_llk, double c1, double c2, double c3);


# endif