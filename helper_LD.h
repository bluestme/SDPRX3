#ifndef HELPER_LD_H
#define HELPER_LD_H
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_map>
#include <math.h>
#include <algorithm>
#include "gsl/gsl_statistics.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "SDPRX_io.h"


std::vector<size_t> findMatchedId(const SnpInfo& S1, const SnpInfo& S2,\
                     const std::vector<size_t>& idx1, const std::vector<size_t>& idx2);

void scaling_LD(gsl_matrix *snp);

size_t* find_ld(gsl_matrix *snp1, gsl_matrix *snp2);

std::vector<size_t> findMatchedId(const SnpInfo& S1, const SnpInfo& S2, \
                                  const std::vector<size_t>& idx1, const std::vector<size_t>& idx2) 
{
    std::unordered_map<std::string, size_t> idToIndexMap;

    // Create a mapping of id values to their indexes in S2
    for (size_t i = 0; i < idx1.size(); i++) 
	{
        idToIndexMap[S1.id[idx1[i]]] = idx1[i];
    }

    std::vector<size_t> matchingIndexes;

    // Find matching indexes in S2 for each id in S1
    for (size_t j = 0; j < idx2.size(); j++) 
    {
        auto it = idToIndexMap.find(S2.id[idx2[j]]);
        if (it != idToIndexMap.end()) 
        {
            matchingIndexes.push_back(it->second);
        }
    }
    return matchingIndexes;
}

void scaling_LD(gsl_matrix *snp)
{
    size_t nrow = snp -> size1;
    size_t ncol = snp -> size2;
    for (size_t i = 0; i < nrow; i++)
    {
        double *geno = new double[ncol];
        for (size_t j = 0; j < ncol; j++) 
        {
            geno[j] = gsl_matrix_get(snp, i, j);
        }
        double mean = gsl_stats_mean(geno, 1, ncol);
        double sd = gsl_stats_sd_m(geno, 1, ncol, mean)*sqrt(ncol - 1)/sqrt(ncol);
        if (sd == 0) sd = 1;
        
        for (size_t j = 0; j < ncol; j++) 
        {
	        gsl_matrix_set(snp, i, j, (geno[j]-mean)/sd);
	    }
        delete[] geno;
    }
}

size_t* find_ld(gsl_matrix *snp, double r2)
{
    double cor = 0; 
    size_t nrow = snp -> size1, ncol = snp -> size2;

    size_t *max_list = new size_t[nrow];
    gsl_vector_view snp1, snp2;

    for (size_t i = 0; i < nrow; i++) 
    {
        size_t left;
        if (i < 300) left = 0;
        if (i >= 300) left = i - 300;
        max_list[i] = i;
	    
	    snp1 = gsl_matrix_row(snp, i);

	    for (size_t j = left; j < i + 300; j++) 
        {
            if (j >= nrow) continue;
	        snp2 = gsl_matrix_row(snp, j);
	        gsl_blas_ddot(&snp1.vector, &snp2.vector, &cor);           
	        cor /= ncol;
	        if (cor * cor > r2) 
            {
                max_list[i] = j;
            }
    	}
	    if (i == 0) continue;
	    if (max_list[i] < max_list[i - 1]) 
        {
	        max_list[i] = max_list[i - 1];
	    }
    }
    return max_list;
}

#endif