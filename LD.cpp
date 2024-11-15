#include <string>
#include "SDPRX_io.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_matrix.h"
#include <thread>
#include <iostream>
#include <unordered_set>
#include "LD.h"
#include "helper_LD.h"
#include <algorithm>

using std::cout; using std::endl;
using std::string; using std::vector;
using std::pair; using std::ofstream; 
using std::thread; using std::ref;

void calc_ref_ld_shrink(size_t k, gsl_matrix **ref_ld_mat, gsl_matrix *snp0, \
	const vector <pair<size_t, size_t>> &boundary, size_t n_sample) 
{
    size_t left = boundary[k].first;
    size_t right = boundary[k].second;
    gsl_matrix_view subsnp0 = gsl_matrix_submatrix(snp0, left, 0, right - left, snp0 -> size2);
    gsl_matrix* snp = gsl_matrix_calloc(right - left, n_sample);
    gsl_matrix_memcpy(snp, &subsnp0.matrix);

    gsl_matrix *snp2 = gsl_matrix_calloc(right - left, n_sample);
    gsl_matrix_memcpy(snp2, snp);
    gsl_matrix_mul_elements(snp2, snp2);

    gsl_matrix *snp2_prod = gsl_matrix_calloc(right-left, right-left);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, snp2, snp2, 0.0, snp2_prod);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, snp, snp, 0.0, ref_ld_mat[k]);
    gsl_matrix_scale(ref_ld_mat[k], 1.0/n_sample);

    double num = 0, denom = 0;

    for (size_t i = 0; i<right-left; i++) 
    {
	    for (size_t j=0; j<right-left; j++) 
        {
	        if (i == j) continue;
	        num +=  gsl_matrix_get(snp2_prod, i, j) * n_sample / gsl_pow_3((double) n_sample-1) - \
	        gsl_pow_2(gsl_matrix_get(ref_ld_mat[k], i, j)) * gsl_pow_2((double) n_sample) / gsl_pow_3((double) n_sample-1); 
	    
	        denom += gsl_pow_2(gsl_matrix_get(ref_ld_mat[k], i, j));
	    }
    }

    double sr = 1.0 - num/denom;
    if (sr < 0) sr = 0;
    if (sr > 1) sr = 1;

    gsl_matrix *tg_mat = gsl_matrix_alloc(right-left, right-left);
    gsl_matrix_set_identity(tg_mat);
    gsl_matrix_scale(ref_ld_mat[k], sr);
    gsl_matrix_scale(tg_mat, 1.0-sr);

    gsl_matrix_add(ref_ld_mat[k], tg_mat);

    gsl_matrix_free(snp); gsl_matrix_free(snp2);
    gsl_matrix_free(snp2_prod); gsl_matrix_free(tg_mat);
}

void calc_ref_parallel(size_t i, const vector<size_t> *v, gsl_matrix **ref_ld_mat, \
	gsl_matrix *snp0, const vector <pair<size_t, size_t>> &boundary, size_t n_sample) {

    for (size_t k=0; k<v[i].size(); k++) 
    {
	    calc_ref_ld_shrink(v[i][k], ref_ld_mat, snp0, boundary, n_sample);
    }
}

bool myCmp(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) {
    return a.second > b.second;
}

void div_block(const string &pfile1, const string &pfile2, const string &pfile3, \
               const string &out_dir, unsigned chrom, size_t n_thread, double r2, int pop_ind) 
{
    string fam_path1 = pfile1 + ".fam";
    string bim_path1 = pfile1 + ".bim";
    string bed_path1 = pfile1 + ".bed";

    string fam_path2 = pfile2 + ".fam";
    string bim_path2 = pfile2 + ".bim";
    string bed_path2 = pfile2 + ".bed";

    string fam_path3 = pfile3 + ".fam";
    string bim_path3 = pfile3 + ".bim";
    string bed_path3 = pfile3 + ".bed";

    SnpInfo snpinfo1;
    SnpInfo snpinfo2;
    SnpInfo snpinfo3;

    size_t n_sample1 = get_nsamples(fam_path1.c_str());
    size_t n_sample2 = get_nsamples(fam_path2.c_str());
    size_t n_sample3 = get_nsamples(fam_path3.c_str());

    for (size_t i = 0; i < 23; i++) 
    {
	    snpinfo1.chr_idx[i] = 0; 
        snpinfo2.chr_idx[i] = 0; 
        snpinfo3.chr_idx[i] = 0; 
    }

    read_bim(bim_path1.c_str(), &snpinfo1);
    read_bim(bim_path2.c_str(), &snpinfo2);
    read_bim(bim_path3.c_str(), &snpinfo3);

    for (size_t i = 0; i < 23; i++) 
    {
	    cout << "chrom " << i+1 << " " << snpinfo1.chr_idx[i]  << " for population 1" << endl;
    }
    cout << endl << endl;
    for (size_t i = 0; i < 23; i++) 
    {
	    cout << "chrom " << i+1 << " " << snpinfo2.chr_idx[i]  << " for population 2" << endl;
    }
    cout << endl << endl;
    for (size_t i = 0; i < 23; i++) 
    {
	    cout << "chrom " << i+1 << " " << snpinfo3.chr_idx[i]  << " for population 3" << endl;
    }

    size_t left1 = snpinfo1.chr_idx[chrom - 1], right1 = snpinfo1.chr_idx[chrom];
    size_t left2 = snpinfo2.chr_idx[chrom - 1], right2 = snpinfo2.chr_idx[chrom];
    size_t left3 = snpinfo3.chr_idx[chrom - 1], right3 = snpinfo3.chr_idx[chrom];

    std::vector<size_t> idx1;
    std::vector<size_t> idx2;
    std::vector<size_t> idx3;

    size_t size_1 = 0; 
    size_t size_2 = 0;
    size_t size_3 = 0;

    if (pop_ind == 2)
    {
        std::unordered_set<std::string> set2(snpinfo2.id.begin() + left2, snpinfo2.id.begin() + right2);
        for (size_t i = left1; i < right1; i++) 
        {
            if (set2.count(snpinfo1.id[i]) > 0) {idx1.push_back(i); size_1++;}
        }
        for (size_t i = left2; i < right2; i++) {idx2.push_back(i); size_2++;}
        for (size_t i = left3; i < right3; i++) 
        {
            if (set2.count(snpinfo3.id[i]) > 0) {idx3.push_back(i); size_3++;}
        }
    }

    if (pop_ind == 3)
    {
        std::unordered_set<std::string> set3(snpinfo3.id.begin() + left3, snpinfo3.id.begin() + right3);
        for (size_t i = left1; i < right1; i++) 
        {
            if (set3.count(snpinfo1.id[i]) > 0) {idx1.push_back(i); size_1++;}
        }
        for (size_t i = left2; i < right2; i++) 
        {
            if (set3.count(snpinfo2.id[i]) > 0) {idx2.push_back(i); size_2++;}
        }
        for (size_t i = left3; i < right3; i++) {idx3.push_back(i); size_3++;}
    }

    gsl_matrix *snp1 = gsl_matrix_calloc(size_1, n_sample1);
    read_bed(snp1, bed_path1, n_sample1, left1, right1, idx1);

    gsl_matrix *snp2 = gsl_matrix_calloc(size_2, n_sample2);
    read_bed(snp2, bed_path2, n_sample2, left2, right2, idx2);

    gsl_matrix *snp3 = gsl_matrix_calloc(size_3, n_sample3);
    read_bed(snp3, bed_path3, n_sample3, left3, right3, idx3);

    scaling_LD(snp1);
    scaling_LD(snp2);
    scaling_LD(snp3);

    size_t *max_list;

    vector<pair<size_t, size_t>> boundary1;
    vector<pair<size_t, size_t>> boundary2;
    vector<pair<size_t, size_t>> boundary3;

    vector<pair<size_t, size_t>> blk_size1;
    vector<pair<size_t, size_t>> blk_size2;
    vector<pair<size_t, size_t>> blk_size3;

    size_t n_blk;

    if (pop_ind == 2)
    {
        max_list = find_ld(snp2, r2);
        size_t left_bound = 0;
        n_blk = 0;
        for (size_t i = 0; i < size_2; i++) 
        {
	        if (max_list[i] == i) 
            {
	            if (i + 1 - left_bound < 300 && i != size_2 - 1) continue;
	            boundary2.push_back(std::make_pair(left_bound, i + 1));
	            blk_size2.push_back(std::make_pair(n_blk, i + 1 - left_bound));
	            left_bound = i + 1;
	            n_blk++;
	        }
        }

        size_t bd_left1 = 0;
        size_t bd_left2 = 0;
        size_t bd_right1 = 0;
        size_t bd_right2 = 0;
        for (size_t i = 0; i < n_blk; i++)
        {
            std::unordered_set<std::string> snpset(snpinfo2.id.begin() + boundary2[i].first + left2, \
                                                   snpinfo2.id.begin() + boundary2[i].second + left2);
            for (size_t j = bd_left1; j < idx1.size(); j++)
            {
                if (snpset.find(snpinfo1.id[idx1[j]]) != snpset.end()) bd_right1++;
                if (snpset.find(snpinfo1.id[idx1[j]]) == snpset.end() || j == idx1.size() - 1)
                {
                    boundary1.push_back(std::make_pair(bd_left1, bd_right1));
                    blk_size1.push_back(std::make_pair(i, bd_right1 - bd_left1));
                    bd_left1 = bd_right1;
                    break;
                }
            }

            for (size_t j = bd_left2; j < idx3.size(); j++)
            {
                if (snpset.find(snpinfo3.id[idx3[j]]) != snpset.end()) bd_right2++;
                if (snpset.find(snpinfo3.id[idx3[j]]) == snpset.end() || j == idx3.size() - 1)
                {
                    boundary3.push_back(std::make_pair(bd_left2, bd_right2));
                    blk_size3.push_back(std::make_pair(i, right2 - bd_left2));
                    bd_left2 = bd_right2;
                    break;
                }
            }
        }

        std::sort(blk_size1.begin(), blk_size1.end(), myCmp);
        std::sort(blk_size2.begin(), blk_size2.end(), myCmp);
        std::sort(blk_size3.begin(), blk_size3.end(), myCmp);
    }
    
    if (pop_ind == 3)
    {
        max_list = find_ld(snp3, r2);
        size_t left_bound = 0;
        n_blk = 0;
        for (size_t i = 0; i < size_3; i++) 
        {
	        if (max_list[i] == i) 
            {
	            if (i + 1 - left_bound < 300 && i != size_3 - 1) continue;
	            boundary3.push_back(std::make_pair(left_bound, i + 1));
	            blk_size3.push_back(std::make_pair(n_blk, i + 1 - left_bound));
	            left_bound = i + 1;
	            n_blk++;
	        }
        }

        size_t bd_left1 = 0;
        size_t bd_left2 = 0;
        size_t bd_right1 = 0;
        size_t bd_right2 = 0;
        for (size_t i = 0; i < n_blk; i++)
        {
            std::unordered_set<std::string> snpset(snpinfo3.id.begin() + boundary3[i].first + left3, \
                                                   snpinfo3.id.begin() + boundary3[i].second + left3);

            for (size_t j = bd_left1; j < idx1.size(); j++)
            {
                if (snpset.find(snpinfo1.id[idx1[j]]) != snpset.end()) bd_right1++;
                if (snpset.find(snpinfo1.id[idx1[j]]) == snpset.end() || j == idx1.size() - 1)
                {
                    boundary1.push_back(std::make_pair(bd_left1, bd_right1));
                    blk_size1.push_back(std::make_pair(i, bd_right1 - bd_left1));
                    bd_left1 = bd_right1;
                    break;
                }
            }

            for (size_t j = bd_left2; j < idx2.size(); j++)
            {
                if (snpset.find(snpinfo2.id[idx2[j]]) != snpset.end()) bd_right2++;
                if (snpset.find(snpinfo2.id[idx2[j]]) == snpset.end() || j == idx2.size() - 1)
                {
                    boundary2.push_back(std::make_pair(bd_left2, bd_right2));
                    blk_size2.push_back(std::make_pair(i, bd_right2 - bd_left2));
                    bd_left2 = bd_right2;
                    break;
                }
            }
        }

        std::sort(blk_size1.begin(), blk_size1.end(), myCmp);
        std::sort(blk_size2.begin(), blk_size2.end(), myCmp);
        std::sort(blk_size3.begin(), blk_size3.end(), myCmp);        
    }    
    // blk_size: the first element is the number of block 
    // and the second is the size of the block
    cout << "Divided into " << n_blk << " indpenent blocks with max size for population1: " << blk_size1[0].second << " for population 2: " << blk_size2[0].second << " for population 3: " << blk_size3[0].second << endl;

    // calculate shrinkage ref ld mat

    gsl_matrix **ref_ld_mat1 = new gsl_matrix*[n_blk];
    gsl_matrix **ref_ld_mat2 = new gsl_matrix*[n_blk];
    gsl_matrix **ref_ld_mat3 = new gsl_matrix*[n_blk];

    for (size_t i = 0; i < n_blk; i++) 
    {
        double blk_sizei1 = boundary1[i].second - boundary1[i].first;
        double blk_sizei2 = boundary2[i].second - boundary2[i].first;
        double blk_sizei3 = boundary3[i].second - boundary3[i].first;

	    ref_ld_mat1[i] = gsl_matrix_calloc(blk_sizei1, blk_sizei1);
        ref_ld_mat2[i] = gsl_matrix_calloc(blk_sizei2, blk_sizei2);
        ref_ld_mat3[i] = gsl_matrix_calloc(blk_sizei3, blk_sizei3);
    }

    vector<thread> threads(n_thread);
    
    unsigned *bin1 = new unsigned[n_thread];
    unsigned *bin2 = new unsigned[n_thread];
    unsigned *bin3 = new unsigned[n_thread];

    for (size_t i = 0; i < n_thread; i++) 
    { 
        bin1[i] = 0; bin2[i] = 0; bin3[i] = 0;
    }

    vector<size_t> *v1 = new vector<size_t> [n_thread];
    vector<size_t> *v2 = new vector<size_t> [n_thread];
    vector<size_t> *v3 = new vector<size_t> [n_thread];
    
    for (size_t i = 0; i < n_blk; i++) 
    {
	    size_t idx = std::min_element(bin1, bin1 + n_thread) - bin1;
	    bin1[idx] += blk_size1[i].second*blk_size1[i].second;
	    v1[idx].push_back(blk_size1[i].first);
    }

    for (size_t i = 0; i < n_blk; i++)
    {
        size_t idx = std::min_element(bin2, bin2 + n_thread) - bin2;
        bin2[idx] += blk_size2[i].second*blk_size2[i].second;
	    v2[idx].push_back(blk_size2[i].first);
    }

    for (size_t i = 0; i < n_blk; i++)
    {
        size_t idx = std::min_element(bin3, bin3 + n_thread) - bin3;
        bin3[idx] += blk_size3[i].second*blk_size3[i].second;
	    v3[idx].push_back(blk_size3[i].first);
    }

    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i] = thread(calc_ref_parallel, i, ref(v1), ref(ref_ld_mat1), snp1, ref(boundary1), n_sample1);
    }
    
    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i].join();
    }

    string out_ldmat1 = out_dir + "/chr" + \
		       std::to_string(chrom) + "pop1.dat";
    FILE *f1 = fopen(out_ldmat1.c_str(), "wb");
    for (size_t i = 0; i < n_blk; i++) 
    {
	    gsl_matrix_fwrite(f1, ref_ld_mat1[i]);
	    gsl_matrix_free(ref_ld_mat1[i]);
    }
    fclose(f1);
    
    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i] = thread(calc_ref_parallel, i, ref(v2), ref(ref_ld_mat2), snp2, ref(boundary2), n_sample2);
    }
    
    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i].join();
    }
    
    string out_ldmat2 = out_dir + "/chr" + \
		       std::to_string(chrom) + "pop2.dat";
    
    FILE *f2 = fopen(out_ldmat2.c_str(), "wb");
    for (size_t i = 0; i < n_blk; i++) 
    {
	    gsl_matrix_fwrite(f2, ref_ld_mat2[i]);
	    gsl_matrix_free(ref_ld_mat2[i]);
    }
    fclose(f2);

    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i] = thread(calc_ref_parallel, i, ref(v3), ref(ref_ld_mat3), snp3, ref(boundary3), n_sample3);
    }
    
    for (size_t i = 0; i < n_thread; i++) 
    {
	    threads[i].join();
    }
    
    string out_ldmat3 = out_dir + "/chr" + \
		       std::to_string(chrom) + "pop3.dat";
    
    FILE *f3 = fopen(out_ldmat3.c_str(), "wb");
    for (size_t i = 0; i < n_blk; i++) 
    {
	    gsl_matrix_fwrite(f3, ref_ld_mat3[i]);
	    gsl_matrix_free(ref_ld_mat3[i]);
    }
    fclose(f3);

    gsl_matrix_free(snp1);
    gsl_matrix_free(snp2);
    gsl_matrix_free(snp3);
    
    string out_snpinfo1 = out_dir + "/chr" + \
			 std::to_string(chrom) + "_1.snpInfo";
    ofstream out1(out_snpinfo1);

    out1 << "start" << '\t' << "end" << endl;

    for (size_t i = 0; i < boundary1.size(); i++) 
    {
	    out1 << boundary1[i].first << '\t' << boundary1[i].second << endl;
    }

    out1 << endl;

    out1 << "SNP" << '\t' << "A1" << '\t' << "A2" << endl;
    for (size_t i = 0; i < size_1; i++) 
    {
	    out1 << snpinfo1.id[idx1[i]] << '\t' << snpinfo1.A1[idx1[i]] \
	        << '\t' << snpinfo1.A2[idx1[i]] << endl;
    }
    out1.close();

    // =====================

    string out_snpinfo2 = out_dir + "/chr" + \
			 std::to_string(chrom) + "_2.snpInfo";
    ofstream out2(out_snpinfo2);

    out2 << "start" << '\t' << "end" << endl;

    for (size_t i = 0; i < boundary2.size(); i++) 
    {
	    out2 << boundary2[i].first << '\t' << boundary2[i].second << endl;
    }

    out2 << endl;

    out2 << "SNP" << '\t' << "A1" << '\t' << "A2" << endl;
    for (size_t i = 0; i < size_2; i++) 
    {
	    out2 << snpinfo2.id[idx2[i]] << '\t' << snpinfo2.A1[idx2[i]] \
	        << '\t' << snpinfo2.A2[idx2[i]] << endl;
    }
    out2.close();

    // =====================

    string out_snpinfo3 = out_dir + "/chr" + \
			 std::to_string(chrom) + "_3.snpInfo";
    ofstream out3(out_snpinfo3);

    out3 << "start" << '\t' << "end" << endl;

    for (size_t i = 0; i < boundary3.size(); i++) 
    {
	    out3 << boundary3[i].first << '\t' << boundary3[i].second << endl;
    }

    out3 << endl;

    out3 << "SNP" << '\t' << "A1" << '\t' << "A2" << endl;
    for (size_t i = 0; i < size_3; i++) 
    {
	    out3 << snpinfo3.id[idx3[i]] << '\t' << snpinfo3.A1[idx3[i]] \
	        << '\t' << snpinfo3.A2[idx3[i]] << endl;
    }
    out3.close();

    delete[] max_list;
    delete[] ref_ld_mat1;
    delete[] ref_ld_mat2;
    delete[] ref_ld_mat3;
    delete[] bin1;
    delete[] bin2;
    delete[] bin3;
    delete[] v1;
    delete[] v2;
    delete[] v3;
}