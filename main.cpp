
#include <iostream>
#include "mcmc.h"
#include <string.h>
#include "time.h"
#include "LD.h"

using std::string;
using std::cout;
using std::endl;

int main(int argc, char *argv[]) 
{
    string ss_path1, ref_prefix1, ref_dir, valid_bim;
	string ss_path2, ref_prefix2, out_path1, out_path2;
	string ss_path3, ref_prefix3, out_path3;
    int N1 = 0, N2 = 0, N3 = 0, n_threads = 1, chr = 0;
    double a = 0.1, c1 = 1, c2 = 1, c3 = 1, a0k = .5, b0k = .5;
	double r2 = .1, rho1 = .8, rho2 = .8, rho3 = .8;
    int iter = 1000, burn = 200, thin = 1;
    int make_ref = 0, run_mcmc = 0;
    int opt_llk = 1;

    // pass command line arguments
    int i = 1;
    while (i < argc) 
	{
		char *end;

		if (strcmp(argv[i], "-ss1") == 0) 
		{
	    	ss_path1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ss2") == 0) 
		{
	    	ss_path2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ss3") == 0) 
		{
	    	ss_path3 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix1") == 0) 
		{
	    	ref_prefix1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix2") == 0) 
		{
	    	ref_prefix2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix3") == 0) 
		{
	    	ref_prefix3 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_dir") == 0) 
		{
	    	ref_dir = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-valid") == 0) 
		{
	    	valid_bim = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-out1") == 0) 
		{
	    	out_path1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-out2") == 0) 
		{
	    	out_path2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-out3") == 0) 
		{
	    	out_path3 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-rho1") == 0) 
		{
	    	rho1 = strtod(argv[i + 1], &end);
	    	i += 2;
		}
		else if (strcmp(argv[i], "-rho2") == 0) 
		{
	    	rho2 = strtod(argv[i + 1], &end);
	    	i += 2;
		}
		else if (strcmp(argv[i], "-rho3") == 0) 
		{
	    	rho3 = strtod(argv[i + 1], &end);
	    	i += 2;
		}
		else if (strcmp(argv[i], "-chr") == 0) 
		{   
	    	chr = strtol(argv[i + 1], &end, 10);
	    	if (*end != 0 || chr > 22 || chr < 0) 
			{
				cout << "Incorrect chromosome: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-N1") == 0) 
		{
	    	N1 = strtol(argv[i+1], &end, 10);
	    	if (*end != '\0' || N1 <= 0) 
			{
				cout << "Incorrect N1: " << argv[i+1] << endl;
				return 0;
	    	}
	    	if (N1 <= 1000) 
			{
				cout << "Warning: sample size too small for population 1, might" \
		    		" not achieve good performance." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-N2") == 0) 
		{
	    	N2 = strtol(argv[i+1], &end, 10);
	    	if (*end != '\0' || N2 <= 0) 
			{
				cout << "Incorrect N2: " << argv[i+1] << endl;
				return 0;
	    	}
	    	if (N2 <= 1000) 
			{
				cout << "Warning: sample size too small for population 2, might" \
		    		" not achieve good performance." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-N3") == 0) 
		{
	    	N3 = strtol(argv[i+1], &end, 10);
	    	if (*end != '\0' || N3 <= 0) 
			{
				cout << "Incorrect N3: " << argv[i+1] << endl;
				return 0;
	    	}
	    	if (N3 <= 1000) 
			{
				cout << "Warning: sample size too small for population 2, might" \
		    		" not achieve good performance." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-a") == 0) 
		{
	    	a = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || a < 0) 
			{
				cout << "Incorrect a: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-c1") == 0) 
		{
	    	c1 = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || c1 <= 0) 
			{
				cout << "Incorrect c1: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-c2") == 0) 
		{
	    	c2 = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || c2 <= 0) 
			{
				cout << "Incorrect c2: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-c3") == 0) 
		{
	    	c3 = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || c3 <= 0) 
			{
				cout << "Incorrect c3: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-a0k") == 0) 
		{
	    	a0k = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || a0k <= 0) 
			{
				cout << "Incorrect a0k: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-b0k") == 0) 
		{
	    	b0k = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || b0k <= 0) 
			{
				cout << "Incorrect b0k: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-iter") == 0) 
		{
	    	iter = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || iter <= 0) 
			{
				cout << "Incorrect iteration: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-burn") == 0) 
		{
	    	burn = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || burn <= 0) 
			{
				cout << "Incorrect number of iterations: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	if (burn >= iter) 
			{
				cout << "Error: burnin is larger than number of iterations." << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-thin") == 0) 
		{
	    	thin = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || thin <= 0) 
			{
				cout << "Incorrect thin: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-n_threads") == 0) 
		{
	    	n_threads = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || n_threads <= 0) 
			{
				cout << "Incorrect number of threads: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-r2") == 0) 
		{
	    	r2 = strtod(argv[i + 1], &end);
	    	if (r2 <= 0) 
			{
				cout << "Incorrect r2: " << argv[i+1] << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-opt_llk") == 0) 
		{
	    	opt_llk = strtol(argv[i + 1], &end, 10);
	    	if (opt_llk != 1 && opt_llk != 2) 
			{
				cout << "opt_llk must be in 1 or 2." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-make_ref") == 0) 
		{
	    	make_ref = 1;
	    	i++;
		}
		else if (strcmp(argv[i], "-mcmc") == 0) 
		{
	    	run_mcmc = 1;
	    	i++;
		}
		else if (strcmp(argv[i], "-ref_prefix1") == 0) 
		{
	    	ref_prefix1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix2") == 0) 
		{
	    	ref_prefix2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix3") == 0) 
		{
	    	ref_prefix3 = argv[i + 1];
	    	i += 2;
		}
		else 
		{
	    	cout << "Invalid option: " << argv[i] << endl;
	    	return 0;
		}
    }

    if (make_ref && run_mcmc) 
	{
		cout << "both -make_ref and -mcmc are specified. Please specify one of them." << endl;
		return 0;
    }

    if (!chr) 
	{
		cout << "Invalid chromosome specified." << endl;
		return 0;
    }

    if (ref_dir.empty()) 
	{
		cout << "Did not specify the directory of reference LD." << endl;
		return 0;
    }

	if (make_ref) 
	{

		if (ref_prefix1.empty()) 
		{
	    	cout << "Did not specify the prefix of the bim file for the reference panel for population 1." << endl;
	    	return 0;
		}
		if (ref_prefix2.empty()) 
		{
	    	cout << "Did not specify the prefix of the bim file for the reference panel for population 2." << endl;
	    	return 0;
		}
		if (ref_prefix3.empty()) 
		{
	    	cout << "Did not specify the prefix of the bim file for the reference panel for population 3." << endl;
	    	return 0;
		}
		div_block(ref_prefix1, ref_prefix2, ref_prefix3, ref_dir, chr, n_threads, r2);
    }

    // mcmc 
    if (run_mcmc) 
	{

		if (ss_path1.empty()) 
		{
	    	cout << "Did not specify the path to summary statistics for population 1." << endl;
	    	return 0;
		}
		if (ss_path2.empty()) 
		{
	    	cout << "Did not specify the path to summary statistics for population 2." << endl;
	    	return 0;
		}
		if (ss_path3.empty()) 
		{
	    	cout << "Did not specify the path to summary statistics for population 3." << endl;
	    	return 0;
		}

		if (out_path1.empty()) 
		{
	    	cout << "Did not specify the path of the output file for population 1." << endl;
	    	return 0;
		}

		if (out_path2.empty()) 
		{
	    	cout << "Did not specify the path of the output file for population 2." << endl;
	    	return 0;
		}

		if (out_path3.empty()) 
		{
	    	cout << "Did not specify the path of the output file for population 3." << endl;
	    	return 0;
		}

		if (!N1) 
		{
	    	cout << "Did not specify GWAS sample size for population 1." << endl;
	    	return 0;
		}

		if (!N2) 
		{
	    	cout << "Did not specify GWAS sample size for population 2." << endl;
	    	return 0;
		}

		if (!N3) 
		{
	    	cout << "Did not specify GWAS sample size for population 3." << endl;
	    	return 0;
		}

		string ref_ldmat1 = ref_dir + "/chr" + \
	       	std::to_string(chr) + "pop1.dat";
		string ref_snpinfo = ref_dir + "/chr" + \
	       std::to_string(chr) + ".snpInfo";
		string ref_ldmat2 = ref_dir + "/chr" + \
	       	std::to_string(chr) + "pop2.dat";
		string ref_ldmat3 = ref_dir + "/chr" + \
	       	std::to_string(chr) + "pop3.dat";
		
		cout << rho3 << endl;
		mcmc(ref_snpinfo, ss_path1, ss_path2, ss_path3, valid_bim, ref_ldmat1, ref_ldmat2, ref_ldmat3, N1, N2, N3, \
		out_path1, out_path2, out_path3, a, rho1, rho2, rho3, a0k, b0k, iter, burn, thin, n_threads, opt_llk, c1, c2, c3);
    }
    return 0;
}