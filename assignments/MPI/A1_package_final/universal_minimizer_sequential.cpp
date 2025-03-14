// ==============================================================================
// 
//  This is the sequential version for your reference and result comparison.
//  Write your code only in "your_mpi_implementation.hpp", NOT in this file. 
//  Anything written to this file will not be submitted or counted.
// 
// ==============================================================================

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include "utilities.hpp"
using namespace std;

int k, n;   // k: k-value of k-mer, n: number of reads
string input_filename;
string output_filename = "./s_output.txt";

int main(int argc, char **argvs) {
    if (argc < 2) {
        cerr << "Missing input data file!" << endl;
        exit(1);
    }
    input_filename = argvs[1];
    
    // Input data (the reads in vector of strings)
    vector<string> reads;
    
    // Input data (the reads in CSR format, easier to use in MPI programming)
    char* reads_CSR;
    int* reads_CSR_offs;

    // Output data, lists of universal minimizers, the order should be consistant with the order of the input reads
    vector<vector<kmer_t>> um_lists;

    LoadInputFile(input_filename.c_str(), k, n, reads); // will load the k-value and the reads from the file
    Vector2CSR(reads, n, reads_CSR, reads_CSR_offs);    // unnecessary in this sequential version, useful in MPI programming
    cout << "k=" << k << " (length of k-mer), n=" << n << "(number of reads)" << endl << endl;
    
    // time measurement starts from here
    auto start_time = chrono::high_resolution_clock::now();
    
    // implementation: generate universal minimizer list for each read
    for (int i=0; i<n; i++) {
        um_lists.push_back(generate_universal_minimizer_list(k, reads[i]));
    }

    // time measurement ends here
    auto end_time = chrono::high_resolution_clock::now();
    auto duration_sec = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count()/1000.0;
    cout << "Sequential program finished in " << duration_sec << " sec." << endl << endl;
        
    // output to text file and correctness checking
    SaveResults(output_filename, um_lists);
    
    delete reads_CSR;
    delete reads_CSR_offs;
    return 0;
}
