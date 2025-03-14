// ==============================================================================
// 
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
#include "your_mpi_implementation.hpp"  // only implement this file in your assignment
#include "mpi.h"
using namespace std;

int k = -1, n = -1;   // k: k-value of k-mer, n: number of reads
string input_filename;
string output_filename = "./output.txt";

void gen_um_lists_MPI(MPI_Comm comm, int my_rank, int num_process, int k, int n, vector<string>& reads, char* reads_CSR, int* reads_CSR_offs, vector<vector<kmer_t>>& um_lists);

int main(int argc, char **argvs) {
    MPI_Init(&argc, &argvs);
    MPI_Comm comm = MPI_COMM_WORLD;
    int num_process; // number of processors
    int my_rank;     // my global rank
    MPI_Comm_size(comm, &num_process);
    MPI_Comm_rank(comm, &my_rank);
    
    if (my_rank == 0) {
        if (argc < 2) {
            cerr << "Missing input data file!" << endl;
            exit(1);
        }
        input_filename = argvs[1];
    }
    
    // Input data (the reads in vector of strings)
    vector<string> reads;
    
    // Input data (the reads in CSR format, easier to use in MPI programming)
    char* reads_CSR;
    int* reads_CSR_offs;

    // Output data, lists of universal minimizers, the order should be consistant with the order of the input reads
    vector<vector<kmer_t>> um_lists;

    if (my_rank == 0) {
        LoadInputFile(input_filename.c_str(), k, n, reads); // will load the k-value and the reads from the file
        Vector2CSR(reads, n, reads_CSR, reads_CSR_offs);    // unnecessary in this sequential version, useful in MPI programming
        cout << "k=" << k << " (length of k-mer), n=" << n << "(number of reads)" << endl << endl;
    }
    
    // time measurement starts from here
    auto start_time = chrono::high_resolution_clock::now();
    
    // Your task: implement this function in >>> your_mpi_implementation.hpp <<< !!!
    gen_um_lists_MPI(comm, my_rank, num_process, k, n, reads, reads_CSR, reads_CSR_offs, um_lists);
    MPI_Barrier(comm);
    
    // Time measurement ends here
    auto end_time = chrono::high_resolution_clock::now();
    auto duration_sec = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count()/1000.0;
    if (my_rank == 0) {
        cout << "Your algorithm finished in " << duration_sec << " sec." << endl << endl;
    }
        
    // Output result to text file
    if (my_rank == 0) {
        SaveResults(output_filename, um_lists);
        delete reads_CSR;
        delete reads_CSR_offs;
    }
    MPI_Finalize();
    return 0;
}
