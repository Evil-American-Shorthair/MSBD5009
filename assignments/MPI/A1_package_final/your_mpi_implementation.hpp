//     [NAME]:            [STUDENT ID]: 
//     [EMAIL]: 

// *********************************************************************
//     NOTICE: Write your code only in this file and only submit this 
//     file to Canvas. You can add more functions, structures, classes,
//     and include other C++ standard libraries in this file if needed.
//     Do NOT change the signature of the function gen_um_lists_MPI, 
//     which will be called by the main function.
// *********************************************************************
#pragma once

#include <string>
#include <vector>
#include "utilities.hpp"
#include "mpi.h"

using namespace std;

/// @brief Generate universal minimizer lists for each read in parallel using MPI.
/// @param comm The MPI communicator.
/// @param my_rank Global rank of this process.
/// @param num_process Total number of processes.
/// @param k length of k-mer (only available in Process 0 when calling).
/// @param n number of reads (only available in Process 0 when calling).
/// @param reads The input reads in vector of strings (only available in Process 0 when calling).
/// @param reads_CSR The input reads in CSR format (only available in Process 0 when calling).
/// @param reads_CSR_offs The offsets of the input reads in CSR format (only available in Process 0 when calling).
/// @param um_lists The output lists of universal minimizers (result should be gathered to this vector in Process 0 when function ends).
void gen_um_lists_MPI (MPI_Comm comm, int my_rank, int num_process, int k, int n, vector<string>& reads, char* reads_CSR, int* reads_CSR_offs, vector<vector<kmer_t>>& um_lists) {
    
    // Finish this function and make sure it works properly by the call in main()
    
}