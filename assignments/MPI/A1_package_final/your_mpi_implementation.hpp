//     [NAME]:  ZHANG Xiao [STUDENT ID]: 21093165
//     [EMAIL]: xzhangin@connect.ust.hk

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
#include <queue>
#include <memory>
#include <mpi.h>

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

void gen_um_lists_MPI(MPI_Comm comm, int my_rank, int num_process, int k, int n, vector<string> &reads, char *reads_CSR, int *reads_CSR_offs, vector<vector<kmer_t>> &um_lists)
{
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    int reads_every_proc = n / num_process;//# of each proc
    int remain = n % num_process;//
    int my_read_count = (my_rank < remain) ? (reads_every_proc + 1) : reads_every_proc;
    int my_start = (my_rank < remain) ? (my_rank * (reads_every_proc + 1)) : (remain * (reads_every_proc + 1) + (my_rank - remain) * reads_every_proc);
    int total_length = 0;
    if (my_rank == 0) total_length = reads_CSR_offs[n];

    MPI_Bcast(&total_length, 1, MPI_INT, 0, comm);

    if (my_rank != 0)
    {
        reads_CSR_offs = new int[n + 1];
        reads_CSR = new char[total_length];
    }

    MPI_Bcast(reads_CSR_offs, n + 1, MPI_INT, 0, comm);
    MPI_Bcast(reads_CSR, total_length, MPI_CHAR, 0, comm);// bad attempt but its hard to chunk...

    vector<vector<kmer_t>> local_um_lists(my_read_count);
    for (int i = 0; i < my_read_count; i++)//get the local res
    {
        int my_i = my_start + i;
        int read_length = reads_CSR_offs[my_i + 1] - reads_CSR_offs[my_i];
        string read(reads_CSR + reads_CSR_offs[my_i], read_length);
        local_um_lists[i] = generate_universal_minimizer_list(k, read);
    }

    if (my_rank == 0)
    {
        um_lists.resize(n);//get res from rank0
        for (int i = 0; i < my_read_count; i++) um_lists[my_start + i] = local_um_lists[i];
        for (int proc = 1; proc < num_process; proc++) // get res form others
        {
            int _my_start;
            int _read_count;
            if (proc < remain)
            {
                _my_start = proc * (reads_every_proc + 1);
                _read_count = reads_every_proc + 1;
            }
            else
            {
                _my_start = remain * (reads_every_proc + 1) + (proc - remain) * reads_every_proc;
                _read_count = reads_every_proc;
            }
            for (int i = 0; i < _read_count; i++)
            {
                int rebuilt_i = _my_start + i;
                int _size;
                MPI_Recv(&_size, 1, MPI_INT, proc, 0, comm, MPI_STATUS_IGNORE);
                um_lists[rebuilt_i].resize(_size);
                if (_size > 0)
                {
                    MPI_Recv(um_lists[rebuilt_i].data(), _size * sizeof(kmer_t), MPI_BYTE,
                             proc, 0, comm, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < my_read_count; i++)//send res to rank 0
        {
            int _size = local_um_lists[i].size();
            MPI_Send(&_size, 1, MPI_INT, 0, 0, comm);
            if (_size > 0)
            {
                MPI_Send(local_um_lists[i].data(), _size * sizeof(kmer_t), MPI_BYTE, 0, 0, comm);
            }
        }
    }
    if (my_rank != 0)
    {
        delete[] reads_CSR;
        delete[] reads_CSR_offs;
    }
}