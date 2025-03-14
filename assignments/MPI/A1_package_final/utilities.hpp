// ==============================================================================
// 
//  Write your code only in "your_mpi_implementation.hpp", NOT in this file. 
//  Anything written to this file will not be submitted or counted.
//  This is the header file for some utility and algorithm functions. You can call
//  these functions in your implementation.
// 
// ==============================================================================

#pragma once
#define _in_
#define _out_

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

const int MIN_N = 1, MAX_N = 100000;
const int MIN_L = 100, MAX_L = 10000;
const int MIN_K = 4, MAX_K = 16;
const int MAX_PROCESS = 32;

typedef unsigned int kmer_t;    // define the encoded k-mer type to be kmer_t

const unsigned char basemap[256] = {
    0, 1, 2, 3, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 0, 255, 1, 255, 255, 255, 2, // 64..71
    255, 255, 255, 255, 255, 255, 0, 255, // 72..79
    255, 255, 255, 255, 3, 0, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 0, 255, 1, 255, 255, 255, 2, // 96..103
    255, 255, 255, 255, 255, 255, 0, 255, // 104..111
    255, 255, 255, 255, 3, 0, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

// ============================================================================================
//   Below are the functions that you may find useful in your implementation.
//   You can use or not use them as you wish.
// ============================================================================================

/// @brief Convert a string-type k-mer to an integer-type (kmer_t/unsigned int) k-mer. \
    You can use this function in your implementation or write your own version.
/// @param k length of k-mer
/// @param kmer the pointer to the first character of the k-mer
/// @return an unsigned integer representing of k-mer
kmer_t kmer_encoding(const int k, const char *kmer_beg_ptr) {
    kmer_t res = 0;
    for (int i=0; i<k; i++) {
        res <<= 2;  // same as res *= 4; here
        res |= basemap[(unsigned char)kmer_beg_ptr[i]]; // same as res += basemap[(unsigned char)kmer[i]]; here
    }
    return res;
}
kmer_t kmer_encoding(const int k, std::string kmer) {
    return kmer_encoding(k, kmer.c_str());
}

/// @brief The function to determine whether a k-mer is a universal minimizer. \
    You should use and NOT CHANGE this function in your implementation.
/// @param kmer the integer representation of the k-mer
/// @return true if the k-mer is a universal minimizer, false otherwise.
bool is_universal_minimizer(kmer_t kmer) {
    return kmer % 7 == 0;
}

/// @brief Given a read, generate the universal minimizer list for the read and return.\
    You can use this function in your implementation or write your own version.
/// @param read A string containing only 'A', 'C', 'G', 'T'.
/// @return A vector of kmer_t, the universal minimizer list of the read.
std::vector<kmer_t> generate_universal_minimizer_list(const int k, int n, const char *read) {
    std::vector<kmer_t> um_list;
    for (int i=0; i<n-k+1; i++) {
        kmer_t kmer_int = kmer_encoding(k, &read[i]);
        if (is_universal_minimizer(kmer_int)) {
            um_list.push_back(kmer_int);
        }
    }
    return um_list;
}
std::vector<kmer_t> generate_universal_minimizer_list(const int k, std::string &read) {
    return generate_universal_minimizer_list(k, read.length(), read.c_str());
}

/// @brief Convert vector of universal minimizer lists to CSR format. \
    You can use this function in your implementation or write your own version.
/// @param um_lists A vector of vector of kmer_t, the universal minimizer lists.
/// @param num_of_lists (out) The number of lists in um_lists.
/// @param um_lists_CSR (out) The universal minimizer lists in CSR format.
/// @param um_lists_CSR_offs (out) The offsets of the universal minimizer lists in CSR format.
void Vector2CSR(std::vector<std::vector<kmer_t>> &um_lists, int &num_of_lists, kmer_t* &um_lists_CSR, int* &um_lists_CSR_offs);

/// @brief Convert CSR format to vector of universal minimizer lists. \
    You can use this function in your implementation or write your own version.
/// @param num_of_lists The number of lists in um_lists.
/// @param um_lists_CSR The universal minimizer lists in CSR format.
/// @param um_lists_CSR_offs The offsets of the universal minimizer lists in CSR format.
/// @param um_lists (out) A vector of vector of kmer_t, the universal minimizer lists.
void CSR2Vector(int num_of_lists, kmer_t* um_lists_CSR, int* um_lists_CSR_offs, std::vector<std::vector<kmer_t>> &um_lists) {
    um_lists.clear();
    for (int i=0; i<num_of_lists; i++) {
        std::vector<kmer_t> um_list;
        for (int j=um_lists_CSR_offs[i]; j<um_lists_CSR_offs[i+1]; j++) {
            um_list.push_back(um_lists_CSR[j]);
        }
        um_lists.push_back(um_list);
    }
}


// --------------------------------------------------------------------------------------------
//   Below are the utility functions that you may not need to use in your implementation.
// --------------------------------------------------------------------------------------------
void LoadInputFile(const char* filename, _out_ int &k, _out_ int &n, _out_ std::vector<std::string> &reads) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "ERROR: unable to open file: " << filename << "." << std::endl;
        perror("fopen");
        exit(1);
    }
    fin >> k;
    if (k < MIN_K || k > MAX_K) {
        std::cerr << "ERROR: invalid k ("<<k<<")." << std::endl;
    }
    std::string read;
    while (fin >> read) {
        if (read.length() < k) {
            std::cerr << "ERROR: invalid read length ("<<read.length()<<")." << std::endl;
            exit(1);
        }
        reads.push_back(read);
    }
    n = reads.size();
    fin.close();
    return;
}

/**
* @brief CSR Format
* @par Sample:
* @code
*   //sample code here
* @endcode
* @note Limitations:
* The length of one single data item cannot exceed INT_MAX;
* The total number of items cannot exceed INT_MAX;
* The total length of the whole data cannot exceed ULL_MAX;
*/
typedef /**/int T_CSR_capacity;
typedef int T_CSR_count;
template<typename T_data, class T_attr=bool> class CSR
{
protected:
    T_data *_data;
    bool _with_attr;
    T_attr *_attr;
    T_CSR_count _item_count = 0, _item_capacity = 4; // for saving attributes
    T_CSR_capacity *_offs;
    
    T_CSR_capacity _size = 0;     // unit: counts not bytes, for saving raw data
    T_CSR_capacity _capacity = 0; // unit: counts not bytes
    
    void _Realloc(T_CSR_capacity new_capacity) {
        // new_capacity: capacity in counts
        _data = (T_data*)realloc(_data, new_capacity * sizeof(T_data));
    }
    
public:
    float size_incre_ratio = 1.5;
    
    CSR(bool with_attr = false, T_CSR_capacity initial_capacity = 8) {
        _with_attr = with_attr;
        if (with_attr) _attr = new T_attr[_item_capacity]();//
        _offs = new T_CSR_capacity[_item_capacity+1]();//
        _offs[0] = 0;
        _capacity = initial_capacity;
        _data = new T_data[_capacity]();//
    }
    CSR(const CSR<T_data, T_attr>& f) {
        _item_capacity = _item_count = f._item_count;
        if (f.has_attr()) {
            _attr = new T_attr[_item_capacity]();//
            _with_attr = true;
            memcpy(_attr, f._attr, sizeof(T_attr) * _item_count);
        }
        _offs = new T_CSR_capacity[_item_capacity+1]();//?
        memcpy(_offs, f._offs, sizeof(T_CSR_capacity)*(_item_count+1));
        _size = _capacity = f._size;
        _data = new T_data[_size]();//
        memcpy(_data, f._data, sizeof(T_data)*_size);
    }
    ~CSR(void) {
        delete [] _offs;//
        delete [] _data;//
        if (_with_attr) delete [] _attr;//
    }
    
    // append with no attr
    void append(const T_data *new_data, T_CSR_capacity new_data_size) {
        // the unit of new_data_size is count not byte
        if (new_data_size + _size >= _capacity) {
            T_CSR_capacity new_capacity = (T_CSR_capacity)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count == _item_capacity-1) {
            _offs = (T_CSR_capacity *)realloc(_offs, sizeof(T_CSR_capacity) * (2*_item_capacity+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (2*_item_capacity));
            _item_capacity = 2*_item_capacity;
        }
        memcpy(_data + _offs[_item_count], new_data, new_data_size * sizeof(T_data));
        _size += new_data_size;
        _offs[_item_count+1] = _size;
        _item_count += 1;
    }

    // append with attr
    void append(const T_data *new_data, T_CSR_capacity new_data_size, T_attr attr) {
        // the unit of new_data_size is count not byte
        if (new_data_size + _size >= _capacity) {
            T_CSR_capacity new_capacity = (T_CSR_capacity)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count == _item_capacity-1) {
            _offs = (T_CSR_capacity *)realloc(_offs, sizeof(T_CSR_capacity) * (2*_item_capacity+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (2*_item_capacity));
            _item_capacity = 2*_item_capacity;
        }
        memcpy(_data + _offs[_item_count], new_data, new_data_size * sizeof(T_data));
        _size += new_data_size;
        if (_with_attr) _attr[_item_count] = attr;
        _offs[_item_count+1] = _size;
        _item_count += 1;
    }
    void append(CSR<T_data, T_attr> new_data) {
        T_CSR_capacity new_data_size = new_data.size();
        if (new_data_size + _size >= _capacity) {
            T_CSR_capacity new_capacity = (T_CSR_capacity)(double(_capacity)*size_incre_ratio);
            _capacity = new_data_size + _size > new_capacity ? new_data_size + _size + 1 : new_capacity;
            _Realloc(_capacity);
        }
        if (_item_count + new_data.items() >= _item_capacity) {
            // should multiply 2 below to avoid float size.
            int new_size = _item_count+new_data.items() > 2*_item_capacity ? _item_count+new_data.items() : 2*_item_capacity;
            _offs = (T_CSR_capacity *)realloc(_offs, sizeof(T_CSR_capacity) * (new_size+1));
            if (_with_attr) _attr = (T_attr *)realloc(_attr, sizeof(T_attr) * (new_size));
            _item_capacity = new_size;
        }
        memcpy(_data + _offs[_item_count], new_data.get_raw_data(), new_data_size * sizeof(T_data));
        memcpy(&_offs[_item_count+1], new_data.get_raw_offs()+1, new_data.items() * sizeof(T_CSR_capacity));
        for(T_CSR_count i=_item_count+1; i<_item_count+new_data.items()+1; i++)
            _offs[i] += _offs[_item_count]; // add the offset value for new elements
        if (_with_attr && new_data.has_attr()) 
            memcpy(&_attr[_item_count], new_data.get_raw_attr(), new_data.items() * sizeof(T_attr));
        _size += new_data_size;
        _item_count += new_data.items();
    }
    T_CSR_capacity capacity() {
        return _capacity;
    }
    T_CSR_capacity size() {
        return _size;
    }
    T_CSR_count items() {
        return _item_count;
    }
    short dtype() {
        return sizeof(T_data);
    }
    void debug_info() {
        std::cout << "ITEMS\t" << this -> items() << "/" << _item_capacity << std::endl;
        for (int i=0; i<_item_count; i++) {
            std::cout << "offs="<< _offs[i+1];
            if (has_attr()) std::cout << " attr=" << _attr[i];
            std::cout << std::endl;
        }
        std::cout << "SIZE\t" << this -> size() << "/" << this -> capacity() << std::endl;
        std::cout << "DTYPE\t" <<this -> dtype() << std::endl;
        for (T_CSR_capacity i=0; i<_size; i++) std::cout<<_data[i]<<" ";
        std::cout << std::endl;
    }
    T_data* get_raw_data() {
        return _data;
    }
    T_CSR_capacity* get_raw_offs() {
        return _offs;
    }
    T_data* fetch_raw_data() {
        T_data* res = new T_data[_size];
        memcpy(res, _data, _size*sizeof(T_data));
        return res;
    }
    T_CSR_capacity* fetch_raw_offs() {
        T_CSR_capacity* res = new T_CSR_capacity[_item_count+1];
        memcpy(res, _offs, (_item_count+1)*sizeof(T_CSR_capacity));
        return res;
    }
    bool has_attr() {
        return _with_attr;
    }
    T_attr* get_raw_attr() {
        return _attr;
    }
    int get_item(int idx, _out_ T_data *data_buffer) {
        if (idx > this->size()) return -1;
        memcpy(data_buffer, &(this->_data[this->_offs[idx]]), sizeof(T_data) * (this->_offs[idx+1] - this->_offs[idx]));
        return 0;
    }
    T_attr get_attr(int idx) {
        if (idx > this->size()) {
            std::cerr << "[ERROR] CSR.get_attr(idx) got wrong idx " << idx << std::endl;
            exit(1);
        }
        return _attr[idx];
    }
};

void Vector2CSR(std::vector<std::vector<kmer_t>> &um_lists, int &num_of_lists, kmer_t* &um_lists_CSR, int* &um_lists_CSR_offs) {
    CSR<kmer_t> t;
    for(std::vector<kmer_t> &um_list: um_lists) {
        t.append(um_list.data(), um_list.size());
    }
    num_of_lists = um_lists.size();
    um_lists_CSR = t.fetch_raw_data();
    um_lists_CSR_offs = t.fetch_raw_offs();
}

void Vector2CSR(std::vector<std::string> &reads, int &num_of_reads, char* &reads_CSR, int* &reads_CSR_offs) {
    CSR<char> t;
    for(std::string &read: reads) {
        t.append(read.c_str(), read.length());
    }
    num_of_reads = reads.size();
    reads_CSR = t.fetch_raw_data();
    reads_CSR_offs = t.fetch_raw_offs();
}

void SaveResults(std::string &output_path, std::vector<std::vector<kmer_t>> &um_lists) {
    FILE *fp = fopen(output_path.c_str(), "w");
    size_t tot_um_cnt = 0;
    if (fp == NULL) {
        std::cerr << "ERROR: unable to write file: " << output_path << "." << std::endl;
        perror("fopen");
        exit(1);
    }
    for (std::vector<kmer_t> um_list: um_lists) {
        if (um_list.size() == 0) fprintf(fp, "-1 \n");
        else {
            for (kmer_t um: um_list) fprintf(fp, "%u ", um);
            fprintf(fp, "\n");
            if (um_list.size() > 1 || um_list[0] != -1) tot_um_cnt += um_list.size();
        }
    }
    fclose(fp);
    std::cout << um_lists.size() << " lists of universal minimizers have been saved to " << output_path << "." << std::endl;
    std::cout << "Total number of universal minimizers in this dataset: " << tot_um_cnt << std::endl;
}

