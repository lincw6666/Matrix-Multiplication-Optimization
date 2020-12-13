#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <immintrin.h>

//#define DEBUG 1
int rank;

inline int block32_ceil(const int x) {
    return (x+0x1F) & (~(unsigned int)0x1F);
}

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    int _m;
    int8_t get_int, *a_mat, *b_mat;
    char ch;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank > 0) return ;
    // Get matrix size.
    scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    _m = block32_ceil(*m_ptr);

    // Allocate memory space for matrices.
    posix_memalign ((void **)a_mat_ptr, 64, *n_ptr * _m * sizeof(int8_t));
    memset(*a_mat_ptr, 0, *n_ptr * _m * sizeof(int8_t));
    a_mat = (int8_t *)*a_mat_ptr;
    // Create a transposed b matrix.
    posix_memalign ((void **)b_mat_ptr, 64, *l_ptr * _m * sizeof(int8_t));
    memset(*b_mat_ptr, 0, *l_ptr * _m * sizeof(int8_t));
    b_mat = (int8_t *)*b_mat_ptr;

    // Get data from stdin.
    flockfile(stdin);
    getchar_unlocked();
    for (int arow = 0; arow < *n_ptr; arow++) { // Matrix a
        for (int acol = 0; acol < *m_ptr; acol++) {
            get_int = getchar_unlocked() & 0xF;
            while ((ch = getchar_unlocked()) != ' ') {
                get_int = get_int*10 + (ch&0xF);
            }
            a_mat[arow*_m + acol] = get_int;
        }
        getchar_unlocked();
    }

    for (int brow = 0; brow < *m_ptr; brow++) { // Transpose of matrix b
        for (int bcol = 0; bcol < *l_ptr; bcol++) {
            get_int = getchar_unlocked() & 0xF;
            while ((ch = getchar_unlocked()) != ' ') {
                get_int = get_int*10 + (ch&0xF);
            }
            b_mat[bcol*_m + brow] = get_int; // Remind the index. It's
                                             // transposed.
        }
        getchar_unlocked();
    }
    funlockfile(stdin);
}

inline int hsum_epi32_avx(__m128i x)
{
    __m128i hi64  = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

// only needs AVX2
inline int hsum_8x32(__m256i v)
{
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v),
                                   _mm256_extracti128_si256(v, 1));
    return hsum_epi32_avx(sum128);
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *__restrict a_mat, const int *__restrict b_mat)
{
    if (rank > 0) return ;
    int _m = block32_ceil(m) >> 2, step = 4096 / _m, crow;
    int *__restrict c = (int *)malloc(n * l * sizeof(int));
    __m256i mask = _mm256_set1_epi16(0xFF);

    // Matrix multiplication.
    for (crow = 0; crow+step < n; crow += step) {
        int ccol;

        for (ccol = 0; ccol+step < l; ccol += step) {
            for (int sa = 0; sa < step; sa++) {
                for (int sb = 0; sb < step; sb++) {
                    __m256i a1, b1, a2, b2, ab1, ab2, ab, tmp;

                    tmp = _mm256_setzero_si256();
                    for (int k = 0; k < _m; k += 8) {
                        b1 = _mm256_load_si256((__m256i *)(b_mat + (ccol+sb)*_m + k));
                        b2 = _mm256_and_si256(b1, mask);
                        b1 = _mm256_srai_epi16(b1, 8);
                        a1 = _mm256_load_si256((__m256i *)(a_mat + (crow+sa)*_m + k));
                        a2 = _mm256_and_si256(a1, mask);
                        a1 = _mm256_srai_epi16(a1, 8);
                        ab1 = _mm256_madd_epi16(a1, b1);
                        ab2 = _mm256_madd_epi16(a2, b2);
                        ab = _mm256_add_epi16(ab1, ab2);
                        tmp = _mm256_add_epi32(tmp, ab);
                    }
                    c[(crow+sa)*l + (ccol+sb)] = hsum_8x32(tmp);
                }
            }
        }
        for (int sa = 0; sa < step; sa++) {
            for (int sb = ccol; sb < l; sb++) {
                __m256i a1, b1, a2, b2, ab1, ab2, ab, tmp;

                tmp = _mm256_setzero_si256();
                for (int k = 0; k < _m; k += 8) {
                    b1 = _mm256_load_si256((__m256i *)(b_mat + sb*_m + k));
                    b2 = _mm256_and_si256(b1, mask);
                    b1 = _mm256_srai_epi16(b1, 8);
                    a1 = _mm256_load_si256((__m256i *)(a_mat + (crow+sa)*_m + k));
                    a2 = _mm256_and_si256(a1, mask);
                    a1 = _mm256_srai_epi16(a1, 8);
                    ab1 = _mm256_madd_epi16(a1, b1);
                    ab2 = _mm256_madd_epi16(a2, b2);
                    ab = _mm256_add_epi16(ab1, ab2);
                    tmp = _mm256_add_epi32(tmp, ab);
                }
                c[(crow+sa)*l + sb] = hsum_8x32(tmp);
            }
        }
    }
    // Do the remain part.
    int ccol;

    for (ccol = 0; ccol+step < l; ccol += step) {
        for (int sa = crow; sa < n; sa++) {
            for (int sb = 0; sb < step; sb++) {
                __m256i a1, b1, a2, b2, ab1, ab2, ab, tmp;

                tmp = _mm256_setzero_si256();
                for (int k = 0; k < _m; k += 8) {
                    b1 = _mm256_load_si256((__m256i *)(b_mat + (ccol+sb)*_m + k));
                    b2 = _mm256_and_si256(b1, mask);
                    b1 = _mm256_srai_epi16(b1, 8);
                    a1 = _mm256_load_si256((__m256i *)(a_mat + sa*_m + k));
                    a2 = _mm256_and_si256(a1, mask);
                    a1 = _mm256_srai_epi16(a1, 8);
                    ab1 = _mm256_madd_epi16(a1, b1);
                    ab2 = _mm256_madd_epi16(a2, b2);
                    ab = _mm256_add_epi16(ab1, ab2);
                    tmp = _mm256_add_epi32(tmp, ab);
                }
                c[sa*l + (ccol+sb)] = hsum_8x32(tmp);
            }
        }
    }
    for (; crow < n; crow++) {
        for (int sb = ccol; sb < l; sb++) {
            __m256i a1, b1, a2, b2, ab1, ab2, ab, tmp;

            tmp = _mm256_setzero_si256();
            for (int k = 0; k < _m; k += 8) {
                b1 = _mm256_load_si256((__m256i *)(b_mat + sb*_m + k));
                b2 = _mm256_and_si256(b1, mask);
                b1 = _mm256_srai_epi16(b1, 8);
                a1 = _mm256_load_si256((__m256i *)(a_mat + crow*_m + k));
                a2 = _mm256_and_si256(a1, mask);
                a1 = _mm256_srai_epi16(a1, 8);
                ab1 = _mm256_madd_epi16(a1, b1);
                ab2 = _mm256_madd_epi16(a2, b2);
                ab = _mm256_add_epi16(ab1, ab2);
                tmp = _mm256_add_epi32(tmp, ab);
            }
            c[crow*l + sb] = hsum_8x32(tmp);
        }
    }

#ifdef DEBUG
    // Output the result to a file.
    #define CHUNK_SIZE 4096
    FILE *fp;
    char file_buf[CHUNK_SIZE + 64];
    int buf_cnt = 0;

    // Open a file to write the result.
    fp = fopen("ans", "w");
    if (fp == NULL) {
        printf("Can't open file: ans\n");
        return ;
    }

    for (int crow = 0; crow < n; crow++) {
        buf_cnt += sprintf(&file_buf[buf_cnt], "%d", c[crow*l]);
        for (int ccol = 1; ccol < l; ccol++) {
            buf_cnt += sprintf(&file_buf[buf_cnt], " %d", c[crow*l + ccol]);
            if (buf_cnt >= CHUNK_SIZE) {
                fwrite(file_buf, CHUNK_SIZE, 1, fp);
                buf_cnt -= CHUNK_SIZE;
                memcpy(file_buf, &file_buf[CHUNK_SIZE], buf_cnt);
            }
        }
        buf_cnt += sprintf(&file_buf[buf_cnt], "\n");
    }

    // Write remainder
    if (buf_cnt > 0) {
        fwrite(file_buf, buf_cnt, 1, fp);
    }

    // Close the file
    fclose(fp);
#else
    // Print the result.
    char buf[100000];
    int buf_idx = 99999;

    buf[99999] = '\n';
    flockfile(stdout);
    for (int crow = 0; crow < n; crow++) {
        int tmp;
        for (int ccol = l-1; ccol > 0; ccol--) {
            tmp = c[crow*l + ccol];
            if (tmp != 0) {
                for (; tmp > 0; tmp /= 10) {
                    buf[--buf_idx] = (tmp%10) | 0x30;
                }
            }
            else buf[--buf_idx] = '0';
            buf[--buf_idx] = ' ';
        }
        tmp = c[crow*l];
        if (tmp != 0) {
            for (; tmp > 0; tmp /= 10) {
                buf[--buf_idx] = (tmp%10) | 0x30;
            }
        }
        else buf[--buf_idx] = '0';
        fwrite_unlocked(&buf[buf_idx], 100000-buf_idx, 1, stdout);
        buf_idx = 99999;
    }
    funlockfile(stdout);
#endif

    free(c);
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    if (rank > 0) return ;
    free(a_mat);
    free(b_mat);
}

