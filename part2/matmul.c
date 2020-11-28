#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <immintrin.h>

//#define DEBUG 1

int block8_ceil(const int x) {
    return (x+7) & (~(unsigned int)0x7);
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
    int get_int, _m;
    char ch;

    // Get matrix size.
    scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    _m = block8_ceil(*m_ptr);

    // Allocate memory space for matrices.
    posix_memalign ((void **)a_mat_ptr, 32, *n_ptr * _m * sizeof(int));
    memset(*a_mat_ptr, 0, *n_ptr * _m * sizeof(int));
    // Create a transposed b matrix.
    posix_memalign ((void **)b_mat_ptr, 32, *l_ptr * _m * sizeof(int));
    memset(*b_mat_ptr, 0, *l_ptr * _m * sizeof(int));

    // Get data from stdin.
    getchar_unlocked();
    for (int arow = 0; arow < *n_ptr; arow++) { // Matrix a
        for (int acol = 0; acol < *m_ptr; acol++) {
            get_int = getchar_unlocked() & 0xF;
            while ((ch = getchar_unlocked()) != ' ') {
                get_int = get_int*10 + (ch&0xF);
            }
            (*a_mat_ptr)[arow*_m + acol] = get_int;
        }
        getchar_unlocked();
    }

    for (int brow = 0; brow < *m_ptr; brow++) { // Transpose of matrix b
        for (int bcol = 0; bcol < *l_ptr; bcol++) {
            get_int = getchar_unlocked() & 0xF;
            while ((ch = getchar_unlocked()) != ' ') {
                get_int = get_int*10 + (ch&0xF);
            }
            (*b_mat_ptr)[bcol*_m + brow] = get_int; // Remind the index. It's
                                                    // transposed.
        }
        getchar_unlocked();
    }
}

int hsum_epi32_avx(__m128i x)
{
    __m128i hi64  = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

// only needs AVX2
int hsum_8x32(__m256i v)
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
    int _m = block8_ceil(m);
    int *__restrict c = (int *)malloc(n * l * sizeof(int));

    // Matrix multiplication.
    for (int crow = 0; crow < n; crow++) {
        for (int ccol = 0; ccol < l; ccol++) {
            __m256i a, b, ab, tmp;

            tmp = _mm256_setzero_si256();
            for (int k = 0; k < _m; k += 8) {
                a = _mm256_load_si256((__m256i *)(a_mat + crow*_m + k));
                b = _mm256_load_si256((__m256i *)(b_mat + ccol*_m + k));
                ab = _mm256_mullo_epi16(a, b);
                tmp = _mm256_add_epi32(tmp, ab);
            }
            c[crow*l + ccol] = hsum_8x32(tmp);
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
    for (int crow = 0; crow < n; crow++) {
        printf("%d", c[crow*l]);
        for (int ccol = 1; ccol < l; ccol++) {
            printf(" %d", c[crow*l + ccol]);
        }
        printf("\n");
    }
#endif

    free(c);
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
}

