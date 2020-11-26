#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define DEBUG 1

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

    // Get matrix size.
    scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    _m = block8_ceil(*m_ptr);

    // Allocate memory space for matrices.
    *a_mat_ptr = (int *)calloc(*n_ptr * _m, sizeof(int));
    // Create a transposed b matrix.
    *b_mat_ptr = (int *)calloc(*l_ptr * _m, sizeof(int));

    // Get data from stdin.
    for (int arow = 0; arow < *n_ptr; arow++) { // Matrix a
        for (int acol = 0; acol < *m_ptr; acol++) {
            scanf("%d", &get_int);
            (*a_mat_ptr)[arow*_m + acol] = get_int;
        }
    }

    for (int brow = 0; brow < *m_ptr; brow++) { // Transpose of matrix b
        for (int bcol = 0; bcol < *l_ptr; bcol++) {
            scanf("%d", &get_int);
            (*b_mat_ptr)[bcol*_m + brow] = get_int; // Remind the index. It's
                                                    // transposed.
        }
    }
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
            int tmp = 0;
            for (int k = 0; k < _m; k++) {
                tmp += a_mat[crow*_m + k] * b_mat[ccol*_m + k];
            }
            c[crow*l + ccol] = tmp;
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

