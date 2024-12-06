#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <omp.h>

#define PI M_PI
#define TOL 1e-6
#define MAX_ITER 1000000
#define BLOCK_SIZE 4

void selectGPU(int rank) {
    int num_gpus;
    hipGetDeviceCount(&num_gpus);
    int gpu_id = rank % num_gpus;
    hipSetDevice(gpu_id);
}

__device__ __host__ double exact_solution(double x, double y, double z) {
    return sin(2.0 * PI * x) * cos(2.0 * PI * y) * sin(2.0 * PI * z);
}

__global__ void jacobi_kernel(double* u_new, const double* u, const double* f,
                            int start_k, int end_k, int N, double d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + start_k;
    
    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > start_k && k < end_k-1) {
        int idx = i + j * N + (k-start_k) * N * N;
        int slice_size = N * N;
        
        u_new[idx] = (1.0/6.0) * (u[idx-1] + u[idx+1] + 
                                 u[idx-N] + u[idx+N] +
                                 u[idx-slice_size] + u[idx+slice_size] - 
                                 d * d * f[idx]);
    }
}

__global__ void error_kernel(double* max_error, const double* u_new, const double* u,
                           int start_k, int end_k, int N) {
    __shared__ double shared_max_error[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

    int local_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + start_k;

    shared_max_error[local_id] = 0.0;

    if (i < N && j < N && k >= start_k && k < end_k) {
        int idx = i + j * N + (k-start_k) * N * N;
        double diff = fabs(u_new[idx] - u[idx]);
        shared_max_error[local_id] = diff;
    }

    __syncthreads();

    for (int stride = (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_max_error[local_id] = fmax(shared_max_error[local_id], 
                                            shared_max_error[local_id + stride]);
        }
        __syncthreads();
    }

    if (local_id == 0) {
        atomicMax(max_error, shared_max_error[0]);
    }
}

void initialize_fields(double* u, double* f, int start_k, int end_k, 
                      int N, double d) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = start_k; k < end_k; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                double x = i * d;
                double y = j * d;
                double z = k * d;
                int idx = i + j * N + (k-start_k) * N * N;
                
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || 
                    k == start_k || k == end_k-1) {
                    u[idx] = exact_solution(x, y, z);
                } else {
                    u[idx] = 0.0;
                }
                
                f[idx] = -12.0 * PI * PI * exact_solution(x, y, z);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Warning: The MPI implementation does not support MPI_THREAD_FUNNELED\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    selectGPU(rank);

    int num_threads = omp_get_max_threads();
    if (rank == 0) {
        printf("Using %d OpenMP threads per MPI rank\n", num_threads);
    }

    int num_gpus;
    hipGetDeviceCount(&num_gpus);
    int color = rank % num_gpus;
    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gpu_comm);

    int N = 512;
    double d = 1.0 / (N - 1);
    
    int base_slices_per_rank = N / size;
    int remaining_slices = N % size;

    int start_k, end_k;
    if (rank < remaining_slices) {
        start_k = rank * (base_slices_per_rank + 1);
        end_k = start_k + base_slices_per_rank + 1;
    } else {
        start_k = rank * base_slices_per_rank + remaining_slices;
        end_k = start_k + base_slices_per_rank;
    }

    int local_slices = end_k - start_k;
    size_t local_size = N * N * local_slices * sizeof(double);
    
    double *u_host = (double*)malloc(local_size);
    double *u_new_host = (double*)malloc(local_size);
    double *f_host = (double*)malloc(local_size);
    
    initialize_fields(u_host, f_host, start_k, end_k, N, d);
    
    #pragma omp parallel for
    for (int i = 0; i < N * N * local_slices; i++) {
        u_new_host[i] = u_host[i];
    }

    double *u_dev, *u_new_dev, *f_dev, *error_dev;
    hipMalloc(&u_dev, local_size);
    hipMalloc(&u_new_dev, local_size);
    hipMalloc(&f_dev, local_size);
    hipMalloc(&error_dev, sizeof(double));

    hipMemcpy(u_dev, u_host, local_size, hipMemcpyHostToDevice);
    hipMemcpy(u_new_dev, u_new_host, local_size, hipMemcpyHostToDevice);
    hipMemcpy(f_dev, f_host, local_size, hipMemcpyHostToDevice);

    double *send_top = (double*)malloc(N * N * sizeof(double));
    double *send_bottom = (double*)malloc(N * N * sizeof(double));
    double *recv_top = (double*)malloc(N * N * sizeof(double));
    double *recv_bottom = (double*)malloc(N * N * sizeof(double));

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (N + block_dim.x - 1) / block_dim.x,
        (N + block_dim.y - 1) / block_dim.y,
        (local_slices + block_dim.z - 1) / block_dim.z
    );

    int iter = 0;
    double local_error = 1.0, gpu_error = 1.0, global_error = 1.0;
    MPI_Request requests[4];
    
    auto start_time = std::chrono::high_resolution_clock::now();

    while (iter < MAX_ITER && global_error > TOL) {
        if (rank > 0) {
            hipMemcpy(send_top, u_dev + N * N, N * N * sizeof(double), hipMemcpyDeviceToHost);
            MPI_Isend(send_top, N * N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(recv_top, N * N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[1]);
        }
        
        if (rank < size-1) {
            hipMemcpy(send_bottom, u_dev + (local_slices-2) * N * N, 
                     N * N * sizeof(double), hipMemcpyDeviceToHost);
            MPI_Isend(send_bottom, N * N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(recv_bottom, N * N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[3]);
        }

        jacobi_kernel<<<grid_dim, block_dim>>>(u_new_dev, u_dev, f_dev, 
                                             start_k, end_k, N, d);
        hipDeviceSynchronize();

        if (rank > 0) {
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
            hipMemcpy(u_dev, recv_top, N * N * sizeof(double), hipMemcpyHostToDevice);
        }
        if (rank < size-1) {
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
            hipMemcpy(u_dev + (local_slices-1) * N * N, recv_bottom, 
                     N * N * sizeof(double), hipMemcpyHostToDevice);
        }

        hipMemset(error_dev, 0, sizeof(double));
        error_kernel<<<grid_dim, block_dim>>>(error_dev, u_new_dev, u_dev,
                                            start_k, end_k, N);
        hipMemcpy(&local_error, error_dev, sizeof(double), hipMemcpyDeviceToHost);
        
        MPI_Allreduce(&local_error, &gpu_error, 1, MPI_DOUBLE, MPI_MAX, gpu_comm);
        MPI_Allreduce(&gpu_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double* temp = u_dev;
        u_dev = u_new_dev;
        u_new_dev = temp;

        if (rank == 0 && iter % 100 == 0) {
            printf("Iteration %d, Error: %e\n", iter, global_error);
        }
        
        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    if (rank == 0) {
        double* full_solution = (double*)malloc(N * N * N * sizeof(double));
        
        hipMemcpy(u_host, u_dev, local_size, hipMemcpyDeviceToHost);
        
        #pragma omp parallel for
        for (int i = 0; i < local_slices * N * N; i++) {
            full_solution[start_k * N * N + i] = u_host[i];
        }
        
        for (int i = 1; i < size; i++) {
            int other_start_k = (i < remaining_slices) ? 
                i * (base_slices_per_rank + 1) : 
                i * base_slices_per_rank + remaining_slices;
                
            int other_slices = (i < remaining_slices) ? 
                base_slices_per_rank + 1 : 
                base_slices_per_rank;
                
            MPI_Recv(full_solution + other_start_k * N * N, 
                    other_slices * N * N, MPI_DOUBLE, i, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        printf("\nResults:\n");
        printf("Total iterations: %d\n", iter);
        printf("Total time: %.3f seconds\n", elapsed.count());
        printf("Final error: %.2e\n", global_error);

        double x_check = 0.01, y_check = 0.01, z_check = 0.01;
        int i_check = round(x_check * (N - 1));
        int j_check = round(y_check * (N - 1));
        int k_check = round(z_check * (N - 1));
        int idx_check = i_check + j_check * N + k_check * N * N;

        double numerical_sol = full_solution[idx_check];
        double exact_sol = exact_solution(x_check, y_check, z_check);
        
        printf("\nSolution verification at (0.01, 0.01, 0.01):\n");
        printf("Numerical solution: %.10f\n", numerical_sol);
        printf("Exact solution: %.10f\n", exact_sol);
        printf("Absolute error: %.10e\n", fabs(numerical_sol - exact_sol));
        
        free(full_solution);
    } else {
        hipMemcpy(u_host, u_dev, local_size, hipMemcpyDeviceToHost);
        MPI_Send(u_host, local_slices * N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&gpu_comm);
    hipFree(u_dev);
    hipFree(u_new_dev);
    hipFree(f_dev);
    hipFree(error_dev);
    free(u_host);
    free(u_new_host);
    free(f_host);
    free(send_top);
    free(send_bottom);
    free(recv_top);
    free(recv_bottom);

    MPI_Finalize();
    return 0;
}