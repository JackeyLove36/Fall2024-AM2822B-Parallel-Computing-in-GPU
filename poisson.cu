#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <hip/hip_runtime.h>

#define PI M_PI
#define TOL 1e-6
#define MAX_ITER 1e6
#define BLOCK_SIZE 8  

__device__ __host__ double exact_solution(double x, double y, double z) {
    return sin(2.0 * PI * x) * cos(2.0 * PI * y) * sin(2.0 * PI * z);
}

__global__ void computing_function(double* u_new, const double* u, const double* f, 
                            int N, double d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1) {
        int idx = i + j * N + k * N * N;
        int slice_size = N * N;
        
        u_new[idx] = (1.0/6.0) * (u[idx-1] + u[idx+1] + u[idx-N] + u[idx+N] +u[idx-slice_size] + u[idx+slice_size] -d * d * f[idx]);
    }
}

__global__ void error_kernel(double* max_error, const double* u_new, const double* u, int N) {
    __shared__ double shared_max_error[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

    int local_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    shared_max_error[local_id] = 0.0;

    if (i < N && j < N && k < N) {
        int idx = i + j * N + k * N * N;
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

void initialize_fields(double* u, double* f, int N, double d) {
    for (int k = 0; k < N; k++) {
        double z = k * d;
        for (int j = 0; j < N; j++) {
            double y = j * d;
            for (int i = 0; i < N; i++) {
                double x = i * d;
                int idx = i + j * N + k * N * N;
                
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
                    u[idx] = exact_solution(x, y, z);
                } else {
                    u[idx] = 0;
                }
                
                f[idx] = -12.0 * PI * PI * exact_solution(x, y, z);
            }
        }
    }
}

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

int main() {
    int N = 101;  
    double d = 1.0 / (N - 1);
    
    size_t size = N * N * N * sizeof(double);
    double *u_host = (double*)malloc(size);
    double *u_new_host = (double*)malloc(size);
    double *f_host = (double*)malloc(size);
    
    initialize_fields(u_host, f_host, N, d);
    memcpy(u_new_host, u_host, size);

    double *u_dev, *u_new_dev, *f_dev, *error_dev;
    hipMalloc(&u_dev, size);
    hipMalloc(&u_new_dev, size);
    hipMalloc(&f_dev, size);
    hipMalloc(&error_dev, sizeof(double));

    struct timeval transfer_start, transfer_end;
    gettimeofday(&transfer_start, NULL);
    hipMemcpy(u_dev, u_host, size, hipMemcpyHostToDevice);
    hipMemcpy(u_new_dev, u_new_host, size, hipMemcpyHostToDevice);
    hipMemcpy(f_dev, f_host, size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    gettimeofday(&transfer_end, NULL);
    double transfer_time = get_elapsed_time(transfer_start, transfer_end);
    printf("Memory transfer time: %.6f seconds\n", transfer_time);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (N + block_dim.x - 1) / block_dim.x,
        (N + block_dim.y - 1) / block_dim.y,
        (N + block_dim.z - 1) / block_dim.z
    );
    
    int iter = 0;
    double error = 1.0;
    
    struct timeval compute_start, compute_end;
    gettimeofday(&compute_start, NULL);
    
    while (iter < MAX_ITER && error > TOL) {
        computing_function<<<grid_dim, block_dim>>>(u_new_dev, u_dev, f_dev, N, d);
        hipDeviceSynchronize();
        
        hipMemset(error_dev, 0, sizeof(double));
        error_kernel<<<grid_dim, block_dim>>>(error_dev, u_new_dev, u_dev, N);
        hipMemcpy(&error, error_dev, sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(u_dev, u_new_dev, size, hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        if (iter % 100 == 0) {
            printf("Iteration %d, Error: %e\n", iter, error);
        }
        iter++;
    }
    
    gettimeofday(&compute_end, NULL);
    double compute_time = get_elapsed_time(compute_start, compute_end);
    printf("Computation time: %.6f seconds\n", compute_time);
    
    size_t bytes_per_iter = size * 4; 
    double compute_bandwidth = (bytes_per_iter * iter) / (compute_time * 1e9); 
    double transfer_bandwidth = (size * 3) / (transfer_time * 1e9);
    
    printf("\nPerformance Results:\n");
    printf("Grid size: %d x %d x %d\n", N, N, N);
    printf("Total iterations: %d\n", iter);
    printf("Computation bandwidth: %.2f GB/s\n", compute_bandwidth);
    printf("Transfer bandwidth: %.2f GB/s\n", transfer_bandwidth);

    double x_check = 0.01, y_check = 0.01, z_check = 0.01;
    hipMemcpy(u_host, u_dev, size, hipMemcpyDeviceToHost);
    printf("Solution at (0.01, 0.01, 0.01): numerical = %.6f, exact = %.6f\n",
           u_host[1 + N + N*N], exact_solution(x_check, y_check, z_check));

    free(u_host);
    free(u_new_host);
    free(f_host);

    hipFree(u_dev);
    hipFree(u_new_dev);
    hipFree(f_dev);
    hipFree(error_dev);
    
    return 0;
}