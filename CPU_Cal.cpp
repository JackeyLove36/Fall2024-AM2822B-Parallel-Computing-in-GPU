#include <iostream>
#include <cmath>
#include <sys/time.h>

#define PI M_PI
#define TOL 1e-6
#define MAX_ITER 1e6

double exact_solution(double x, double y, double z) {
    return sin(2.0 * PI * x) * cos(2.0 * PI * y) * sin(2.0 * PI * z);
}

void initialize_fields(double* u, double* u_new, double* f, int N, double d) {
    for (int k = 0; k < N; k++) {
        double z = k * d;
        for (int j = 0; j < N; j++) {
            double y = j * d;
            for (int i = 0; i < N; i++) {
                double x = i * d;
                int idx = i + j * N + k * N * N;
                
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
                    u[idx] = exact_solution(x, y, z);
                    u_new[idx] = u[idx];
                } else {
                    u[idx] = 0;
                    u_new[idx] = 0;
                }
                
                f[idx] = -12.0 * PI * PI * exact_solution(x, y, z);
            }
        }
    }
}

double compute_iteration(double* u_new, double* u, double* f, int N, double d) {
    double max_error = 0.0;
    
    for (int k = 1; k < N-1; k++) {
        for (int j = 1; j < N-1; j++) {
            for (int i = 1; i < N-1; i++) {
                int idx = i + j * N + k * N * N;
                
                u_new[idx] = (1.0/6.0) * (u[idx-1] + u[idx+1] +  u[idx-N] + u[idx+N] + u[idx-N*N] + u[idx+N*N] - d * d * f[idx]);
                
                double diff = u_new[idx] - u[idx];
                if (diff < 0) diff = -diff;
                if (diff > max_error) {
                    max_error = diff;
                }
            }
        }
    }
    
    return max_error;
}

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
}

int main() {
    int N = 101;
    double d = 1.0 / (N - 1);
    
    double *u = new double[N * N * N];
    double *u_new = new double[N * N * N];
    double *f = new double[N * N * N];
    
    initialize_fields(u, u_new, f, N, d);
    
    int iter = 0;
    double error = 1.0;
    
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    while (iter < MAX_ITER && error > TOL) {
        error = compute_iteration(u_new, u, f, N, d);
        
        for (int i = 0; i < N*N*N; i++) {
            u[i] = u_new[i];
        }
        
        if (iter % 100 == 0) {
            printf("Iteration %d, Error: %e\n", iter, error);
        }
        iter++;
    }
    
    gettimeofday(&end_time, NULL);
    double compute_time = get_elapsed_time(start_time, end_time);
    
    printf("\nResults:\n");
    printf("Grid size: %d x %d x %d\n", N, N, N);
    printf("Total iterations: %d\n", iter);
    printf("Computation time: %.6f seconds\n", compute_time);
    
    double x_check = 0.01, y_check = 0.01, z_check = 0.01;
    printf("Solution at (0.01, 0.001, 0.001): numerical = %.6f, exact = %.6f\n",
           u[1 + N + N*N], exact_solution(x_check, y_check, z_check));
    
    delete[] u;
    delete[] u_new;
    delete[] f;
    
    return 0;
}