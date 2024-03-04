#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#include <vector>
#include <random>

class rnd_t {
    private:
        std::mt19937 gen;
        std::uniform_real_distribution<float> dis;

    public:
        rnd_t() : gen{std::random_device()()}, dis{0.0f,100.0f} {}
        double operator()() { return dis(gen); }
};

__global__ void relaxation(float* grid_new, float* grid_old, int Nx, int Ny) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = row*Nx + col;
    // dont update the boundary points. this if statement will probably be slow
    if (!(col < 1 || col == Ny-1 || row < 1 || row == Nx-1))
        grid_new[tid] = 0.25f*(grid_old[(row-1)*Nx+col] + grid_old[(row+1)*Nx+col]
                + grid_old[row*Nx+(col-1)]+grid_old[row*Nx+(col+1)]);
}

__global__ void ComputeResidual(int Nx, int Ny, float* grid_new, float* grid_old, float* result) {

    // Declare some dynamic shared memory for all threads in this block
    extern __shared__ float residual[];

    // Identify which thread we're using in this instance
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // Compute the residual for this grid point
    residual[tid] = sqrt((grid_new[i]-grid_old[i])*(grid_new[i]-grid_old[i]));
    __syncthreads();

    /**
     * Reduce Method: we can just treat this as a 1D reduction
     */
    // Note that the iterator is updated using a bit shift operator
    // each iteration, s is set to itself shifted by 2 bits
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            residual[tid] += residual[tid+s];
        }
        __syncthreads();
    }

    // write result for this block
    if (tid == 0) result[blockIdx.x] = residual[0];
}

void PrintMatrix(float* A, int Nx, int Ny) {
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) printf("%f, ", A[i+j*Nx]);
        printf("\n");
    }
    printf("\n");
}

// Enter an array and the number of elements to print
// For this to work, we have to be able to sync the threads and can therefore only use a single block
__global__ void PrintMatrix_D(float* A, int Nx, int Ny) {
    int tid = threadIdx.x;
    for (int j = 0; j < Nx*Ny; ++j) {
        if (tid == j) {
            printf("%f, ",A[tid]);
        }
        __syncthreads();
    }
}

void WriteMatrix(float* A, int Nx, int Ny) {
    FILE* fp = fopen("FinalGrid.txt", "w");
    fprintf(fp,"# %d %d\n",Nx,Ny);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) fprintf(fp,"%f, ", A[i+j*Nx]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    int Tx, Ty, Gx, Gy;
    if (argc != 5) {
        printf("Error!\n\tUsage: ./V4 Gx Gy Tx Ty");
        exit(1);
    } else {
        Gx = atof(argv[1]);
        Gy = atof(argv[2]);
        Tx = atof(argv[3]);
        Ty = atof(argv[4]);
    }

    // Define grid size based on the block size and number of threads
    const int Nx = Gx*Tx;
    const int Ny = Gy*Ty;
    // Lets store the data in a single array row by row
    size_t size = Nx*Ny*sizeof(float); // size in bytes
    size_t size_grid = Tx*Ty*sizeof(float);
    // Allocate memory on host and device
    float* h_grid = (float*)malloc(size);
    // Allocate Global Memory on device for Grid Arrays
    float* d_grid;
    float* d_grid_new;
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_grid_new, size);

    // set dirichlet boundary conditions and initialize interior points to random numbers between 0 and 100
    rnd_t random;
    for (int i = 0; i < Nx; ++i) {
        h_grid[i] = 100.0f;
    }
    for (int j = 1; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            if (i == 0 || i == Nx - 1 || j == Ny - 1)
                h_grid[i+j*Nx] = 0.0f;
            else
                h_grid[i+j*Nx] = random();
        }
    }
    cudaMemcpy(d_grid,h_grid,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new,h_grid,size,cudaMemcpyHostToDevice);

    // define block and grid size
    dim3 gridsize(Gx,Gy,1);
    dim3 blocksize(Tx,Ty,1);

    /** Main Loop **/
    float max_error = 1e-6;
    int max_iter = 1'000'000;
    float total_residual = 1.0f;
    int iter = 0;
    // Define some variables for computing the residual
    float* h_residual = (float*)malloc(Gx*Gy*sizeof(float));
    float* d_residual;
    cudaMalloc(&d_residual,Gx*Gy*sizeof(float));
    auto time_start = std::chrono::high_resolution_clock::now();
    while (((total_residual) > max_error) && iter < max_iter) {
        iter++;
        // Compute grid
        relaxation<<<gridsize,blocksize>>>(d_grid_new, d_grid, Nx, Ny);
        // Compute the residual by flattening the grid block: there's probably a better way of doing this
        total_residual = 0.0f;
        ComputeResidual<<<Gx*Gy,Tx*Ty,size_grid>>>(Tx,Ty,d_grid,d_grid_new,d_residual);
        cudaMemcpy(h_residual,d_residual,Gx*Gy*sizeof(float),cudaMemcpyDeviceToHost);
        for (int i = 0; i < Gx*Gy; ++i) total_residual += h_residual[i];
        // Update the grid
        cudaMemcpy(d_grid,d_grid_new,size,cudaMemcpyDeviceToDevice);
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_delta = time_end-time_start;

    // Print out results
    std::cout << Gx << " " << Gy << " " << Tx << " " << Ty
        << " " << iter << " " << std::setprecision(4) << time_delta.count() << std::endl;

    // Print Matrix
    cudaMemcpy(h_grid,d_grid,size,cudaMemcpyDeviceToHost);
    PrintMatrix(h_grid,Nx,Ny);

    // Cleanup
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    cudaFree(d_residual);
    free(h_grid);
    free(h_residual);

    return 0;
}
