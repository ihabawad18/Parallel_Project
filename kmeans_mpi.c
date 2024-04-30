#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 100000  // Number of points
#define K 20      // Number of clusters
#define NUM_THREADS 512  // Threads per block

typedef struct {
    double x;
    double y;
} Point;

// Device function to calculate distance between two points
device double distance_gpu(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
// Host function to calculate distance for convergence checking
double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
// Custom atomicAdd function for double-precision floating points
device double atomicAdd_custom(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

global void assignPoints(Point* points, Point* centroids, int* labels, int numPoints, int numCentroids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numPoints) {
        double minDist = DBL_MAX;
        int bestIndex = 0;
        for (int i = 0; i < numCentroids; i++) {
            double dist = distance_gpu(points[index], centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                bestIndex = i;
            }
        }
        labels[index] = bestIndex;
    }
}

global void updateCentroids(Point* points, Point* centroids, int* labels, int* counts, int numPoints, int numCentroids) {
    extern shared char shared_memory[];
    Point* sharedCentroids = (Point*)shared_memory;
    int* sharedCounts = (int*)(shared_memory + numCentroids * sizeof(Point));

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;

    // Initialize shared memory
    if (threadId < numCentroids) {
        sharedCentroids[threadId].x = 0;
        sharedCentroids[threadId].y = 0;
        sharedCounts[threadId] = 0;
    }
    __syncthreads();

    // Update shared memory
    if (index < numPoints) {
        int label = labels[index];
        atomicAdd_custom(&sharedCentroids[label].x, points[index].x);
        atomicAdd_custom(&sharedCentroids[label].y, points[index].y);
        atomicAdd(&sharedCounts[label], 1);
    }
    __syncthreads();

    // Reduce shared memory to global memory
    if (threadId < numCentroids) {
        atomicAdd_custom(&centroids[threadId].x, sharedCentroids[threadId].x);
        atomicAdd_custom(&centroids[threadId].y, sharedCentroids[threadId].y);
        atomicAdd(&counts[threadId], sharedCounts[threadId]);
    }
}

void readCSV(const char* filename, Point* array) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        if (fscanf(file, "%lf,%lf", &array[i].x, &array[i].y) != 2) {
            fprintf(stderr, "Error reading file at row %d\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}
// Comparison function for sorting points based on x-coordinate
int comparePointsX(const void* a, const void* b) {
    Point* pointA = (Point*)a;
    Point* pointB = (Point*)b;
    if (pointA->x < pointB->x) return -1;
    if (pointA->x > pointB->x) return 1;
    return 0;
}

Point sortedPoints[N];

// Function for choosing centroids using sorting
void chooseCentroids(Point points[N], Point centroids[K]) {
    // Create a temporary array to store points for sorting
    for (int i = 0; i < N; i++) {
        sortedPoints[i] = points[i];
    }

    // Sort the temporary array based on x-coordinate
    qsort(sortedPoints, N, sizeof(Point), comparePointsX);

    // Choose centroids from sorted points
    for (int i = 0; i < K; i++) {
        centroids[i] = sortedPoints[i * (N / K)];
    }
}


// function to check convergence
int checkConvergence(Point* centroids, Point* old_centroids, double threshold) {
    double total_movement = 0.0;
    for (int i = 0; i < K; i++) {
        total_movement += distance(centroids[i], old_centroids[i]);
    }
    return (total_movement < threshold);
}

int main() {
    Point* h_points = (Point*)malloc(N * sizeof(Point));
    Point* h_centroids = (Point*)malloc(K * sizeof(Point));
    int* h_labels = (int*)malloc(N * sizeof(int));
    int* h_counts = (int*)calloc(K, sizeof(int));
    Point* old_centroids = (Point*)malloc(K * sizeof(Point));

    readCSV("generated_points_large_2.csv", h_points);
    chooseCentroids(h_points, h_centroids);

    // Initialize old centroids for comparison
    for (int i = 0; i < K; i++) {
        old_centroids[i] = h_centroids[i];
    }

    Point* d_points, * d_centroids;
    int* d_labels, * d_counts;
    cudaMalloc(&d_points, N * sizeof(Point));
    cudaMalloc(&d_centroids, K * sizeof(Point));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_counts, K * sizeof(int));

    cudaMemcpy(d_points, h_points, N * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * sizeof(Point), cudaMemcpyHostToDevice);

    dim3 blocks((N + NUM_THREADS - 1) / NUM_THREADS);
    dim3 threads(NUM_THREADS);

    clock_t start = clock();

    int iterations = 0;

    int max_iterations = 105;
    double centroid_movement_threshold = 0.01;
    do {

        assignPoints << <blocks, threads >> > (d_points, d_centroids, d_labels, N, K);
        cudaDeviceSynchronize();
        //clear centroids and counts of device
        cudaMemset(d_centroids, 0, K * sizeof(Point));
        cudaMemset(d_counts, 0, K * sizeof(int));
        cudaDeviceSynchronize();

        updateCentroids << <blocks, threads >> > (d_points, d_centroids, d_labels, d_counts, N, K);
        cudaDeviceSynchronize();

        cudaMemcpy(h_centroids, d_centroids, K * sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost);

        // finalize centroids positions
        for (int j = 0; j < K; j++) {
            if (h_counts[j] != 0) {
                h_centroids[j].x /= h_counts[j];
                h_centroids[j].y /= h_counts[j];
            }

        }
        // Check for convergence
        if (checkConvergence(h_centroids, old_centroids, centroid_movement_threshold)) {
            printf("Converged. Centroid movement below threshold.\n");
            break;
        }
        //recopy from host to device
        cudaMemcpy(d_centroids, h_centroids, K * sizeof(Point), cudaMemcpyHostToDevice);
        
        // update old centroids array
        for (int j = 0; j < K; j++) {
            old_centroids[j] = h_centroids[j];

        }
        iterations++;
        printf("Iteration: %d\n", iterations);

    } while (iterations < max_iterations);


    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost);

    printf("K-means clustering completed in %f seconds\n", cpu_time_used);
    printf("Cluster membership counts:\n");
    for (int i = 0; i < K; i++) {
        printf("Cluster %d has %d points.\n", i, h_counts[i]);
    }

    free(h_points);
    free(h_centroids);
    free(h_labels);
    free(h_counts);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);

    return 0;
}