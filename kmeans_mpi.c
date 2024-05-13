#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <mpi.h>

#define N 100000 // Number of points
#define K 20     // Number of clusters

typedef struct {
    double x;
    double y;
} Point;

int count[K] = { 0 }; // Global array to store the count of points in each cluster

// Function to calculate Euclidean distance between two points
double distance(Point point1, Point point2) {
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

// Function to assign points to the nearest cluster center
void assignPoints(Point points[N], Point centroids[K], int labels[N], int start, int end) {
    for (int i = start; i < end; i++) {
        double minDist = DBL_MAX;
        int newLabel = 0;
        for (int j = 0; j < K; j++) {
            double dist = distance(points[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                newLabel = j;
            }
        }
        if (labels[i] != newLabel) {
            labels[i] = newLabel;
        }
    }
}

// Function to update the cluster centroids
void updateCentroids(Point points[N], Point centroids[K], int labels[N], int start, int end) {
    // Reset counts to 0 for all clusters
    for (int i = 0; i < K; i++) {
        count[i] = 0;
        centroids[i].x = 0;
        centroids[i].y = 0;
    }

    // Update centroids and count points in each cluster
    for (int i = start; i < end; i++) {
        int label = labels[i];
        centroids[label].x += points[i].x;
        centroids[label].y += points[i].y;
        count[label]++;
    }
}

void readCSV(const char* filename, Point array[], int rows) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        if (fscanf(file, "%lf,%lf,", &array[i].x, &array[i].y) != 2) {
            fprintf(stderr, "Error reading file at row %d\n", i + 1);
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
void chooseCentroids(Point points[N], Point centroids[K], int rank, int size) {
    // Create a temporary array to store points for sorting
    for (int i = 0; i < N; i++) {
        sortedPoints[i] = points[i];
    }

    // Sort the temporary array based on x-coordinate
    qsort(sortedPoints, N, sizeof(Point), comparePointsX);

    // Choose centroids from sorted points
    for (int i = 0; i < K; i++) {
        centroids[i] = sortedPoints[(i * N) / K];
    }
}

// Function to check convergence based on centroid movement threshold
int checkConvergence(Point centroids[K], Point old_centroids[K], double threshold) {
    double total_movement = 0.0;
    for (int i = 0; i < K; i++) {
        total_movement += distance(centroids[i], old_centroids[i]);
    }
    return (total_movement < threshold);
}

Point points[N];          
Point centroids[K];       // Centroids of the clusters
Point old_centroids[K];   // Previous centroids
int labels[N] = { 0 };    // Cluster labels for each point

int global_changed = 0;     


int main() {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read data from CSV file
    readCSV("generated_points_large_2.csv", points, N);

    // Initialize centroids using choose centroids function
    chooseCentroids(points, centroids, rank, size);

    // Initialize old centroids for comparison
    for (int i = 0; i < K; i++) {
        old_centroids[i] = centroids[i];
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    int i = 0;
    int max_iterations = 105; // Maximum number of iterations
    double centroid_movement_threshold = 0.01; // Centroid movement threshold

    // Main K-means algorithm with convergence check
    do {
        // Assign points to the nearest cluster center
        assignPoints(points, centroids, labels, rank * (N / size), (rank + 1) * (N / size));

        // Update centroids based on local data
        updateCentroids(points, centroids, labels, rank * (N / size), (rank + 1) * (N / size));

        // Synchronize centroids across processes
        MPI_Allreduce(MPI_IN_PLACE, centroids, K * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // Synchronize counts across processes
        MPI_Allreduce(MPI_IN_PLACE, count, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Finalize centroid positions
        for (int j = 0; j < K; j++) {
            if (count[j] != 0) {
                centroids[j].x /= count[j];
                centroids[j].y /= count[j];
            }
        }

        i++;
        if (rank == 0) {
            printf("Rank %d, Iteration: %d\n", rank, i);
        }

        // Check for convergence on the root process
        if (checkConvergence(centroids, old_centroids, centroid_movement_threshold)) {
            if (rank == 0) {
                printf("Converged. Centroid movement below threshold.\n");
            }
            break;
        }

        // Save current centroids for next iteration
        for (int j = 0; j < K; j++) {
            old_centroids[j] = centroids[j];
        }

    } while (i < max_iterations); // Continue until max iterations reached

    end = clock(); 
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

   
    if (rank == 0) {
        printf("K-means clustering completed in %f seconds\n", cpu_time_used);
        printf("Cluster membership counts:\n");
        for (int i = 0; i < K; i++) {
            printf("Cluster %d has %d points.\n", i, count[i]);
        }
        printf("Final iteration: %d\n", i);
    }

    MPI_Finalize();
    return 0;
}