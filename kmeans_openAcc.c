#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 100000  // Number of points
#define K 20      // Number of clusters

typedef struct {
    double x;
    double y;
} Point;

// Function to calculate distance between two points
double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Comparison function for sorting points based on x-coordinate
int comparePointsX(const void* a, const void* b) {
    Point* pointA = (Point*)a;
    Point* pointB = (Point*)b;
    if (pointA->x < pointB->x) return -1;
    if (pointA->x > pointB->x) return 1;
    return 0;
}

// Function for choosing centroids using sorting
void chooseCentroids(Point points[N], Point centroids[K]) {
    Point sortedPoints[N];
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

// Function to check convergence
int checkConvergence(Point* centroids, Point* old_centroids, double threshold) {
    double total_movement = 0.0;
    for (int i = 0; i < K; i++) {
        total_movement += distance(centroids[i], old_centroids[i]);
    }
    return (total_movement < threshold);
}

int main() {
    clock_t start = clock();

    Point* points = (Point*)malloc(N * sizeof(Point));
    Point* centroids = (Point*)malloc(K * sizeof(Point));
    int* labels = (int*)malloc(N * sizeof(int));
    int* counts = (int*)calloc(K, sizeof(int));
    Point* old_centroids = (Point*)malloc(K * sizeof(Point));
    double* partial_sums_x = (double*)malloc(K * sizeof(double));
    double* partial_sums_y = (double*)malloc(K * sizeof(double));

    // Read points and choose initial centroids
    readCSV("generated_points_large_2.csv", points);
    chooseCentroids(points, centroids);

    int max_iterations = 105;
    double threshold = 0.01;
    int converged = 0;

    // Data region - copy points and centroids to device, create labels and counts
    #pragma acc data copy(points[0:N], centroids[0:K], labels[0:N], counts[0:K], old_centroids[0:K], partial_sums_x[0:K], partial_sums_y[0:K])
    {
        for (int iter = 0; iter < max_iterations && !converged; iter++) {
            // Zero out centroids and counts
            #pragma acc parallel loop
            for (int i = 0; i < K; i++) {
                partial_sums_x[i] = 0;
                partial_sums_y[i] = 0;
                counts[i] = 0;
            }

            // Kernel to assign each point to the nearest centroid
            #pragma acc parallel loop
            for (int i = 0; i < N; i++) {
                double minDist = DBL_MAX;
                int bestIndex = -1;
                for (int j = 0; j < K; j++) {
                    double dist = distance(points[i], centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestIndex = j;
                    }
                }
                labels[i] = bestIndex;
            }

            // Kernel to update centroids based on assigned points
            #pragma acc parallel loop
            for (int i = 0; i < N; i++) {
                int label = labels[i];
                partial_sums_x[label] += points[i].x;
                partial_sums_y[label] += points[i].y;
                counts[label]++;
            }

            // Combine partial sums
            #pragma acc parallel loop
            for (int i = 0; i < K; i++) {
                centroids[i].x = partial_sums_x[i];
                centroids[i].y = partial_sums_y[i];
            }

            // Normalize centroids
            #pragma acc parallel loop
            for (int i = 0; i < K; i++) {
                if (counts[i] > 0) {
                    centroids[i].x /= counts[i];
                    centroids[i].y /= counts[i];
                }
            }

            // Check for convergence
            double total_movement = 0.0;
            #pragma acc parallel loop reduction(+:total_movement)
            for (int i = 0; i < K; i++) {
                total_movement += distance(centroids[i], old_centroids[i]);
            }

            if (total_movement < threshold) {
                converged = 1;
            }

            // Copy centroids to old centroids for next iteration
            #pragma acc parallel loop
            for (int i = 0; i < K; i++) {
                old_centroids[i] = centroids[i];
            }

            printf("Iteration: %d\n", iter + 1);
        }
    }

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("K-means clustering completed in %f seconds\n", cpu_time_used);
    printf("Cluster membership counts:\n");
    for (int i = 0; i < K; i++) {
        printf("Cluster %d has %d points.\n", i, counts[i]);
    }

    free(points);
    free(centroids);
    free(labels);
    free(counts);
    free(old_centroids);
    free(partial_sums_x);
    free(partial_sums_y);

    return 0;
}