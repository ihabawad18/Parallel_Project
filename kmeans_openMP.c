#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h> 

#define N 100000 // Number of points
#define K 20    // Number of clusters
#define NB_Threads 12
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
void assignPoints(Point points[N], Point centroids[K], int labels[N]) {

#pragma omp parallel for num_threads(NB_Threads)
    for (int i = 0; i < N; i++) {
        double minDist = DBL_MAX;
        int newLabel = 0;
        for (int j = 0; j < K; j++) {
            double dist = distance(points[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                newLabel = j;
            }
        }
        if(labels[i]!=newLabel)
            labels[i] = newLabel;
        }
}

void updateCentroids(Point points[N], Point centroids[K], int labels[N]) {
    
    Point local_centroids[K][NB_Threads];  // Array of centroids for each thread
    int local_count[K][NB_Threads];        // Array of counts for each thread
    int num_threads = NB_Threads;

    // Initialize local arrays + reset count and centroids global arrays
    for (int i = 0; i < K; i++) {
        count[i] = 0;
        centroids[i].x = 0;
        centroids[i].y = 0;
        for (int j = 0; j < num_threads; j++) {
            local_centroids[i][j].x = 0.0;
            local_centroids[i][j].y = 0.0;
            local_count[i][j] = 0;
        }
    }

#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            int thread_id = omp_get_thread_num();
            int label = labels[i];
            local_centroids[label][thread_id].x += points[i].x;
            local_centroids[label][thread_id].y += points[i].y;
            local_count[label][thread_id]++;
        }
    

    // Reduce results from all threads
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < num_threads; j++) {
            centroids[i].x += local_centroids[i][j].x;
            centroids[i].y += local_centroids[i][j].y;
            count[i] += local_count[i][j];
        }
        // Finalize centroid positions
        if (count[i] > 0) {
            centroids[i].x /= count[i];
            centroids[i].y /= count[i];
        }
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


int main() {

    // Read data from CSV files
    readCSV("generated_points_large_2.csv", points, N);
    // function to choose the initial centroids
    chooseCentroids(points, centroids);
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
        assignPoints(points, centroids, labels);
        updateCentroids(points, centroids, labels);
        
        // Check for convergence
        if (checkConvergence(centroids, old_centroids, centroid_movement_threshold)) {
            printf("Converged. Centroid movement below threshold.\n");
            break;
        }

        // Save current centroids for next iteration
        for (int j = 0; j < K; j++) {
            old_centroids[j] = centroids[j];
        }
        i++;
        printf("Iteration: %d\n", i);


    } while (i < max_iterations); // Continue until no changes in point assignments or max iterations reached

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; 

    printf("K-means clustering completed in %f seconds\n", cpu_time_used);
    printf("Cluster membership counts:\n");
    for (int i = 0; i < K; i++) {
        printf("Cluster %d has %d points.\n", i, count[i]);
       // printf("Coordinates of cluster %d: x %f , y %f\n", i, centroids[i].x, centroids[i].y);

    }
    printf("Final iteration: %d\n", i);
    return 0;
}