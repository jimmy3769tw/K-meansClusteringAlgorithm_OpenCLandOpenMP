#pragma once 
#include <vector>
#include <CL/cl2.hpp>

using std::vector;
/* 
The struct myData stores:
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/     
struct myData {
    static const int D2_1 = 16 ;
    static const int D2_2 = 16 ;
    static const int D1 = 256;
	size_t DIM; 
	size_t N; 
	size_t K; 
	vector<float> pts; 
	vector<float> currentCentroids;
	vector<float> oldCentroids; 
	vector<int> group;

    // vector<cl::Platform> platform;
    cl::Platform platform;
    // vector<cl::Device> device;
    cl::Device device;
    cl::Context ctx;
    cl::Program prg;
    cl::CommandQueue queue;
    cl_command_queue queue11;
    // * Buffer
    cl::Buffer dev_pts, dev_currentCentroids, dev_oldCentroids, dev_group;
}; 

// 0. Read CSV file and save data into myData.
// returns false if something went wrong; otherwise returns true.
bool readCSV(myData&, const char* filename); 

// 1. initializes K randomly selected centroids, and you should set K to noClusters 
//    and allocate memory for storing centroids.
void InitializeCentroid(myData&, size_t noClusters);

// 2. for each point, find the nearest centroid and assign to that group
void AssignGroups(myData&);

// 3. copy data in cCentroid into pCentroid, and then update centroids based on the kmeans algorithm.
void UpdateCentroids(myData&);

// 4. find the centroid that moves furtherest, and if its moving distance is less than the given tolerance, then we say it has converged.
bool HasConverged(myData&, const float tolerance=1.0e-6);
