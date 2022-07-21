using namespace std;

// config cl2.hpp through Macros
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#include "../inc/stopWatch.hpp"
#include "../inc/kmeans.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <string>
#include <map>


auto CL_SRC = R"(
	// *AssignGroups



__kernel 
void distanceK(__global const float *pts,__global const float *dev_currentCentroids, uint N, uint K,uint DIM, __global float *dev_distance){
	ulong whichPt = get_global_id(0);
	ulong i = get_global_id(1);
	if ( i < K && whichPt < N){
		float distance_sum_DIM = 0.0f;
		for (uint whichDim = 0 ; whichDim < DIM ; whichDim ++){
			distance_sum_DIM += pow(dev_currentCentroids[i+whichDim*K]- pts[whichPt*DIM+whichDim],2); //AOS
		}
		dev_distance[whichPt * K + i] = distance_sum_DIM;
	}
}


__kernel 
void assignGroup(__global const float *dev_distance, __global int *group, uint N, uint K){
	ulong whichPt = get_global_id(0);
	// __local float ptr[K];
	float ptr[16];
	for(uint i = 0  ; i < K ; ++i){
		ptr[i] = dev_distance[whichPt * K + i];
	}
	if ( whichPt < N){
		float min = ptr[0];
		int grp = 0;
		for (uint i = 0; i < K; i++){
			if(ptr[i] < min){
				min = ptr[i]; 
				grp = i;
			}
		}
		group[whichPt] = grp;
	}
}

__kernel
void assignGroupT(__global const float *dev_distance, __global int *group, uint N, uint K){
	ulong whichPt = get_global_id(0);
	__global float *ptr = dev_distance + whichPt * K;

	if ( whichPt < N){
		float min = ptr[0];
		int grp = 0;
		for (uint i = 0; i < K; i++){
			if(ptr[i] < min){
				min = ptr[i]; 
				grp = i;
			}
		}
		group[whichPt] = grp;
	}
}


// * UpdateCenroids

__kernel
void myAtomicAddG(__global float *addr, float val) {
	union {
		uint u32;
		float f32;
	} current, expected, next;

	do {
		current.f32 = *addr;
		next.f32 = current.f32 + val;
		expected.u32 = current.u32;
		current.u32 = atomic_cmpxchg( (__global uint*) addr, expected.u32, next.u32 );
	} while( current.u32 != expected.u32 );
}


__kernel 
void myAtomicAddL(__local float *addr, float val) {
	union {
		uint u32;
		float f32;
	} current, expected, next;

	do {
		current.f32 = *addr;
		next.f32 = current.f32 + val;
		expected.u32 = current.u32;
		current.u32 = atomic_cmpxchg( (__local uint*) addr, expected.u32, next.u32 );
	} while( current.u32 != expected.u32 );
}


__kernel
void sumCentroidA( __global const int* group,__global int *count, uint N, uint K){
	ulong whichPt = get_global_id(0);
	if(whichPt >= N) return;
	__private int i = group[whichPt];
	atomic_inc( count+i );
}


__kernel
void sumCentroidB( __global const int* group, __global float *centroids,__global float *pts, uint N, uint K, uint DIM){
	ulong whichPt = get_global_id(0);
	uint whichDim = get_global_id(1);

	if(whichPt >= N || whichDim >= DIM) return;

	__private int i = group[whichPt];

	// centroids [ i+whichDim*K ] += pts[whichPt*DIM + whichDim];
	myAtomicAddG(centroids +i + whichDim*K, pts[whichPt*DIM + whichDim]); //*AOS
}



__kernel
void sumCentroidB1( __global const int* group, __global float *centroids,__global float *pts, uint N, uint K, uint DIM){
	ulong whichPt = get_global_id(0);
	uint whichDim = get_global_id(1);
	__private int i = group[whichPt];

	if (whichDim >= DIM) return;

	// * Local
	uint whichPt_ = get_local_id(0);
	__local float center_[1][8][8];

	if (get_local_id(0) ==0){
		for(uint id = 0 ; id < 8 ; ++id){
			for(uint ik = 0 ; ik < 8 ; ++ik){
				center_[0][id][ik] = 0.0f;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (whichPt < N){
		myAtomicAddL(center_ + whichDim*8+i, pts[whichPt*DIM+whichDim]);// *AOS
	}
	
	// * Global

	if (get_local_id(0) ==0){
		myAtomicAddG(centroids +i + whichDim*K, center_[0][whichDim][i]);
	}

}


__kernel
void sumCentroidB2( __global const int* group, __global float *centroids,__global float *pts, uint N, uint K, uint DIM){
	ulong whichPt = get_global_id(0);
	uint whichDim = get_global_id(1);
	__private int i = group[whichPt];

	if (whichDim >= DIM) return;
	// * Local
	uint whichPt_ = get_local_id(0);
	__local float center_[170][8][8];

	if (whichPt < N){
		center_[whichPt_][whichDim][i] = pts[whichPt*DIM+whichDim];// *AOS
	}
	else{
		center_[whichPt_][whichDim][i] = 0.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint offset = get_local_size(0)/2 ; offset > 0 ; offset /= 2) {
		if (whichPt_ < offset){
			center_[whichPt_][whichDim][i] += center_[whichPt_+offset][whichDim][i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	

	// * Global

	// centroids [ i+whichDim*K ] += pts[whichPt*DIM + whichDim];
	if(whichPt_==0){
		myAtomicAddG(centroids +i + whichDim*K, center_[0][whichDim][i]);
	}
		

}


__kernel
void avcgCentroid(__global float* centroids, __global int* groupCount, uint DIM, uint K){
	uint i = get_global_id(0);
	uint whichDim = get_global_id(1);
	if(i >= K || whichDim >= DIM) return;

	centroids[i+whichDim*K] /= (groupCount[i] );
}

__kernel
void hasConverged(__global float const *centroids, __global float const *oldCentroids, __global int *result, uint DIM, uint K,float tolerance){
	uint i = get_global_id(0);
	if (i > K) return;

	float distance = 0.0f;
	for(uint whichDim = 0; whichDim < DIM ; whichDim++ ){
		distance += pow(oldCentroids[i+whichDim*K] - centroids[i+whichDim*K],2);
	}

	if(distance > tolerance){
		atomic_inc(result);
	}
		

	// atomic_inc(result, distance > tolerance);
}
		// TODO : 其實 center_tem 可以省略->減少頻寬要求，但是會增加 atomic 區域 另外最後一個 事情(myid=0)其實沒有必要平行但是，不想下交流道(交給cpu做事)
		// TODO : 通常 OpenCL 不要使用 if 但是現在不清楚 if 成本與迴圈成本增加成本，所以一律先加入

)";


/*
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/

// A utility function for getting a specific platform based on vendor's name
auto getPlatform(const std::string& vendorNameFilter) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(const auto& p: platforms) {
        if(p.getInfo<CL_PLATFORM_VENDOR>().find(vendorNameFilter) != std::string::npos) {
            return p;
        }
    }
    throw cl::Error(CL_INVALID_PLATFORM, "No platform has given vendorName");
}


// A utility function for getting a device based on the amount of global memory.
auto getDevice(cl::Platform& platform, cl_device_type type, size_t globalMemoryMB) {
    std::vector<cl::Device> devices;
    platform.getDevices(type, &devices);
    globalMemoryMB *= 1024 * 1024; // from MB to bytes
    for(const auto& d: devices) {
        if( d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() >= globalMemoryMB ) return d;
    }
    throw cl::Error(CL_INVALID_DEVICE, "No device has needed global memory size");
}



bool readCSV(myData& data, const char *filename) {
	size_t noAttributes = 0; 
	size_t noLines = 0;

	ifstream inp(filename);
	if (!inp) {
		return false;
	}
	char buf[4096];
	inp.getline(buf, 4096);
	for(auto it = buf; *it!=0; ++it) {
		if(*it==',') ++noAttributes;
	}

	noLines = 1;
	// let's count # lines first.
	while (inp.good()) {
		inp.getline(buf, 4096);
 		if( inp.gcount()>0 ) noLines++;
	}
	data.N = noLines;
	data.DIM = noAttributes;
	// cout << "noAttributes is the number of " << noAttributes <<endl;
	// cout << "noLines is the number of " << noLines << endl;
	// cout << "noLines*noAttributes is equal to "  << noLines*noAttributes << endl;

	inp.clear();
	inp.seekg(0, inp.beg);
	data.pts.resize(data.N*data.DIM);

	for (size_t whichPt = 0; whichPt < noLines; ++whichPt) {
		inp.getline(buf, 4096);
		auto it = buf;
		for (size_t whichDim = 0; whichDim < data.DIM;++whichDim) {
			auto x = atof(it);
			data.pts[whichPt*data.DIM+whichDim] = x;
			while(*it!=',') ++it;
			it++;
		}
	}

	inp.close();

	//platform A1'
	data.platform = getPlatform("NVIDIA");       // Mesa, pocl
	// data.platform = getPlatform("Mesa");
	// data.platform = getPlatform("pocl");

    // device A2
	data.device = getDevice(data.platform,CL_DEVICE_TYPE_GPU, 1024); // CL_DEVICE_TYPE_ALL,1024 -> 1G

	// device A3
	data.ctx = cl::Context(data.device);

	// queue B3
	data.queue = cl::CommandQueue(data.ctx,data.device);
	data.queue11 = data.queue();
	// context A3

	// B1
	data.prg = cl::Program(data.ctx, CL_SRC);

	try {
		data.prg.build();
	} catch (cl::Error &e) {
		std::cerr << "\n" << data.prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(data.device);
		throw cl::Error(CL_INVALID_PROGRAM, "Failed to build kernel");
	}

	// create a buffer and fill it with data from an array C
	data.dev_pts = cl::Buffer(data.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
		data.N*data.DIM*sizeof(float), data.pts.data());



	return true;
}



// AOS version
// array of structs x1, y1, z1, x2, y2, z2, ...
void InitializeCentroid(myData& data, size_t noClusters) {

	data.K = noClusters;
	data.currentCentroids.resize(data.K*data.DIM); 
	data.oldCentroids.resize(data.K*data.DIM); 
	data.group.resize(data.N); 
	// Buffer
	data.dev_currentCentroids = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K*data.DIM*sizeof(float), data.pts.data());
	data.dev_oldCentroids = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K*data.DIM*sizeof(float), data.pts.data());
	data.dev_group = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.N*sizeof(cl_int));

	for (size_t i = 0; i < data.K; i++){
		size_t whichPt = (data.N / data.K) * (i);
		for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
			data.currentCentroids[i+whichDim*data.K] = data.pts.at(whichPt*data.DIM+whichDim);
		}
	}

	data.queue.enqueueWriteBuffer(data.dev_currentCentroids, CL_TRUE, 0, data.K*data.DIM*sizeof(float), data.currentCentroids.data());
}

void AssignGroups(myData& data) {
	const int which = 0;
	std::map<int, const char*> distanceKernelMap {
		{0,"assignGroup"},
		{1,"assignGroupT"},
	};

	// static reduce time of construct
	static cl::Buffer dev_distance(data.ctx, CL_MEM_READ_WRITE|CL_MEM_HOST_NO_ACCESS, data.N * data.K*sizeof(float));
	static cl::KernelFunctor<const cl::Buffer&,const cl::Buffer&, cl_uint, cl_uint, cl_uint, cl::Buffer& > distanceKernel(data.prg, "distanceK");
	static cl::KernelFunctor<const cl::Buffer&, cl::Buffer&, cl_uint, cl_uint> assingGroupKernel(data.prg, distanceKernelMap[which]);
	// EnqueueArgs (CommandQueue &queue, NDRange global, NDRange local)

	// * setup work items (global_ , local_)
					// n				for 整除  ->  16 work items * 16 work items
	auto config2D = cl::EnqueueArgs(data.queue,{(data.N+data.D2_1-1)/data.D2_1*data.D2_1,(data.K+data.D2_2-1)/data.D2_2*data.D2_2},{data.D2_1,data.D2_2}); 
	auto config1D = cl::EnqueueArgs(data.queue,(data.N+data.D1-1)/data.D1*data.D1,data.D1);

	distanceKernel(config2D, data.dev_pts, data.dev_currentCentroids, data.N, data.K, data.DIM, dev_distance);
	assingGroupKernel(config1D,dev_distance, data.dev_group,data.N, data.K);

	clFinish(data.queue11);
	
	//! Check begin
	// const size_t N_value = data.N;
	// const size_t K_value = data.K;
	// const size_t DIM_value = data.DIM;
	// vector<float> dis_check(N_value * K_value,1.);
	// vector<int> group_check(N_value,1.);
    // data.queue.enqueueReadBuffer(dev_distance, CL_TRUE, 0,data.N*data.K*sizeof(float), dis_check.data() );
    // data.queue.enqueueReadBuffer(data.dev_group, CL_TRUE, 0, data.N*sizeof(cl_int), group_check.data());
    // data.queue.enqueueReadBuffer(data.dev_currentCentroids, CL_TRUE, 0, data.K*data.DIM*sizeof(float), data.currentCentroids.data());
	// float distance_sum_DIM;
	// float dis_min = 0.;

	// float dis[N_value][K_value]= {0.};
	// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 	for (size_t count_k = 0; count_k < K_value; count_k++){
	// 		distance_sum_DIM = 0;
	// 		for (size_t whichDim = 0; whichDim < DIM_value ; whichDim++){
	// 			distance_sum_DIM += pow((data.currentCentroids.at(count_k+whichDim*data.K)- data.pts.at(whichPt*data.DIM+whichDim)),2);
	// 		}
	// 		// distance_sum_DIM = sqrt(distance_sum_DIM);
	// 		dis[whichPt][count_k] = distance_sum_DIM;
	// 	}
	// }

	// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 	dis_min = FLT_MAX;
	// 	for (size_t count_k = 0; count_k < K_value; count_k++){
	// 		if (dis_min > dis[whichPt][count_k] ){
	// 			dis_min = dis[whichPt][count_k];
	// 			data.group[whichPt] = count_k;
	// 		}
	// 	}
	// }

	// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 	cout << " point :: "<< whichPt ;
	// 	cout << endl  << "dis " ;
	// 	for (size_t i = 0; i < data.K; i++){
	// 		cout <<  dis[whichPt][i] << "\t" ;
	// 	}
	// 	cout <<'\t' <<  "k:  " << data.group[whichPt];
	// 	cout << endl  << "current " ;
	// 	for (size_t i = 0; i < data.K; i++){
	// 		cout << " "<< dis_check[whichPt * data.K + i] << '\t' ;
	// 	}
	// 	cout << '\t'  <<  "k: " <<  group_check[whichPt]<< endl ;
	// }

	// for (size_t i = 0; i < data.K; i++){
	// 	cout << "currentCentroids" << "NO:K" << "\(" << i << ")" ;
	// 	for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
	// 		cout << '\t'<<data.currentCentroids.at(i+whichDim*data.K);
	// 	}
	// 	cout << endl;
	// }
	//! Check end

}


void UpdateCentroids(myData& data) {
	int which_uTAA = 0;// 0
	int which_uTAB = 0;// 0
	int which_uTB = 0;// 0

	std::map<int, const char*> uTAA {
		{0,"sumCentroidA"}
	};
	std::map<int, const char*> uTAB {
		{0,"sumCentroidB"},
		{1,"sumCentroidB1"},
		{2,"sumCentroidB2"}
	};
	std::map<int, const char*> uTB {
		{0,"avcgCentroid"}
	};

	// * conut buffer -> conut_dev
	static cl::KernelFunctor<const cl::Buffer&, cl::Buffer&, cl_uint, cl_uint> sumCentroidKernelA(data.prg, uTAA[which_uTAA]);

	static cl::KernelFunctor<const cl::Buffer&, cl::Buffer&, cl::Buffer&,
								cl_uint, cl_uint, cl_uint> sumCentroidKernelB(data.prg, uTAB[which_uTAB]);

	static cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl_int, cl_int> avcgCentroidKernel(data.prg, uTB[which_uTB]);

	// ! 老師寫的方法中跑這個function 一開使就swap使得下一個不用去等於，所以更新的與我們之前寫的不相同所以為了符合此要求，自己寫的版本也進行更新
	// * prepare buffer
	static auto groupCount = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K * sizeof(float));
	swap(data.dev_currentCentroids, data.dev_oldCentroids); // swap my buffer
	data.queue.enqueueFillBuffer(data.dev_currentCentroids, 0, 0,data.DIM*data.K*sizeof(float));
	data.queue.enqueueFillBuffer(groupCount, 0, 0, data.K*sizeof(cl_int));
	// * setup work items (global_ , local_)
	auto config2D = cl::EnqueueArgs(data.queue,{(data.N+data.D2_1-1)/data.D2_1*data.D2_1,(data.K+data.D2_2-1)/data.D2_2*data.D2_2},{data.D2_1,data.D2_2}); 
	auto config1D = cl::EnqueueArgs(data.queue,(data.N+data.D1-1)/data.D1*data.D1,data.D1);

	// * eneueue

	// static stopWatch timer[3];
	// timer[0].start();
	sumCentroidKernelA(config1D,data.dev_group,groupCount,data.N, data.K);
	// clFinish(data.queue11);
	// timer[0].stop();
	// timer[1].start();
	sumCentroidKernelB(config2D,data.dev_group,data.dev_currentCentroids,data.dev_pts, data.N, data.K, data.DIM);
	// clFinish(data.queue11);
	// timer[1].stop();
	// timer[2].start();
	avcgCentroidKernel(config2D,data.dev_currentCentroids,groupCount,data.DIM, data.K);
	// clFinish(data.queue11);
	// timer[2].stop();


	// std::cout   << timer[0].elapsedTime() << ", "
				// << timer[1].elapsedTime() << ", "
				// << timer[2].elapsedTime() << endl;



	clFinish(data.queue11);
	// //!Check begin
	// const size_t N_value = data.N;
	// const size_t K_value = data.K;
	// const size_t DIM_value = data.DIM;
	// vector<int> count(data.K);
	// data.queue.enqueueReadBuffer(groupCount,CL_TRUE, 0,data.K*sizeof(cl_int),count.data());
    // data.queue.enqueueReadBuffer(data.dev_group, CL_TRUE, 0, data.N*sizeof(float), data.group.data());
    // data.queue.enqueueReadBuffer(data.dev_currentCentroids, CL_TRUE, 0, data.K*data.DIM*sizeof(float), data.currentCentroids.data());

	// for (size_t i = 0; i < K_value; i++){
	// 	cout << "check_count ," << count[i] << endl;
	// }

	// int count_Check[K_value] = {0};
	// for (size_t count_k = 0; count_k < K_value; count_k++){
	// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 		if (data.group[whichPt] == count_k){
	// 		count_Check[count_k]++;
	// 	}
	// }}

	// for (size_t count_k = 0; count_k < data.K; count_k++)
	// {
	// 	cout << "UpdateCentroids currentCentroids" << "NO:K" << "\(" << count_k << ")"<<'\t'<<'\"' << count_Check[count_k]<< "\"";
	// for (size_t whichDim = 0; whichDim < data.DIM; whichDim++)
	// {
	// 	cout << '\t'<<data.currentCentroids.at(count_k+whichDim*data.K);
	// }
	// 	cout << endl;
	// }
	// 	cout << endl;
	//! Check end

}

bool HasConverged(myData& data, const float tolerance) {
	// * setup work items (global_ , local_)
	auto config2D = cl::EnqueueArgs(data.queue,{(data.N+data.D2_1-1)/data.D2_1*data.D2_1,(data.K+data.D2_2-1)/data.D2_2*data.D2_2},{data.D2_1,data.D2_2}); 
	auto config1D = cl::EnqueueArgs(data.queue,(data.N+data.D1-1)/data.D1*data.D1,data.D1);


	static const float tolerance_ = tolerance * tolerance;
	// * kernel fucntion
	static cl::KernelFunctor<const cl::Buffer&,const  cl::Buffer&, cl::Buffer&,
			 cl_uint, cl_uint, cl_float> hasConvergedKernel(data.prg, "hasConverged");

	// * prepare buffer
	int resultHost = 0;
	static cl::Buffer result(data.ctx, CL_MEM_READ_WRITE,sizeof(cl_int));
	data.queue.enqueueFillBuffer(result, 0, 0, sizeof(cl_int));
	

	// * enqueue
	hasConvergedKernel(config1D, data.dev_currentCentroids, data.dev_oldCentroids, result, data.DIM, data.K, tolerance_);

	// * enqueue ReadBuffer
	data.queue.enqueueReadBuffer(result, CL_TRUE, 0, sizeof(cl_int), &resultHost);

	// !check begin
	// data.queue.enqueueReadBuffer(data.dev_currentCentroids, CL_TRUE, 0,data.K*data.DIM*sizeof(float), data.currentCentroids.data());
    // data.queue.enqueueReadBuffer(data.dev_group, CL_TRUE, 0, data.N*sizeof(float), data.group.data());
	// 	const size_t K_value = data.K;
	// 	const size_t N_value = data.N;
	// 	const size_t DIM_value = data.DIM;
	// 	float count_Check[K_value] = {0.};
	// 	for (size_t count_k = 0; count_k < K_value; count_k++){
	// 	for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 			if (data.group[whichPt] == count_k){
	// 					count_Check[count_k]++;
	// 		}
	// 	}}
	
	// 	for (size_t count_k = 0; count_k < data.K; count_k++)
	// 	{
	// 		cout << "U currentCentroids" << "NO:K" << "\(" << count_k << ")"<<'\t'<<'\"' << count_Check[count_k]<< "\"";
	// 		for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
	// 			cout << '\t'<<data.currentCentroids.at(count_k+whichDim*data.K);
	// 		}
	// 		cout << endl;
	// 	}
	// !check end
	return !resultHost;
}

