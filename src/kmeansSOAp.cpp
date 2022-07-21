#include "../inc/kmeans.hpp"
#include <iostream>
#include <fstream>

#include <vector>
#include <unordered_set>
#include <omp.h>
#include <random>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cfloat>

using namespace std;

/*
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/

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
	// !information
	// cout << "noAttributes is the number of " << noAttributes <<endl;
	// cout << "noLines is the number of " << noLines << endl;
	// cout << "noLines*noAttributes is equal to "  << noLines*noAttributes << endl;
	inp.clear();
	inp.seekg(0, inp.beg);
	data.pts.resize(noLines*noAttributes);
	#pragma omp parallel for schedule(static)
	for (size_t whichPt = 0; whichPt < noLines; ++whichPt)
	{
		for (size_t whichDim = 0; whichDim < noAttributes;++whichDim) {
			data.pts[whichPt*noAttributes+whichDim] = 0.0;
		}
	}

	for (size_t whichPt = 0; whichPt < noLines; ++whichPt) {
		inp.getline(buf, 4096);
		auto it = buf;
		for (size_t whichDim = 0; whichDim < noAttributes;++whichDim) {
			auto x = atof(it);
			data.pts[whichPt+data.N*whichDim] = x; //*soa
			while(*it!=',') ++it;
			it++;
		}
	}

	//! check data
	// for (size_t i = 0; i < noLines*noAttributes; i++ ) 
	// 		cout << i << ","<< data.pts.at(i)<< endl ;
	// cout << "Finish reading datas according to AOS: coordinates of the same point are stored continuously." << endl;
	// Check data

	inp.close();
	////ex 
	// for (size_t whichPt = 0; whichPt < noLines; ++whichPt)
	// for (size_t whichDim = 0; whichDim < noAttributes;++whichDim) {
	// {
	// cout << data.pts.at(whichPt*noAttributes+whichDim)<< endl;
	// }}

	return true;
}


// AOS version
// array of structs x1, y1, z1, x2, y2, z2, ...
void InitializeCentroid(myData& data, size_t noClusters) {

	data.K = noClusters;
	data.currentCentroids.resize(data.K*data.DIM);
	data.oldCentroids.resize(data.K*data.DIM);
	data.group.resize(data.N);
	// ! specific I.C.
	

	for (size_t i = 0; i < data.K; i++){
		size_t whichPt = (data.N / data.K) * (i);
		for (size_t whichDim = 0; whichDim < data.DIM; whichDim++)
		{
			data.currentCentroids[i+whichDim*data.K] = data.pts[whichPt+data.N*whichDim];
		}
	}

	// !Check begin
	// for (size_t i = 0; i < data.K; i++){
	// 		cout << "I.C. oldCentroids" << "NO:K" << "\(" << i << ")";
	// 	for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
	// 		cout << '\t'<<data.currentCentroids[i+whichDim*data.K];
	// 	}
	// 	cout << endl;
	// }
	//! Check end
}

void AssignGroups(myData& data) {
	const size_t N_value = data.N;
	const size_t K_value = data.K;
	const size_t DIM_value = data.DIM;
	float distance_sum_DIM;

	// cout << "DIM_value "<< DIM_value << endl;
	// cout << "N_value "<< N_value << endl;
	
	///判斷群新與之間 /// count_k 群
	float dis_min = 0.;

	//  Select k member (currentCentroids)
	/// Seting data.group
	float dis[N_value][K_value]= {0.};

	#pragma omp parallel for schedule(static) default(none) firstprivate(distance_sum_DIM,DIM_value,K_value,N_value) shared(data,dis)
	for (size_t whichPt = 0; whichPt < N_value; whichPt++)
	{
		for (size_t count_k = 0; count_k < K_value; count_k++)
		{
			distance_sum_DIM = 0;
			for (size_t whichDim = 0; whichDim < DIM_value ; whichDim++)
			{
				distance_sum_DIM += pow(data.currentCentroids[count_k+whichDim*data.K]- data.pts[whichPt+data.N*whichDim],2);
			}
			// distance_sum_DIM = sqrt(distance_sum_DIM);
			dis[whichPt][count_k] = distance_sum_DIM; //first polo
		}
	}


	#pragma omp parallel for schedule(static) default(none) firstprivate(N_value,K_value,dis_min) shared(data,dis)
	for (size_t whichPt = 0; whichPt < N_value; whichPt++)
	{
		dis_min = FLT_MAX;
		for (size_t count_k = 0; count_k < K_value; count_k++)
		{
			if (dis_min > dis[whichPt][count_k] )
			{
				dis_min = dis[whichPt][count_k];
				data.group[whichPt] = count_k;
			}
		}
	}

	//! Check begin
	// for (size_t whichPt = 0; whichPt < N_value; whichPt++)
	// {
	// 		cout << " point :: "<< whichPt << '\t';
	// for (size_t count_k = 0; count_k < data.K; count_k++)
	// 	{
	// 		cout << " "<< dis[whichPt][count_k] << '\t' ;
	// 	}
	// 	cout <<'\t' <<  "k ::  " << data.group[whichPt] << endl ;
	// }

	// for (size_t count_k = 0; count_k < data.K; count_k++)
	// {
	// 	cout << "currentCentroids" << "NO:K" << "\(" << count_k << ")" ;
	// 	for (size_t whichDim = 0; whichDim < data.DIM; whichDim++)
	// 	{
	// 		cout << '\t'<<data.currentCentroids.at(count_k+whichDim*data.K);
	// 	}
	// 	cout << endl;
	// }
	// ! Check end

}

void UpdateCentroids(myData& data) {
	const size_t K_value = data.K;
	const size_t DIM_value = data.DIM;
	const size_t N_value = data.N;

	// cout << "UpdateCentroids-----------------------------------------------------------------------------" << endl;
	//  For oldCentroids
	

	swap(data.oldCentroids, data.currentCentroids);
	float	center_tem[DIM_value*K_value] = {0.};
	int		count[K_value] = {0};


	#pragma omp parallel default(none) firstprivate(K_value,N_value,DIM_value) shared(data) \
	reduction(+:count[:]) reduction(+:center_tem[:])
	{
		for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
			#pragma omp for schedule(static) nowait
			for (size_t whichPt = 0; whichPt < N_value; whichPt++){
				center_tem[whichDim*K_value + data.group[whichPt]]+=data.pts[whichPt+data.N*whichDim];
				count[data.group[whichPt]]++;
			}
		}
	}
	
	// 最後再去跑一個沒有平行的< 這邊還有令一個寫法 就是前面 counter_tem[whichDim*K_value + count_k]++; 
	// 改成 counter_tem[count_k]++ 但是最後面要 counter_tem[count_k] /= data.DIM; 可以節省記憶體空間
	for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
		for (size_t i = 0; i < K_value; i++){
			data.currentCentroids[i+whichDim*data.K]= center_tem[whichDim*K_value + i]*DIM_value/float(count[i]);
		}
	}

	// ! Check begin
	// float count_Check[K_value] = {0.};
	// for (size_t count_k = 0; count_k < K_value; count_k++){
	// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
	// 		if (data.group[whichPt] == count_k){
	// 				count_Check[count_k]++;
	// 	}
	// }}
	// for (size_t count_k = 0; count_k < data.K; count_k++)
	// {
	// 	cout << "Up currentCentroids" << "NO:K" << "\(" << count_k << ")"<<'\t'<<'\"' << count_Check[count_k]<< "\"";
	// for (size_t whichDim = 0; whichDim < data.DIM; whichDim++)
	// {
	// 	cout << '\t'<<data.currentCentroids.at(count_k+whichDim*data.K);
	// }
	// 	cout << endl;
	// }
	// 	cout << endl;
	// ! cout << endl;

}

bool HasConverged(myData& data, const float tolerance) {
	const size_t K_value = data.K;
	const size_t N_value = data.N;
	const size_t DIM_value = data.DIM;
	float tolerance_sum = {0.};

	
	for (size_t whichDim = 0 ; whichDim < DIM_value ; whichDim++){
		for (size_t i  = 0 ; i < K_value ; i++){	
			tolerance_sum += std::pow(data.oldCentroids[i+whichDim*data.K]
								 - data.currentCentroids[i+whichDim*data.K],2);
		}
	}

	if (tolerance*tolerance > tolerance_sum)
	{
		// cout << "CONVERGENCE!!!!!! " << endl;

		// // Information
	    // cout <<"omp_get_max_threads = " << omp_get_max_threads() << endl;
 		// cout <<"OpenMP: " << _OPENMP << endl;

		// // check begin
		// float count_Check[K_value] = {0.};
		// for (size_t whichPt = 0; whichPt < N_value; whichPt++){
		// 	count_Check[data.group[whichPt]]++;
		// }
		// for (size_t i = 0; i < data.K; i++){
		// 	cout << "(Has) currentCentroids" << "NO:K" << "\(" << i << ")"<<'\t'<<'\"' << count_Check[i]<< "\"";
		// 	for (size_t whichDim = 0; whichDim < data.DIM; whichDim++){
		// 		cout << '\t'<<data.currentCentroids.at(i+whichDim*data.K);
		// 	}
		// 	cout << endl;
		// }
		// check end
		return true;
	}
	else
	{
		return false;
	}
}