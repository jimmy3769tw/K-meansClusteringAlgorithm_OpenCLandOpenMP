#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cstring>

#include "../inc/kmeans.hpp"
#include "../inc/stopWatch.hpp"

int main(int argc, char **argv) {
	// get all command line arguments
	if( argc != 4 ) {
		std::cerr << "\n" << argv[0] << " [noClusters] [tolerance] [CSV data filename]";
		return 255;
	}
	size_t noClusters	= atoi( argv[1] );
	const float tolerance = atof(argv[2]); 

	// initialize data 
	myData data; 
	if(! readCSV(data, argv[3])) {
		std::cerr << "\nFailed to read CSV file, abort!"; 
		return 255; 
	}

	// 1. randomly choose k centroids
	InitializeCentroid (data, noClusters);

	stopWatch timer[4];
	timer[0].start();
	size_t it;
	for(it=0;it<3000;++it) {
		// #1, 2. for each point, find which centroid that it is closest to
		timer[1].start();
		AssignGroups(data);
		timer[1].stop(); 
		// #2, 3. for each identified cluster, find the mean centroid
		timer[2].start();
		UpdateCentroids(data);
		timer[2].stop(); 
		// 4. go back to 2 until convergence
		timer[3].start();

		if(HasConverged(data, tolerance)) break;
			

		timer[3].stop();

		// cout << "\n" << it << ": " << correction;
	}
	timer[0].stop();


	std::cout 	<< argv[1] << ", " 
				<< argv[2] << ", " 
				<< it << "," 
				<< timer[0].elapsedTime() << ","
				<< timer[1].elapsedTime() << ","
				<< timer[2].elapsedTime() << ","
				<< timer[3].elapsedTime() << endl;
	
	// std::cout << "1D" << (data.N+data.D1-1)/data.D1*data.D1 << ", " << data.D1 << ", " << ((data.N+data.D1-1)/data.D1*data.D1) /data.D1 << endl;
	
	return 0;
}

