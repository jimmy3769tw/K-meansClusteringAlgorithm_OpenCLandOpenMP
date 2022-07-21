#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
using std::cerr;
using std::cout; 
using std::endl;

int main(int argc, char **argv) {
	if (argc != 4) {
		cerr << argv[0] << " [nClusters] [nDimensions] [nPts]" << endl;
		return 255;
	}
	auto nCluster = atoi(argv[1]); 
	auto nDim = atoi(argv[2]);
	auto nPts = atoi(argv[3]); 

	const auto range = 100.0f, variance = 2.0f; 
	std::random_device rd;
	std::default_random_engine gen(rd());

	// random number generator (RNG) for the centroids
	std::uniform_real_distribution<float> rngCentroid(-range, range);

	for (auto i = 0; i < nCluster; i++) {
		// create a centroid
		auto centroid = std::vector<float>(nDim); 
		std::generate(centroid.begin(), centroid.end(), [&]() {
			return rngCentroid(gen);
		});
		
		// prepare cluster generators
		auto generators = std::vector<std::normal_distribution<float>>(); 
		generators.reserve(nDim); 
		for (auto x : centroid) {
			generators.push_back(std::normal_distribution<float>(x, variance));
		}

		// output the current cluster
		for (auto j = 0; j < nPts; ++j) {
			for (auto& rngCluster : generators) {
				cout << rngCluster(gen) << ", ";
			}
			cout << "\n"; 
		}
	}
	return 0;
}
