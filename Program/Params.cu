#include "Params.h"
#include <cuda_runtime.h>
#include "LocalSearch_cuda_headers.cuh"

// The universal constructor for both executable and shared library
// When the executable is run from the commandline,
// it will first generate an CVRPLIB instance from .vrp file, then supply necessary information.
Params::Params(
	const std::vector<double> &x_coords,
	const std::vector<double> &y_coords,
	const std::vector<std::vector<double>> &dist_mtx,
	const std::vector<double> &service_time,
	const std::vector<double> &demands,
	double vehicleCapacity,
	double durationLimit,
	int nbVeh,
	bool isDurationConstraint,
	bool verbose,
	const AlgorithmParameters &ap)
	: ap(ap), isDurationConstraint(isDurationConstraint), nbVehicles(nbVeh), durationLimit(durationLimit),
	  vehicleCapacity(vehicleCapacity), timeCost(dist_mtx), verbose(verbose)
{
	// This marks the starting time of the algorithm
	startTime = clock();

	nbClients = (int)demands.size() - 1; // Need to substract the depot from the number of nodes
	totalDemand = 0.;
	maxDemand = 0.;

	// Initialize RNG
	ran.seed(ap.seed);

	// check if valid coordinates are provided
	areCoordinatesProvided = (demands.size() == x_coords.size()) && (demands.size() == y_coords.size());

	cli = std::vector<Client>(nbClients + 1);
	for (int i = 0; i <= nbClients; i++)
	{
		// If useSwapStar==false, x_coords and y_coords may be empty.
		if (ap.useSwapStar == 1 && areCoordinatesProvided)
		{
			cli[i].coordX = x_coords[i];
			cli[i].coordY = y_coords[i];
			cli[i].polarAngle = CircleSector::positive_mod(
				32768. * atan2(cli[i].coordY - cli[0].coordY, cli[i].coordX - cli[0].coordX) / PI);
		}
		else
		{
			cli[i].coordX = 0.0;
			cli[i].coordY = 0.0;
			cli[i].polarAngle = 0.0;
		}

		cli[i].serviceDuration = service_time[i];
		cli[i].demand = demands[i];
		if (cli[i].demand > maxDemand)
			maxDemand = cli[i].demand;
		totalDemand += cli[i].demand;
	}

	if (verbose && ap.useSwapStar == 1 && !areCoordinatesProvided)
		std::cout << "----- NO COORDINATES HAVE BEEN PROVIDED, SWAP* NEIGHBORHOOD WILL BE DEACTIVATED BY DEFAULT" << std::endl;

	// Default initialization if the number of vehicles has not been provided by the user
	if (nbVehicles == INT_MAX)
	{
		nbVehicles = (int)std::ceil(1.3 * totalDemand / vehicleCapacity) + 3; // Safety margin: 30% + 3 more vehicles than the trivial bin packing LB
		if (verbose)
			std::cout << "----- FLEET SIZE WAS NOT SPECIFIED: DEFAULT INITIALIZATION TO " << nbVehicles << " VEHICLES" << std::endl;
	}
	else
	{
		if (verbose)
			std::cout << "----- FLEET SIZE SPECIFIED: SET TO " << nbVehicles << " VEHICLES" << std::endl;
	}

	// Calculation of the maximum distance
	maxDist = 0.;
	for (int i = 0; i <= nbClients; i++)
		for (int j = 0; j <= nbClients; j++)
			if (timeCost[i][j] > maxDist)
				maxDist = timeCost[i][j];

	// Calculation of the correlated vertices for each customer (for the granular restriction)
	correlatedVertices = std::vector<std::vector<int>>(nbClients + 1);
	std::vector<std::set<int>> setCorrelatedVertices = std::vector<std::set<int>>(nbClients + 1);
	std::vector<std::pair<double, int>> orderProximity;
	for (int i = 1; i <= nbClients; i++)
	{
		orderProximity.clear();
		for (int j = 1; j <= nbClients; j++)
			if (i != j)
				orderProximity.emplace_back(timeCost[i][j], j);
		std::sort(orderProximity.begin(), orderProximity.end());

		for (int j = 0; j < std::min<int>(ap.nbGranular, nbClients - 1); j++)
		{
			// If i is correlated with j, then j should be correlated with i
			setCorrelatedVertices[i].insert(orderProximity[j].second);
			setCorrelatedVertices[orderProximity[j].second].insert(i);
		}
	}

	// Filling the vector of correlated vertices
	for (int i = 1; i <= nbClients; i++)
		for (int x : setCorrelatedVertices[i])
			correlatedVertices[i].push_back(x);

	// Safeguards to avoid possible numerical instability in case of instances containing arbitrarily small or large numerical values
	if (maxDist < 0.1 || maxDist > 100000)
		throw std::string(
			"The distances are of very small or large scale. This could impact numerical stability. Please rescale the dataset and run again.");
	if (maxDemand < 0.1 || maxDemand > 100000)
		throw std::string(
			"The demand quantities are of very small or large scale. This could impact numerical stability. Please rescale the dataset and run again.");
	if (nbVehicles < std::ceil(totalDemand / vehicleCapacity))
		throw std::string("Fleet size is insufficient to service the considered clients.");

	// A reasonable scale for the initial values of the penalties
	penaltyDuration = 1;
	penaltyCapacity = std::max<double>(0.1, std::min<double>(1000., maxDist / maxDemand));

	// Copy data to CUDA global variables
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	int n = nbClients + 1;
	*d_service_duration = nullptr;
	*d_demand = nullptr;
	*d_timeCost = nullptr;
	*d_correlated_customers = nullptr;
	*d_numberof_correlated = nullptr;
	size_t bytes_n = n * sizeof(double);
	size_t bytes_nn = (size_t)n * (size_t)n * sizeof(double);
	cudaMalloc(&d_service_duration, bytes_n);
	cudaMalloc(&d_demand, bytes_n);
	cudaMalloc(&d_timeCost, bytes_nn);
	cudaMalloc(&d_numberof_correlated, n * sizeof(int));
	std::vector<double> h_timeCost_flat(n * n);
	std::vector<int> temp_numberof_correlated(n, 0);
	temp_numberof_correlated[0] = correlatedVertices[0].size();
	for (int i = 1; i < n; i++)
	{
		temp_numberof_correlated[i] = correlatedVertices[i].size() + temp_numberof_correlated[i - 1];
	}
	std::vector<int> temp_correlated;

	for (int i = 0; i < n; i++)
	{
		temp_correlated.insert(temp_correlated.end(), correlatedVertices[i].begin(), correlatedVertices[i].end());
	}
	cudaMalloc(&d_correlated_customers, temp_correlated.size() * sizeof(int));

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
			h_timeCost_flat[i * n + j] = timeCost[i][j];
	}
	cudaMemcpyAsync(d_correlated_customers, temp_correlated.data(), temp_correlated.size() * sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_numberof_correlated, temp_numberof_correlated.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_service_duration, service_time.data(), bytes_n, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_demand, demands.data(), bytes_n, cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_timeCost, h_timeCost_flat.data(), bytes_nn, cudaMemcpyHostToDevice, stream3);
	cudaDeviceSynchronize();
	initializing_variable_CUDA<<<1, 1>>>(n, d_service_duration, d_demand, d_timeCost, d_correlated_customers, d_numberof_correlated, durationLimit, vehicleCapacity);

	if (verbose)
		std::cout << "----- INSTANCE SUCCESSFULLY LOADED WITH " << n << " CLIENTS AND " << nbVehicles << " VEHICLES" << std::endl;
}

Params::~Params()
{
	cudaFree(d_service_duration);
	cudaFree(d_demand);
	cudaFree(d_timeCost);
	cudaFree(d_correlated_customers);
	cudaFree(d_numberof_correlated);
}