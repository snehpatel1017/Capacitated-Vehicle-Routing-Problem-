#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
#include "InstanceCVRPLIB.h"
using namespace std;

int main(int argc, char *argv[])
{
	try
	{
		// Reading the arguments of the program
		auto start = std::chrono::high_resolution_clock::now();
		CommandLine commandline(argc, argv);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		cout << "CommandLine Obj generated : " << elapsed_seconds.count() << "\n";

		// Print all algorithm parameter values
		if (commandline.verbose)
			print_algorithm_parameters(commandline.ap);

		// Reading the data file and initializing some data structures
		if (commandline.verbose)
			std::cout << "----- READING INSTANCE: " << commandline.pathInstance << std::endl; // this is printing the intance filename/filepath.

		start = std::chrono::high_resolution_clock::now();
		InstanceCVRPLIB cvrp(commandline.pathInstance, commandline.isRoundingInteger); // giving instance filepath and isrounding flag only
		end = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end - start;
		cout << "InstanceCVRPLIB Obj generated : " << elapsed_seconds.count() << "\n";
		Params params(cvrp.x_coords, cvrp.y_coords, cvrp.dist_mtx, cvrp.service_time, cvrp.demands,
					  cvrp.vehicleCapacity, cvrp.durationLimit, commandline.nbVeh, cvrp.isDurationConstraint, commandline.verbose, commandline.ap);

		// Running HGS
		Genetic solver(params);
		solver.run();

		// Exporting the best solution
		if (solver.population.getBestFound() != NULL)
		{
			if (params.verbose)
				std::cout << "----- WRITING BEST SOLUTION IN : " << commandline.pathSolution << std::endl;
			solver.population.exportCVRPLibFormat(*solver.population.getBestFound(), commandline.pathSolution);
			solver.population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance);
		}
	}
	catch (const string &e)
	{
		std::cout << "EXCEPTION | " << e << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << "EXCEPTION | " << e.what() << std::endl;
	}
	return 0;
}
