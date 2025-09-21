#include "Genetic.h"
#include <chrono>

void Genetic::run()
{
	/* INITIAL POPULATION */
	auto start = std::chrono::high_resolution_clock::now();
	population.generatePopulation();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "GeneratePopulation Completed : " << elapsed_seconds.count() << "\n";

	int nbIter;
	int nbIterNonProd = 1;
	if (params.verbose)
		std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
	double corssoverOX_time = 0, localSearch_time = 0, addIndividual_time = 0;
	for (nbIter = 0; nbIterNonProd <= params.ap.nbIter && (params.ap.timeLimit == 0 || (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC < params.ap.timeLimit); nbIter++)
	{
		/* SELECTION AND CROSSOVER */
		start = std::chrono::high_resolution_clock::now();
		crossoverOX(offspring, population.getBinaryTournament(), population.getBinaryTournament());
		end = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end - start;
		corssoverOX_time += elapsed_seconds.count();

		/* LOCAL SEARCH */
		start = std::chrono::high_resolution_clock::now();
		localSearch.run(offspring, params.penaltyCapacity, params.penaltyDuration);
		end = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end - start;
		localSearch_time += elapsed_seconds.count();
		start = std::chrono::high_resolution_clock::now();
		bool isNewBest = population.addIndividual(offspring, true);
		end = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end - start;
		addIndividual_time += elapsed_seconds.count();
		if (!offspring.eval.isFeasible && params.ran() % 2 == 0) // Repair half of the solutions in case of infeasibility
		{
			start = std::chrono::high_resolution_clock::now();
			localSearch.run(offspring, params.penaltyCapacity * 10., params.penaltyDuration * 10.);
			end = std::chrono::high_resolution_clock::now();
			elapsed_seconds = end - start;
			localSearch_time += elapsed_seconds.count();
			if (offspring.eval.isFeasible)
			{
				start = std::chrono::high_resolution_clock::now();
				isNewBest = (population.addIndividual(offspring, false) || isNewBest);
				end = std::chrono::high_resolution_clock::now();
				elapsed_seconds = end - start;
				addIndividual_time += elapsed_seconds.count();
			}
		}

		/* TRACKING THE NUMBER OF ITERATIONS SINCE LAST SOLUTION IMPROVEMENT */
		if (isNewBest)
			nbIterNonProd = 1;
		else
			nbIterNonProd++;

		/* DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES */
		if (nbIter % params.ap.nbIterPenaltyManagement == 0)
			population.managePenalties();
		if (nbIter % params.ap.nbIterTraces == 0)
			population.printState(nbIter, nbIterNonProd);

		/* FOR TESTS INVOLVING SUCCESSIVE RUNS UNTIL A TIME LIMIT: WE RESET THE ALGORITHM/POPULATION EACH TIME maxIterNonProd IS ATTAINED*/
		if (params.ap.timeLimit != 0 && nbIterNonProd == params.ap.nbIter)
		{
			population.restart();
			nbIterNonProd = 1;
		}
	}
	std::cout << "crossoverOx time : " << corssoverOX_time << "\n";
	std::cout << "localSearch time : " << localSearch_time << "\n";
	std::cout << "addIndividual time : " << addIndividual_time << "\n";
	if (params.verbose)
		std::cout << "----- GENETIC ALGORITHM FINISHED AFTER " << nbIter << " ITERATIONS. TIME SPENT: " << (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC << std::endl;
}

void Genetic::crossoverOX(Individual &result, const Individual &parent1, const Individual &parent2)
{
	// Frequency table to track the customers which have been already inserted
	std::vector<bool> freqClient = std::vector<bool>(params.nbClients + 1, false);

	// Picking the beginning and end of the crossover zone
	std::uniform_int_distribution<> distr(0, params.nbClients - 1);
	int start = distr(params.ran);
	int end = distr(params.ran);

	// Avoid that start and end coincide by accident
	while (end == start)
		end = distr(params.ran);

	// Copy from start to end
	int j = start;
	while (j % params.nbClients != (end + 1) % params.nbClients)
	{
		result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
		freqClient[result.chromT[j % params.nbClients]] = true;
		j++;
	}

	// Fill the remaining elements in the order given by the second parent
	for (int i = 1; i <= params.nbClients; i++)
	{
		int temp = parent2.chromT[(end + i) % params.nbClients];
		if (freqClient[temp] == false)
		{
			result.chromT[j % params.nbClients] = temp;
			j++;
		}
	}

	// Complete the individual with the Split algorithm
	split.generalSplit(result, parent1.eval.nbRoutes);
}

Genetic::Genetic(Params &params) : params(params),
								   split(params),
								   localSearch(params),
								   population(params, this->split, this->localSearch),
								   offspring(params) {}
