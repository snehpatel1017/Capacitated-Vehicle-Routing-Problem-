#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include "LocalSearch_cuda_headers.cuh"
#define DELTA_INF_CUDA 1e300
#define MY_EPSILON_CUDA 0.00001

__device__ int customers_CUDA;
__device__ double *service_duration_CUDA; // n
__device__ double *demand_CUDA;
__device__ double *timeCost_CUDA; // n*n
__device__ double durationLimit_CUDA;
__device__ double vehicleCapacity_CUDA;
__device__ double penaltyCapacityLS_CUDA, penaltyDurationLS_CUDA;

__device__ double global_best_cost = 0;

// lock (0 = free, 1 = locked)
__device__ unsigned int global_best_lock;

// initialize by params constructor
__device__ int *correlated_customers_CUDA;
__device__ int *numberof_correlated_CUDA;

// Initialized by LocalSearch_CUDA

__device__ Node_CUDA *clients_CUDA;
__device__ Route_CUDA *routes_CUDA;
__device__ Node_CUDA *depots_CUDA;
__device__ Node_CUDA *depotsEnd_CUDA;
__device__ int nbVehicles_CUDA;
__device__ int nbMoves_CUDA;

// Initialized by every run of LocalSearch
__device__ int *chrom_T;
__device__ int *route_size;

__device__ double penaltyExcessDuration_CUDA(double myDuration_CUDA)
{
    return fmax(0.0, myDuration_CUDA - durationLimit_CUDA) * penaltyDurationLS_CUDA;
}
__device__ double penaltyExcessLoad_CUDA(double myLoad_CUDA)
{
    return fmax(0.0, myLoad_CUDA - vehicleCapacity_CUDA) * penaltyCapacityLS_CUDA;
}

__device__ double move1_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, int nodeUPrevIndex_CUDA, int nodeXIndex_CUDA, int nodeUIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double serviceU_CUDA, bool intraRouteMove_CUDA)
{

    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA - serviceU_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load - loadU_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA + serviceU_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeUIndex_CUDA == nodeYIndex_CUDA)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move2_CUDA(Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, int nodeUPrevIndex_CUDA, int nodeUIndex_CUDA, int nodeVIndex_CUDA, int nodeXNextIndex_CUDA, int nodeXIndex_CUDA, int nodeYIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA)
{
    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - serviceU_CUDA - serviceX_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load - loadU_CUDA - loadX_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + serviceU_CUDA + serviceX_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA + loadX_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeU_CUDA == nodeY_CUDA || nodeV_CUDA == nodeX_CUDA || nodeX_CUDA->isDepot)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move3_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, int nodeUPrevIndex_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeXNextIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA)
{
    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA - serviceU_CUDA - serviceX_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load - loadU_CUDA - loadX_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA + serviceU_CUDA + serviceX_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA + loadX_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeU_CUDA == nodeY_CUDA || nodeX_CUDA == nodeV_CUDA || nodeX_CUDA->isDepot)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move4_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, int nodeUPrevIndex_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, double loadU_CUDA, double loadV_CUDA, double serviceU_CUDA, double serviceV_CUDA, bool intraRouteMove_CUDA)
{
    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] + timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA + serviceV_CUDA - serviceU_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load + loadV_CUDA - loadU_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA - serviceV_CUDA + serviceU_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA - loadV_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeUIndex_CUDA == nodeVPrevIndex_CUDA || nodeUIndex_CUDA == nodeYIndex_CUDA)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move5_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, int nodeUPrevIndex_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeXNextIndex_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double loadV_CUDA, double serviceU_CUDA, double serviceX_CUDA, double serviceV_CUDA, bool intraRouteMove_CUDA)
{
    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] + timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + serviceV_CUDA - serviceU_CUDA - serviceX_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load + loadV_CUDA - loadU_CUDA - loadX_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - serviceV_CUDA + serviceU_CUDA + serviceX_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA + loadX_CUDA - loadV_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeU_CUDA == nodeV_CUDA->prev || nodeX_CUDA == nodeV_CUDA->prev || nodeU_CUDA == nodeY_CUDA || nodeX_CUDA->isDepot)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move6_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, int nodeUPrevIndex_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeXNextIndex_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double loadV_CUDA, double loadY_CUDA, double serviceU_CUDA, double serviceX_CUDA, double serviceV_CUDA, double serviceY_CUDA, bool intraRouteMove_CUDA)
{
    double costSuppU_CUDA = timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] + timeCost_CUDA[nodeYIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA] - timeCost_CUDA[nodeUPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] - timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeXNextIndex_CUDA];
    double costSuppV_CUDA = timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeUIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYNextIndex_CUDA] - timeCost_CUDA[nodeVPrevIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] - timeCost_CUDA[nodeYIndex_CUDA * customers_CUDA + nodeYNextIndex_CUDA];

    if (!intraRouteMove_CUDA)
    {
        // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if (costSuppU_CUDA + costSuppV_CUDA >= routeU_CUDA->penalty + routeV_CUDA->penalty)
            return DELTA_INF_CUDA;

        costSuppU_CUDA += penaltyExcessDuration_CUDA(routeU_CUDA->duration + costSuppU_CUDA - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + serviceV_CUDA + serviceY_CUDA - serviceU_CUDA - serviceX_CUDA) + penaltyExcessLoad_CUDA(routeU_CUDA->load + loadV_CUDA + loadY_CUDA - loadU_CUDA - loadX_CUDA) - routeU_CUDA->penalty;

        costSuppV_CUDA += penaltyExcessDuration_CUDA(routeV_CUDA->duration + costSuppV_CUDA + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - serviceV_CUDA - serviceY_CUDA + serviceU_CUDA + serviceX_CUDA) + penaltyExcessLoad_CUDA(routeV_CUDA->load + loadU_CUDA + loadX_CUDA - loadV_CUDA - loadY_CUDA) - routeV_CUDA->penalty;
    }

    if (costSuppU_CUDA + costSuppV_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeX_CUDA->isDepot || nodeY_CUDA->isDepot || nodeY_CUDA == nodeU_CUDA->prev || nodeU_CUDA == nodeY_CUDA || nodeX_CUDA == nodeV_CUDA || nodeV_CUDA == nodeX_CUDA->next)
        return DELTA_INF_CUDA;
    return costSuppU_CUDA + costSuppV_CUDA;
}

__device__ double move7_CUDA(Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA)
{
    if (nodeU_CUDA->position > nodeV_CUDA->position)
        return DELTA_INF_CUDA;

    double cost_CUDA = timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + nodeV_CUDA->cumulatedReversalDistance - nodeX_CUDA->cumulatedReversalDistance;

    if (cost_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    if (nodeU_CUDA->next == nodeV_CUDA)
        return DELTA_INF_CUDA;
    return cost_CUDA;
}

__device__ double move8_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, Node_CUDA *nodeU_CUDA, Node_CUDA *nodeX_CUDA, Node_CUDA *nodeV_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA)
{
    double cost_CUDA = timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeVIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + nodeV_CUDA->cumulatedReversalDistance + routeU_CUDA->reversalDistance - nodeX_CUDA->cumulatedReversalDistance - routeU_CUDA->penalty - routeV_CUDA->penalty;

    // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
    if (cost_CUDA >= 0)
        return DELTA_INF_CUDA;

    cost_CUDA += penaltyExcessDuration_CUDA(nodeU_CUDA->cumulatedTime + nodeV_CUDA->cumulatedTime + nodeV_CUDA->cumulatedReversalDistance + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeVIndex_CUDA]) + penaltyExcessDuration_CUDA(routeU_CUDA->duration - nodeU_CUDA->cumulatedTime - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + routeU_CUDA->reversalDistance - nodeX_CUDA->cumulatedReversalDistance + routeV_CUDA->duration - nodeV_CUDA->cumulatedTime - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + timeCost_CUDA[nodeXIndex_CUDA * customers_CUDA + nodeYIndex_CUDA]) + penaltyExcessLoad_CUDA(nodeU_CUDA->cumulatedLoad + nodeV_CUDA->cumulatedLoad) + penaltyExcessLoad_CUDA(routeU_CUDA->load + routeV_CUDA->load - nodeU_CUDA->cumulatedLoad - nodeV_CUDA->cumulatedLoad);

    if (cost_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    return cost_CUDA;
}

__device__ double move9_CUDA(Route_CUDA *routeU_CUDA, Route_CUDA *routeV_CUDA, Node_CUDA *nodeU_CUDA, Node_CUDA *nodeV_CUDA, int nodeUIndex_CUDA, int nodeXIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA)
{
    double cost_CUDA = timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] - routeU_CUDA->penalty - routeV_CUDA->penalty;

    // Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
    if (cost_CUDA >= 0)
        return DELTA_INF_CUDA;

    cost_CUDA += penaltyExcessDuration_CUDA(nodeU_CUDA->cumulatedTime + routeV_CUDA->duration - nodeV_CUDA->cumulatedTime - timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeYIndex_CUDA] + timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeYIndex_CUDA]) + penaltyExcessDuration_CUDA(routeU_CUDA->duration - nodeU_CUDA->cumulatedTime - timeCost_CUDA[nodeUIndex_CUDA * customers_CUDA + nodeXIndex_CUDA] + nodeV_CUDA->cumulatedTime + timeCost_CUDA[nodeVIndex_CUDA * customers_CUDA + nodeXIndex_CUDA]) + penaltyExcessLoad_CUDA(nodeU_CUDA->cumulatedLoad + routeV_CUDA->load - nodeV_CUDA->cumulatedLoad) + penaltyExcessLoad_CUDA(nodeV_CUDA->cumulatedLoad + routeU_CUDA->load - nodeU_CUDA->cumulatedLoad);

    if (cost_CUDA > -MY_EPSILON_CUDA)
        return DELTA_INF_CUDA;
    return cost_CUDA;
}

__global__ void kernel1_CUDA(int loopID, int emptyRouteIndex, unsigned int *global_best_v_node, unsigned int *global_best_u_node, unsigned int *global_best_move)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= 0 || tid > customers_CUDA)
        return;

    Node_CUDA *nodeU_CUDA = &clients_CUDA[tid];
    int lastTestRINodeU = nodeU_CUDA->whenLastTestedRI;
    nodeU_CUDA->whenLastTestedRI = nbMoves_CUDA;

    // setLocalVariablesRouteU
    Route_CUDA *routeU_CUDA = nodeU_CUDA->route;
    Node_CUDA *nodeX_CUDA = nodeU_CUDA->next;
    int nodeXNextIndex_CUDA = nodeX_CUDA->next->cour;
    int nodeUIndex_CUDA = nodeU_CUDA->cour;
    int nodeUPrevIndex_CUDA = nodeU_CUDA->prev->cour;
    int nodeXIndex_CUDA = nodeX_CUDA->cour;
    double loadU_CUDA = demand_CUDA[nodeUIndex_CUDA];
    double serviceU_CUDA = service_duration_CUDA[nodeUIndex_CUDA];
    double loadX_CUDA = demand_CUDA[nodeXIndex_CUDA];
    double serviceX_CUDA = service_duration_CUDA[nodeXIndex_CUDA];
    bool intraRouteMove_CUDA = false;

    int len = numberof_correlated_CUDA[tid];
    int start = 0;
    if (tid > 0)
    {
        len -= numberof_correlated_CUDA[tid - 1];
        start = numberof_correlated_CUDA[tid - 1];
    }
    double best_cost = 0;
    unsigned int best_move = 0;
    unsigned int best_v_node = 0;
    for (int i = 0; i < numberof_correlated_CUDA[tid]; i++)
    {
        Node_CUDA *nodeV_CUDA = &clients_CUDA[correlated_customers_CUDA[start + i]];

        // setLocalVariablesRouteV
        Route_CUDA *routeV_CUDA = nodeV_CUDA->route;
        Node_CUDA *nodeY_CUDA = nodeV_CUDA->next;
        int nodeYNextIndex_CUDA = nodeY_CUDA->next->cour;
        int nodeVIndex_CUDA = nodeV_CUDA->cour;
        int nodeVPrevIndex_CUDA = nodeV_CUDA->prev->cour;
        int nodeYIndex_CUDA = nodeY_CUDA->cour;
        double loadV_CUDA = demand_CUDA[nodeVIndex_CUDA];
        double serviceV_CUDA = service_duration_CUDA[nodeVIndex_CUDA];
        double loadY_CUDA = demand_CUDA[nodeYIndex_CUDA];
        double serviceY_CUDA = service_duration_CUDA[nodeYIndex_CUDA];
        intraRouteMove_CUDA = (routeU_CUDA == routeV_CUDA);
        double current_cost = move1_CUDA(routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeXIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, nodeYNextIndex_CUDA, loadU_CUDA, serviceU_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 1;
            best_v_node = nodeVIndex_CUDA;
        }
        current_cost = move2_CUDA(nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeXNextIndex_CUDA, nodeXIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 2;
            best_v_node = nodeVIndex_CUDA;
        }
        current_cost = move3_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeXNextIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 3;
            best_v_node = nodeVIndex_CUDA;
        }

        if (nodeUIndex_CUDA <= nodeVIndex_CUDA)
        {
            current_cost = move4_CUDA(routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVPrevIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadV_CUDA, serviceU_CUDA, serviceV_CUDA, intraRouteMove_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 4;
                best_v_node = nodeVIndex_CUDA;
            }
        }

        current_cost = move5_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeXNextIndex_CUDA, nodeVPrevIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, loadV_CUDA, serviceU_CUDA, serviceX_CUDA, serviceV_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 5;
            best_v_node = nodeVIndex_CUDA;
        }

        if (nodeUIndex_CUDA <= nodeVIndex_CUDA)
        {
            current_cost = move6_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeXNextIndex_CUDA, nodeVPrevIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, nodeYNextIndex_CUDA, loadU_CUDA, loadX_CUDA, loadV_CUDA, loadY_CUDA, serviceU_CUDA, serviceX_CUDA, serviceV_CUDA, serviceY_CUDA, intraRouteMove_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 6;
                best_v_node = nodeVIndex_CUDA;
            }
        }
        if (intraRouteMove_CUDA)
        {
            current_cost = move7_CUDA(nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 7;
                best_v_node = nodeVIndex_CUDA;
            }
        }
        else
        {
            current_cost = move8_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 8;
                best_v_node = nodeVIndex_CUDA;
            }
            current_cost = move9_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 9;
                best_v_node = nodeVIndex_CUDA;
            }
        }

        // Trying moves that insert nodeU directly after the depot
        if (nodeV_CUDA->prev->isDepot)
        {
            nodeV_CUDA = nodeV_CUDA->prev;
            routeV_CUDA = nodeV_CUDA->route;
            nodeY_CUDA = nodeV_CUDA->next;
            nodeYNextIndex_CUDA = nodeY_CUDA->next->cour;
            nodeVIndex_CUDA = nodeV_CUDA->cour;
            nodeVPrevIndex_CUDA = nodeV_CUDA->prev->cour;
            nodeYIndex_CUDA = nodeY_CUDA->cour;
            loadV_CUDA = demand_CUDA[nodeVIndex_CUDA];
            serviceV_CUDA = service_duration_CUDA[nodeVIndex_CUDA];
            loadY_CUDA = demand_CUDA[nodeYIndex_CUDA];
            serviceY_CUDA = service_duration_CUDA[nodeYIndex_CUDA];
            intraRouteMove_CUDA = (routeU_CUDA == routeV_CUDA);

            current_cost = move1_CUDA(routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeXIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, nodeYNextIndex_CUDA, loadU_CUDA, serviceU_CUDA, intraRouteMove_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 1;
                best_v_node = nodeVIndex_CUDA;
            }

            current_cost = move2_CUDA(nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeXNextIndex_CUDA, nodeXIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 2;
                best_v_node = nodeVIndex_CUDA;
            }

            current_cost = move3_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeXNextIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
            if (current_cost < best_cost)
            {
                best_cost = current_cost;
                best_move = 3;
                best_v_node = nodeVIndex_CUDA;
            }

            if (!intraRouteMove_CUDA)
            {
                current_cost = move8_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
                if (current_cost < best_cost)
                {
                    best_cost = current_cost;
                    best_move = 8;
                    best_v_node = nodeVIndex_CUDA;
                }
                current_cost = move9_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
                if (current_cost < best_cost)
                {
                    best_cost = current_cost;
                    best_move = 9;
                    best_v_node = nodeVIndex_CUDA;
                }
            }
        }
    }

    if (loopID > 0 && emptyRouteIndex != -1)
    {
        Node_CUDA *nodeV_CUDA = routes_CUDA[emptyRouteIndex].depot;
        // setLocalVariablesRouteV
        Route_CUDA *routeV_CUDA = nodeV_CUDA->route;
        Node_CUDA *nodeY_CUDA = nodeV_CUDA->next;
        int nodeYNextIndex_CUDA = nodeY_CUDA->next->cour;
        int nodeVIndex_CUDA = nodeV_CUDA->cour;
        int nodeVPrevIndex_CUDA = nodeV_CUDA->prev->cour;
        int nodeYIndex_CUDA = nodeY_CUDA->cour;
        double loadV_CUDA = demand_CUDA[nodeVIndex_CUDA];
        double serviceV_CUDA = service_duration_CUDA[nodeVIndex_CUDA];
        double loadY_CUDA = demand_CUDA[nodeYIndex_CUDA];
        double serviceY_CUDA = service_duration_CUDA[nodeYIndex_CUDA];
        intraRouteMove_CUDA = (routeU_CUDA == routeV_CUDA);

        double current_cost = move1_CUDA(routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeXIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, nodeYNextIndex_CUDA, loadU_CUDA, serviceU_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 1;
            best_v_node = nodeVIndex_CUDA;
        }
        current_cost = move2_CUDA(nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, routeU_CUDA, routeV_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeVIndex_CUDA, nodeXNextIndex_CUDA, nodeXIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 2;
            best_v_node = nodeVIndex_CUDA;
        }
        current_cost = move3_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeX_CUDA, nodeV_CUDA, nodeY_CUDA, nodeUPrevIndex_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeXNextIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA, loadU_CUDA, loadX_CUDA, serviceU_CUDA, serviceX_CUDA, intraRouteMove_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 3;
            best_v_node = nodeVIndex_CUDA;
        }

        current_cost = move9_CUDA(routeU_CUDA, routeV_CUDA, nodeU_CUDA, nodeV_CUDA, nodeUIndex_CUDA, nodeXIndex_CUDA, nodeVIndex_CUDA, nodeYIndex_CUDA);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_move = 9;
            best_v_node = nodeVIndex_CUDA;
        }
    }

    // update the best move found by this thread
    //  executed only by the per-block winner thread (so contention is blocks, not threads)
    if (best_move > 0)
    {

        unsigned int expected = 0u;
        while (atomicCAS(&global_best_lock, expected, 1u) != expected)
        {

            expected = 0u;
        }

        // --- CRITICAL SECTION: we hold the lock now ---
        double cur = global_best_cost; // read current best
        if (best_cost < cur)
        {
            global_best_cost = best_cost;
            *global_best_v_node = best_v_node;
            *global_best_u_node = tid; // your thread id
            *global_best_move = best_move;
        }

        __threadfence();

        // Release lock
        atomicExch(&global_best_lock, 0u);
    }
}

__global__ void initializing_variable_CUDA(int n, double *service_time, double *demand, double *time_cost, int *d_correlated_customers, int *numberof_correlated, double duration_limit, double vehicle_capacity)
{
    printf("Initializing CUDA variables on GPU...\n");
    customers_CUDA = n;
    service_duration_CUDA = service_time;
    demand_CUDA = demand;
    timeCost_CUDA = time_cost;
    numberof_correlated_CUDA = numberof_correlated;
    correlated_customers_CUDA = d_correlated_customers;
    durationLimit_CUDA = duration_limit;
    vehicleCapacity_CUDA = vehicle_capacity;
}

__global__ void Initialize_LocalSearch_CUDA(int nbClients, int nbVehicles, Node_CUDA *d_clients, Route_CUDA *d_routes, Node_CUDA *d_depots, Node_CUDA *d_depotsEnd, int *d_route_size, int *d_chrom_T)
{
    nbVehicles_CUDA = nbVehicles;
    chrom_T = d_chromT;
    route_size = d_route_size;
    clients_CUDA = d_clients;
    routes_CUDA = d_routes;
    depots_CUDA = d_depots;
    depotsEnd_CUDA = d_depotsEnd;
    for (int i = 0; i <= nbClients; i++)
    {
        clients_CUDA[i].cour = i;
        clients_CUDA[i].isDepot = false;
    }
    for (int i = 0; i < nbVehicles; i++)
    {
        routes_CUDA[i].cour = i;
        routes_CUDA[i].depot = &depots_CUDA[i];
        depots_CUDA[i].cour = 0;
        depots_CUDA[i].isDepot = true;
        depots_CUDA[i].route = &routes_CUDA[i];
        depotsEnd_CUDA[i].cour = 0;
        depotsEnd_CUDA[i].isDepot = true;
        depotsEnd_CUDA[i].route = &routes_CUDA[i];
    }
}

__global__ void loadRoutes_CUDA(int nbVehicles, int nbClients, int nbMoves, double penaltyCapacity, double penaltyDuration)
{

    nbMoves_CUDA = nbMoves;
    penaltyCapacityLS_CUDA = penaltyCapacity;
    penaltyDurationLS_CUDA = penaltyDuration;
    for (int r = 0; r < nbVehicles; r++)
    {
        Node_CUDA *myDepot = &depots_CUDA[r];
        Node_CUDA *myDepotFin = &depotsEnd_CUDA[r];
        Route_CUDA *myRoute = &routes_CUDA[r];
        myDepot->prev = myDepotFin;
        myDepotFin->next = myDepot;
        int len = route_size[r];
        int start = 0;
        if (r > 0)
        {
            start = route_size[r - 1];
            len -= route_size[r - 1];
        }

        if (len > 0)
        {
            Node_CUDA *myClient = &clients_CUDA[chrom_T[start]];
            myClient->route = myRoute;
            myClient->prev = myDepot;
            myDepot->next = myClient;
            for (int i = 1; i < len; i++)
            {
                Node_CUDA *myClientPred = myClient;
                myClient = &clients_CUDA[chrom_T[start + i]];
                myClient->prev = myClientPred;
                myClientPred->next = myClient;
                myClient->route = myRoute;
            }
            myClient->next = myDepotFin;
            myDepotFin->prev = myClient;
        }
        else
        {
            myDepot->next = myDepotFin;
            myDepotFin->prev = myDepot;
        }

        routes_CUDA[r].whenLastTestedSWAPStar = -1;
    }

    for (int i = 1; i <= nbClients; i++)
        clients_CUDA[i].whenLastTestedRI = -1;
}

__global__ void testing_memory()
{
    for (int i = 0; i < nbVehicles_CUDA; i++)
    {
        printf("Route %d: ", i);
        Node_CUDA *current = routes_CUDA[i].depot;
        while (true)
        {
            printf("%d -> ", current->cour);
            current = current->next;
            if (current->isDepot)
                break;
        }
        printf("DepotEnd\n");
    }
}

// each thread will have these
// Node_CUDA *nodeV;
// Node_CUDA *nodeY;
// Route_CUDA *routeV;
// int nodeVPrevIndex, nodeVIndex, nodeYIndex, nodeYNextIndex;
// double loadU, loadX;
// double serviceU, serviceX;
// bool intraRouteMove_CUDA;
