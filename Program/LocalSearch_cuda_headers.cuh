#ifndef LOCAL_SEARCH_CUDA_HEADERS_CUH
#define LOCAL_SEARCH_CUDA_HEADERS_CUH
#pragma once
struct Route_CUDA;

struct Node_CUDA
{
    bool isDepot;         // Tells whether this node represents a depot or not
    int cour;             // Node_CUDA index
    int position;         // Position in the route
    int whenLastTestedRI; // "When" the RI moves for this node have been last tested
    Node_CUDA *next;      // Next node in the route order
    Node_CUDA *prev;
    Route_CUDA *route;                // Previous node in the route order
    double cumulatedLoad;             // Cumulated load on this route until the customer (including itself)
    double cumulatedTime;             // Cumulated time on this route until the customer (including itself)
    double cumulatedReversalDistance; // Difference of cost if the segment of route (0...cour) is reversed (useful for 2-opt moves with asymmetric problems)
    double deltaRemoval;              // Difference of cost in the current route if the node is removed (used in SWAP*)
};

struct Route_CUDA
{
    int cour;                   // Route index
    int nbCustomers;            // Number of customers visited in the route
    int whenLastModified;       // "When" this route has been last modified
    int whenLastTestedSWAPStar; // "When" the SWAP* moves for this route have been last tested
    double duration;
    Node_CUDA *depot;            // Total time on the route
    double load;                 // Total load on the route
    double reversalDistance;     // Difference of cost if the route is reversed
    double penalty;              // Current sum of load and duration penalties
    double polarAngleBarycenter; // Polar angle of the barycenter of the route
};

// __device__ double penaltyExcessDuration_CUDA(double myDuration_CUDA);
// __device__ double penaltyExcessLoad_CUDA(double myLoad_CUDA);
// __device__ double move1_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move2_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move3_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move4_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move5_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move6_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move7_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move8_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);
// __device__ double move9_CUDA(Node_CUDA *nodeV_CUDA, Node_CUDA *nodeY_CUDA, Route_CUDA *routeV_CUDA, int nodeVPrevIndex_CUDA, int nodeVIndex_CUDA, int nodeYIndex_CUDA, int nodeYNextIndex_CUDA, double loadU_CUDA, double loadX_CUDA, double serviceU_CUDA, double serviceX_CUDA, bool intraRouteMove_CUDA);

__global__ void kernel1_CUDA(int loopID, int emptyRouteIndex, unsigned int *global_best_v_node, unsigned int *global_best_u_node, unsigned int *global_best_move);
__global__ void initializing_variable_CUDA(int n, double *service_time, double *demand, double *time_cost, int *d_correlated_customers, int *numberof_correlated, double duration_limit, double vehicle_capacity);
__global__ void Initialize_LocalSearch_CUDA(int nbClients, int nbVehicles, Node_CUDA *d_clients, Route_CUDA *d_routes, Node_CUDA *d_depots, Node_CUDA *d_depotsEnd, int *d_route_size, int *d_chrom_T);
__global__ void loadRoutes_CUDA(int nbVehicles, int nbClients, int nbMoves, double penaltyCapacity, double penaltyDuration);
#endif