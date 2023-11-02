#include <fstream>
#include <iostream>

#include "../include/RandomSearch.hpp"

int main() {
    // Define the quadratic objective function: f(x, y) = (x - 1)^2 + (y - 2)^2
    auto objective = [](const Eigen::VectorXd& v) -> double { return std::pow(v[0] - 1, 2) + std::pow(v[1] - 2, 2); };

    // Instantiate the optimization problem with the objective function
    tinyopt::RandomSearch<double, 2> problem(objective);

    // Set bounds for the variables: 0 <= x <= 2, 0 <= y <= 3
    Eigen::VectorXd lower_bounds(2);
    lower_bounds << 0, 0;
    Eigen::VectorXd upper_bounds(2);
    upper_bounds << 2, 3;

    problem.set_bounds(lower_bounds, upper_bounds);

    // Solve the optimization problem
    Eigen::VectorXd solution = problem.solve();

    // Output the solution
    std::cout << "Optimal solution found at: " << solution.transpose() << std::endl;

    // Save the data for plotting
    std::ofstream data_file("solution.dat");
    if (data_file.is_open()) {
        data_file << "x_opt y_opt\n";
        data_file << solution[0] << " " << solution[1] << std::endl;
        data_file.close();
    }
    else {
        std::cerr << "Unable to open file for writing the solution." << std::endl;
        return -1;
    }

    return 0;
}
