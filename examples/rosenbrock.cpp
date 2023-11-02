#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "../include/RandomSearch.hpp"  // Adjust the include path according to your project structure

int main() {
    // Rosenbrock function: f(x, y) = (a - x)^2 + b*(y - x^2)^2
    const double a  = 1.0;
    const double b  = 100.0;
    auto rosenbrock = [a, b](const Eigen::VectorXd& v) -> double {
        return std::pow(a - v[0], 2) + b * std::pow(v[1] - v[0] * v[0], 2);
    };

    // Instantiate the optimization problem with the Rosenbrock function
    tinyopt::RandomSearch<double, 2> problem(rosenbrock);

    // Set bounds for the variables: -2 <= x <= 2, -2 <= y <= 3
    Eigen::VectorXd lower_bounds(2);
    lower_bounds << -2, -1;
    Eigen::VectorXd upper_bounds(2);
    upper_bounds << 2, 3;

    problem.set_bounds(lower_bounds, upper_bounds);

    // Solve the optimization problem
    Eigen::VectorXd solution = problem.solve();

    // Output the solution
    std::cout << "Optimal solution found at: " << solution.transpose() << std::endl;

    // Save the data for plotting
    std::ofstream data_file("rosenbrock_solution.dat");
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
