#define CATCH_CONFIG_MAIN
#include <string>

#include "../include/RandomSearch.hpp"
#include "catch2/catch.hpp"

TEST_CASE("Optimization of a 2D quadratic function", "[Random search]") {
    // Define the quadratic objective function: f(x, y) = (x - 1)^2 + (y - 2)^2
    auto objective = [](const Eigen::Vector2d& v) -> double { return std::pow(v[0] - 1, 2) + std::pow(v[1] - 2, 2); };

    // Instantiate the optimization problem with the objective function
    tinyopt::RandomSearch<double, 2> problem(objective);

    // Set bounds for the variables: 0 <= x <= 2, 0 <= y <= 3
    Eigen::Vector2d lower_bounds(2);
    lower_bounds << 0, 0;
    Eigen::Vector2d upper_bounds(2);
    upper_bounds << 2, 3;

    problem.set_bounds(lower_bounds, upper_bounds);

    // Solve the optimization problem
    Eigen::Vector2d solution = problem.solve();

    // Check if the solution is within the bounds
    CHECK(solution[0] >= lower_bounds[0]);
    CHECK(solution[0] <= upper_bounds[0]);
    CHECK(solution[1] >= lower_bounds[1]);
    CHECK(solution[1] <= upper_bounds[1]);

    // Check if the solution is close to the known minimum of the objective function
    CHECK(solution[0] == Approx(1).margin(0.1));
    CHECK(solution[1] == Approx(2).margin(0.1));
}

TEST_CASE("Optimization of the 2D Rosenbrock function", "[Random search]") {
    // The Rosenbrock's "banana" function: f(x, y) = (a - x)^2 + b*(y - x^2)^2 with a typical choice of a=1, b=100
    auto rosenbrock = [](const Eigen::Vector2d& v) -> double {
        const double a = 1.0;
        const double b = 100.0;
        return std::pow(a - v[0], 2) + b * std::pow(v[1] - v[0] * v[0], 2);
    };

    // Instantiate the optimization problem with the Rosenbrock function
    tinyopt::RandomSearch<double, 2> problem(rosenbrock);

    // Set the bounds for the variables where we expect the minimum: 0 <= x <= 2, 0 <= y <= 2
    Eigen::Vector2d lower_bounds(0, 0);
    Eigen::Vector2d upper_bounds(2, 2);
    problem.set_bounds(lower_bounds, upper_bounds);

    // Run the optimizer multiple times to check for consistency
    const int num_runs                  = 5;
    Eigen::Vector2d cumulative_solution = Eigen::Vector2d::Zero();
    for (int i = 0; i < num_runs; ++i) {
        Eigen::Vector2d solution = problem.solve();

        // Accumulate the solutions
        cumulative_solution += solution;

        // Check if each solution is within bounds
        CHECK(solution[0] >= lower_bounds[0]);
        CHECK(solution[0] <= upper_bounds[0]);
        CHECK(solution[1] >= lower_bounds[1]);
        CHECK(solution[1] <= upper_bounds[1]);
    }

    // Average the solutions
    Eigen::Vector2d average_solution = cumulative_solution / num_runs;

    // Check if the average solution is close to the known minimum of the objective function
    CHECK(average_solution[0] == Approx(1).epsilon(0.1));
    CHECK(average_solution[1] == Approx(1).epsilon(0.1));
}