#ifndef TINYOPT
#define TINYOPT

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <future>
#include <random>
#include <thread>
#include <vector>

/** \file random_search.hpp
 * @brief Contains the random search nonlinear optimization class.
 */
namespace tinyopt {

    /** @brief RandomSearch optimization class
     */
    template <typename Scalar, int n>
    class RandomSearch {
    public:
        using Vector   = Eigen::Matrix<Scalar, n, 1>;
        using Function = std::function<Scalar(const Vector&)>;

        /** @brief Constructor for the optimization problem
         * @param objective The objective function to minimize or maximize
         */
        explicit RandomSearch(Function objective) : objective_(std::move(objective)) {}

        /** @brief Set the bounds for the optimization variables
         * @param lower The lower bounds
         * @param upper The upper bounds
         */
        void set_bounds(const Vector& lower, const Vector& upper) {
            lower_bounds_ = lower;
            upper_bounds_ = upper;
        }

        /** @brief Set the number of samples to generate
         * @param num The number of samples
         */
        void set_num_samples(const int& num) {
            num_samples = num;
        }

        /** @brief Solve the optimization problem using random search
         * @return The solution vector
         */
        Vector solve() {
            std::mt19937 rng(std::random_device{}());  // Common random number generator (for seed generation)

            Vector best_sample = Vector::Zero(lower_bounds_.size());
            Scalar best_cost   = std::numeric_limits<Scalar>::max();

            const size_t num_threads =
                std::thread::hardware_concurrency();  // Get the number of concurrent threads supported
            std::vector<std::future<void>> futures;

            std::mutex best_mutex;  // For thread-safe access to best_cost and best_sample

            // Calculate samples per thread and extras
            const size_t samples_per_thread = num_samples / num_threads;
            const size_t extra_samples = num_samples % num_threads;  // Remainder samples that need to be distributed

            // Lambda to perform the random search in parallel
            auto worker =
                [this, &best_sample, &best_cost, &best_mutex](size_t start_index, size_t end_index, int seed) {
                    std::mt19937 local_rng(seed);  // Local RNG for this thread
                    std::uniform_real_distribution<Scalar> dist;
                    Vector local_best_sample = Vector::Zero(lower_bounds_.size());
                    Scalar local_best_cost   = std::numeric_limits<Scalar>::max();

                    for (size_t i = start_index; i < end_index; i++) {
                        Vector sample = Vector::Zero(lower_bounds_.size());
                        for (size_t j = 0; j < lower_bounds_.size(); ++j) {
                            typename std::uniform_real_distribution<Scalar>::param_type dist_param(lower_bounds_[j],
                                                                                                   upper_bounds_[j]);
                            dist.param(dist_param);
                            sample[j] = dist(local_rng);
                        }

                        Scalar cost = objective_(sample);

                        if (cost < local_best_cost) {
                            local_best_cost   = cost;
                            local_best_sample = sample;
                        }
                    }

                    // Combine results in a thread-safe manner
                    std::lock_guard<std::mutex> lock(best_mutex);
                    if (local_best_cost < best_cost) {
                        best_cost   = local_best_cost;
                        best_sample = local_best_sample;
                    }
                };

            // Start threads
            size_t start_index = 0;
            for (size_t i = 0; i < num_threads; ++i) {
                // Determine the number of samples for this thread to process
                size_t end_index = start_index + samples_per_thread + (i < extra_samples ? 1 : 0);

                futures.push_back(std::async(std::launch::async, worker, start_index, end_index, rng()));
                start_index = end_index;  // Update start index for the next thread
            }

            // Wait for all threads to complete
            for (auto& future : futures) {
                future.get();
            }

            return best_sample;
        }

    private:
        // Parameters
        int num_samples = 10000;
        Function objective_;
        Vector lower_bounds_ = Vector::Zero();
        Vector upper_bounds_ = Vector::Zero();
    };

}  // namespace tinyopt

#endif  // TINYOPT
