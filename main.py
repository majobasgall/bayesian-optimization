from BayesianOptimization import BayesianOptimization, black_box_system, plot_black_box_system_1D, \
    plot_black_box_and_surrogate_1D


def main():
    parameter_bounds = [0, 1]  # using the same for all parameters
    num_parameters = 1
    num_initial_samples = 10  # A small amount of samples because the system is expensive
    num_samples_surrogate = 20  # As the surrogate is cheaper, we can use more samples to guide the approximation
    num_optimization_steps = 100

    if num_parameters == 1:
        # show black box system plot
        plot_black_box_system_1D()

    bayesian_optimization = BayesianOptimization(black_box_system, num_initial_samples, num_samples_surrogate,
                                                 parameter_bounds, num_parameters, num_optimization_steps)
    result = bayesian_optimization.optimize(show_plots=True)
    print(f"Approx Best Configuration: {result.best_configuration} | Approx Best Output: {result.best_output}")

    if num_parameters == 1:
        # Show comparison
        plot_black_box_and_surrogate_1D(bayesian_optimization.get_samples(), bayesian_optimization.get_outputs(),
                                        bayesian_optimization.get_surrogate(), result)


if __name__ == '__main__':
    main()
