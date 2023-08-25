from BayesianOptimization import BayesianOptimization, black_box_system, plot_black_box_system, \
    get_parameter_space_samples, plot_black_box_and_surrogate


def main():
    # show black box system plot
    plot_black_box_system()

    parameter_bounds = [0, 1]  # using the same for all parameters
    num_parameters = 1
    num_initial_samples = 10  # A small amount of samples because the system is expensive
    num_samples_surrogate = 20  # As the surrogate is cheaper, we can use more samples to guide the approximation
    num_optimization_steps = 100

    bayesian_optimization = BayesianOptimization(black_box_system, num_initial_samples, num_samples_surrogate,
                                                 parameter_bounds, num_parameters, num_optimization_steps)
    result = bayesian_optimization.optimize(show_plots=True)
    print(f"Best Configuration: {result.best_configuration} | Best Output: {result.best_output}")

    if num_parameters == 1:
        plot_black_box_and_surrogate(num_parameters, bayesian_optimization.get_surrogate())


if __name__ == '__main__':
    main()
