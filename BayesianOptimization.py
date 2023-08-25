from math import sin, pi

import numpy as np
from matplotlib import pyplot, pyplot as plt
from numpy import asarray, arange
from numpy.random import normal
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, Exponentiation, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score


class BayesianOptimization:
    def __init__(self, objective_function, num_samples_bb, num_samples_surrogate, bounds, num_parameters,
                 max_iterations=50):
        self.objective_function = objective_function
        self.bounds = bounds
        self.parameter_space_samples = get_parameter_space_samples(self.bounds[0], self.bounds[1], num_samples_bb,
                                                                   num_parameters)
        self.output_values = [black_box_system(params) for params in self.parameter_space_samples]
        self.surrogate_model = GaussianProcessRegressor(alpha=0.1, kernel=self.find_best_kernel())
        self.max_iterations = max_iterations
        self.num_samples_surrogate = num_samples_surrogate
        self.num_parameters = num_parameters
        self.best_approximations = Result()

    def get_surrogate(self):
        return self.surrogate_model

    def find_best_kernel(self):
        # Example data and targets
        x = self.parameter_space_samples
        y = self.output_values

        # Experiment with different kernels
        kernels = [
            RBF(length_scale=1.0),
            Matern(nu=1.5),
            RationalQuadratic(length_scale=1.0, alpha=0.1),
            Exponentiation(RBF(length_scale=1.0), exponent=2.0),
            ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0))
        ]

        best_score = float("-inf")
        best_kernel = None
        for kernel in kernels:
            gp = GaussianProcessRegressor(alpha=0.1, kernel=kernel)
            gp.fit(x, y)
            # Evaluate model performance
            score = gp.score(x, y)
            if score > best_score:
                best_score = score
                best_kernel = kernel

            # print(f"Kernel: {kernel}\nScore: {score}\n")
        print(f"Best Kernel: {best_kernel}\nScore: {best_score}\n")
        return best_kernel

    def predict_surrogate(self, parameter_combinations):
        return self.surrogate_model.predict(parameter_combinations, return_std=True)

    def fit_surrogate(self):
        self.surrogate_model.fit(self.parameter_space_samples, self.output_values)

    def get_best_approximations(self):
        best_index = np.argmax(self.output_values)
        best_configuration = self.parameter_space_samples[best_index]
        best_output = self.output_values[best_index]
        return Result(best_configuration, best_output)

    # plot real observations vs surrogate function
    def plot_surrogate(self, title=None):
        # scatter plot of inputs and real objective function
        pyplot.scatter(self.parameter_space_samples, self.output_values)
        # line plot of surrogate function across domain
        x_samples = asarray(arange(0, 1, 0.0011))
        x_samples = x_samples.reshape(len(x_samples), 1)

        ysamples, _ = self.predict_surrogate(x_samples)
        pyplot.plot(x_samples, ysamples)
        pyplot.title(title)
        # show the plot
        pyplot.show()

    def optimize(self, show_plots=False):
        # Fit the surrogate model with initial data
        self.fit_surrogate()

        if self.num_parameters == 1 and show_plots:
            self.plot_surrogate(title="Surrogate Model - Before iterations")

        desired_plots = 5  # Total number of plots to show
        # Calculate the plot interval to evenly distribute messages
        message_interval = max(self.max_iterations // desired_plots, 1)  # Ensure interval is at least 1
        counter = 0  # Initialize the counter
        for it in range(self.max_iterations):
            # Create a bigger sample of parameters space for the surrogate model
            tmp_parameter_space_samples = get_parameter_space_samples(self.bounds[0], self.bounds[1],
                                                                      self.num_samples_surrogate, self.num_parameters)
            # Predict using surrogate model over temporal data
            new_predicted_mean, new_predicted_std = self.predict_surrogate(tmp_parameter_space_samples)

            # Predict using surrogate model over incremental data
            predicted_mean, _ = self.predict_surrogate(self.parameter_space_samples)

            # show some partial info
            if self.num_parameters == 1 and show_plots:
                counter += 1
                # Check if the counter has reached the calculated interval
                if counter == message_interval:
                    # Evaluate the quality of the surrogate model
                    surrogate_model_quality(predicted_mean, self.output_values)
                    self.plot_surrogate(title=f"Surrogate - Iteration # {it}")
                    counter = 0

            best_mean = np.argmax(predicted_mean)

            # Calculate acquisition function values
            acquisition_values = acquisition_function_2(new_predicted_mean, new_predicted_std,
                                                        predicted_mean[best_mean])

            # Find the next parameter combination to evaluate using the expensive model
            next_index = np.argmax(acquisition_values)
            next_sample = tmp_parameter_space_samples[next_index]

            # Evaluate the black box system and update the surrogate model
            next_output = black_box_system(next_sample)
            self.parameter_space_samples = np.vstack((self.parameter_space_samples, next_sample))
            self.output_values = np.append(self.output_values, next_output)

            # Improve surrogate model
            self.fit_surrogate()

        # Return the best result
        return self.get_best_approximations()


# Helper functions
class Result:
    def __init__(self, value1=None, value2=0):
        if value1 is None:
            value1 = []
        self.best_configuration = value1
        self.best_output = value2


def get_parameter_space_samples(low, high, num_samples, num_parameters):
    return np.random.uniform(low=low, high=high, size=(num_samples, num_parameters))


# Simulated function to represent the black box system
def black_box_system(data, pnoise=0.1):
    noise = normal(0, pnoise, data.shape)
    # simulated_output = (-np.sqrt(data) * np.sin(10 * data ** 2)) + noise
    #simulated_output = (data ** 2 * sin(5 * pi * data) ** 6.0) + noise
    simulated_output = (data ** 2 * np.sin(5 * np.pi * data) ** 6.0) + noise

    return simulated_output


def plot_black_box_system():
    # grid-based sample of the domain [0,1]
    x1 = arange(0, 1, 0.01)
    # sample the domain without noise
    y = [black_box_system(x, pnoise=0) for x in x1]
    # sample the domain with noise
    ynoise = [black_box_system(x) for x in x1]
    # find best result
    ix = np.argmax(y)
    print('Optima: x1=%.3f, y=%.3f' % (
        x1[ix], y[ix]))  # This is something that in real problems we can't do, as we don't know y without noise

    # plot the points with noise
    pyplot.scatter(x1, ynoise)
    # plot the points without noise
    pyplot.plot(x1, y)
    pyplot.title("Black Box System")
    # show the plot
    pyplot.show()


def acquisition_function(mean, std, kappa=2):
    """
    Acquisition function: Upper Confidence Bound (UCB)
    :param mean: mean via surrogate function
    :param std: stdev via surrogate function
    :param kappa:
    :return: Upper Confidence Bound (UCB
    """
    return mean + kappa * std


def acquisition_function_2(mean, std, best):
    """
    Probability of improvement acquisition function
    :param mean: mean via surrogate function
    :param std: stdev via surrogate function
    :param best: the best surrogate score found so far
    :return:the probability of improvement
    """
    return norm.cdf((mean - best) / (std + 1E-9))


def surrogate_model_quality(surrogate_preds, true_values):

    mse = mean_squared_error(true_values, surrogate_preds)
    r2 = r2_score(true_values, surrogate_preds)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2) Score: {r2}")

    # Interpretation based on the calculated values
    # Adjust these thresholds based on the context of the problem and data.
    if mse < 0.1:
        print("MSE suggests a very close match between surrogate and true values.")
    elif mse < 1:
        print("MSE suggests a reasonable approximation by the surrogate.")
    else:
        print("MSE indicates a significant difference between surrogate and true values.")

    if r2 > 0.8:
        print("R2 score suggests the surrogate explains a high portion of variability.")
    elif r2 > 0.5:
        print("R2 score suggests the surrogate captures a moderate portion of variability.")
    else:
        print("R2 score indicates the surrogate's explanations are limited.")


def plot_black_box_and_surrogate(num_parameters, surrogate):
    # Calculate the black box system outputs
    # Generate data points for plotting
    x_bb = get_parameter_space_samples(0, 1, 20, num_parameters)
    x = np.linspace(min(x_bb), max(x_bb), 100)  # Adjust the range as needed

    # Calculate the black box system outputs
    true_y = black_box_system(x_bb)

    # Predict surrogate model outputs
    predicted_y, _ = surrogate.predict(x.reshape(-1, 1), return_std=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the black box system
    plt.scatter(x_bb, true_y, label='Black Box System', color='blue', alpha=0.5)

    # Plot the surrogate model predictions
    plt.plot(x, predicted_y, label='Surrogate Model', color='red')

    # Add labels, title, and legend
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Comparison: Black Box System vs. Surrogate Model')
    plt.legend()

    # Show the plot
    plt.show()
