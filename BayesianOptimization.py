import numpy as np
from matplotlib import pyplot as plt
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
        # parameter_space_samples = `num_samples_bb` lists of `num_parameters` elements
        self.parameter_space_samples = get_parameter_space_samples(self.bounds[0], self.bounds[1], num_samples_bb,
                                                                   num_parameters)
        # output_values = `num_samples_bb` arrays of `num_parameters` elements
        self.output_values = [black_box_system(params) for params in self.parameter_space_samples]
        self.surrogate_model = GaussianProcessRegressor(alpha=0.1, kernel=self.find_best_kernel())
        self.max_iterations = max_iterations
        self.num_samples_surrogate = num_samples_surrogate
        self.num_parameters = num_parameters
        self.best_approximations = Result()

    def get_surrogate(self):
        return self.surrogate_model

    def get_samples(self):
        return self.parameter_space_samples

    def get_outputs(self):
        return self.output_values

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
        # print(self.parameter_space_samples, self.output_values)
        plt.scatter(self.parameter_space_samples, self.output_values)
        # line plot of surrogate function across domain
        x_samples = asarray(arange(0, 1, 0.0011))
        x_samples = x_samples.reshape(len(x_samples), 1)

        ysamples, _ = self.predict_surrogate(x_samples)
        plt.plot(x_samples, ysamples)
        plt.title(title)
        plt.show()

    def optimize(self, show_plots=False):
        # Fit the surrogate model with initial data
        self.fit_surrogate()

        if self.num_parameters < 3 and show_plots:
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
            tmp_predicted_mean, tmp_predicted_std = self.predict_surrogate(tmp_parameter_space_samples)

            # Predict using surrogate model over incremental data
            predicted_mean, predicted_std = self.predict_surrogate(self.parameter_space_samples)

            # show some partial info
            if self.num_parameters < 3 and show_plots:
                counter += 1
                # Check if the counter has reached the calculated interval
                if counter == message_interval:
                    # Evaluate the quality of the surrogate model
                    surrogate_model_quality(predicted_mean, self.output_values)
                    self.plot_surrogate(title=f"Surrogate - Iteration # {it}")
                    counter = 0

            best_mean = np.argmax(predicted_mean)

            # Calculate acquisition function values
            acquisition_values = acquisition_function_prob_improvement(tmp_predicted_mean, tmp_predicted_std,
                                                                       predicted_mean[best_mean])
            # acquisition_values = acquisition_function_ucb(tmp_predicted_mean, tmp_predicted_std)

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


def black_box_system(data, pnoise=0.1):
    """
    Simulated function to represent the black box system
    :param data: list with n elements where n is the variables of the problem
    :param pnoise: level of noise
    :return: single output value with random noise
    """
    data = np.asarray(data)  # Convert input to NumPy array
    # print("inside bb system,  print data:")
    # print(data)
    # noise = normal(0, pnoise, data.shape)
    noise = normal(0, pnoise)
    # simulated_output = (-np.sqrt(data) * np.sin(10 * data ** 2)) + noise
    output = np.sum(data ** 2 * np.sin(5 * np.pi * data) ** 6.0) + noise
    # print(f"output = {output}")
    return output
    # return np.sum((data ** 2 * np.sin(5 * np.pi * data) ** 6.0) + noise)


def create_mesh_grid(a=None, dim=2):
    if a is None:
        a = [1, 2]
    return np.array(np.meshgrid(*[a] * dim)).T.reshape(-1, dim)


def get_estimated_optima(grid=create_mesh_grid(a=None, dim=2)):
    # sample the domain without noise
    y = [black_box_system(x, pnoise=0) for x in grid]

    # find best result
    ix = np.argmax(y)
    print(f"Optima: {y[ix]}")


def plot_black_box_system_1D():
    """
    Approximate plot of the black box system for 1 dimension
    As the model is black box, we cannot do this in real world problems, but it is just a way to get an idea
    """
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
    plt.scatter(x1, ynoise)
    # plot the points without noise
    plt.plot(x1, y)

    # Add labels, title, and legend
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Approximate plot of the Black Box system')
    plt.legend()
    plt.show()


def acquisition_function_ucb(mean, std, kappa=2):
    """
    Acquisition function: Upper Confidence Bound (UCB)
    :param mean: mean via surrogate function
    :param std: stdev via surrogate function
    :param kappa:
    :return: Upper Confidence Bound (UCB
    """
    return mean + kappa * std


def acquisition_function_prob_improvement(mean, std, best):
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
    print("---------- Surrogate Model quality and interpretation ----------")
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


def plot_black_box_and_surrogate_1D(x, true_y, surrogate, result):
    # Calculate the black box system outputs
    # Generate data points for plotting
    # Calculate the black box system outputs

    # true_y = [black_box_system(param_sample) for param_sample in x]

    # Predict surrogate model outputs
    # predicted_y, _ = [surrogate.predict(param_sample, return_std=True) for param_sample in x]
    predicted_y = [prediction for prediction in surrogate.predict(x, return_std=False)]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the black box system
    plt.scatter(x, true_y, label='Black Box System', color='blue', alpha=0.5)
    plt.scatter(x, predicted_y, label='Surrogate Model', color='red', alpha=0.5)
    plt.scatter(result.best_configuration, result.best_output, label='Approx Best', color='green', alpha=0.5)

    # Add labels, title, and legend
    plt.xlabel('Input (parameters space samples)')
    plt.ylabel('Output')
    plt.title('Comparison: Black Box System vs. Surrogate Model')
    plt.legend()
    plt.xlim(min(x), max(x))
    plt.show()
