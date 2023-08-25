# Bayesian Optimization: Approximate to the optimum of an expensive Black Box system by using a cheap surrogate model

This code demonstrates the use of Bayesian optimization to approximate the optimal solution of a black box system
using surrogate models. Bayesian optimization is an efficient technique for optimizing expensive-to-evaluate functions
by iteratively selecting the next point to evaluate based on the surrogate model's predictions and an acquisition
function.

<div style="display: flex;">
  <img src="https://github.com/majobasgall/bayesian-optimization/raw/main/img/bb_system.jpg" style="width: 25%;" />
  <img src="https://github.com/majobasgall/bayesian-optimization/raw/main/img/surrogate_it39.jpg" style="width: 25%;" />
  <img src="https://github.com/majobasgall/bayesian-optimization/raw/main/img/surrogate_it79.jpg" style="width: 25%;" />
  <img src="https://github.com/majobasgall/bayesian-optimization/raw/main/img/bb_vs_surrogate.jpg" style="width: 25%;" />
</div>

## Objective

The idea is to find the optimal configuration of parameters for a black box system, where
evaluating the black box system is time-consuming. Bayesian Optimization uses a surrogate model, in this case, a
Gaussian Process
Regressor (but can be Random Forest, for example), to predict the behavior of the black box system and guide the search
for the optimal configuration.

## Considerations

When using surrogate models for optimization, keep the following considerations in mind:

- **Surrogate Model Accuracy:** The accuracy of the surrogate model is crucial. It's important to validate the model's
  predictions against actual evaluations to ensure reliable optimization. The `surrogate_model_quality` function shows
  two metrics with some interpretations.

- **Sampling Strategy:** The selection of sampling points and the balance between exploration and exploitation impact
  the optimization process. Choose a sampling strategy that suits your optimization problem.
  The `get_parameter_space_samples` function gets random parameter space samples.

- **Model Selection:** Experiment with different surrogate models (kernels) such as RBF, Matern, and others to find the
  best fit for your problem. The `find_best_kernel` function selects the "best" model. You can play with that to try more
  kernels and configurations.

- **Overfitting:** Be cautious of overfitting. Regularly validate the surrogate model's performance and consider adding
  more data points if necessary.

- **Visualization:** Visualize the surrogate model's predictions against the black box system to assess its accuracy and
  behavior. The `plot_black_box_and_surrogate` provides this plot when the system has one parameter.

## Getting Started

1. Install the required Python packages using `pip install -r requirements.txt`.

2. Configure your black box system and its parameters, as well as other settings in the main program.

3. If you are working with a one-dimensional problem, you can call the optimization with `show_plots=True` to see, every
   5 iterations, the surrogate model plot.

4. Run the program to perform Bayesian optimization with the surrogate model.

## Result Interpretation

After running the program, you will obtain the optimal configuration of parameters as predicted by the surrogate model.
Compare this result with the actual evaluation of the black box system to determine the accuracy of the surrogate
model's predictions.

## Disclaimer

While surrogate models are effective in approximating the behavior of the black box system, there may be cases where
the surrogate model doesn't precisely capture the system's shape or nuances. The model's predictions rely on existing
data and assumptions inherent in the chosen kernel.

## Conclusion

Bayesian optimization with surrogate models provides an efficient way to optimize complex and time-consuming processes.
By using the surrogate model's predictions, you can make informed decisions about which points to evaluate, saving
computational resources and time.

Feel free to experiment with different surrogate models, acquisition functions, sampling strategies, and configurations
to find the best optimization approach for your specific problem.