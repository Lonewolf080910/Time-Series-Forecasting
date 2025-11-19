Project: Advanced Time Series Forecasting with Neural Networks and Attention Mechanisms
This project aims to develop and rigorously evaluate an attention-based Long Short-Term Memory (LSTM) network for time series forecasting, comparing its performance against a baseline LSTM model. The dataset used is the 'Atmospheric CO2 Concentration' from statsmodels, serving as a substitute for the initially planned 'Electric Power Consumption' which was unavailable.

Table of Contents
Technical Report
Architecture of the Attention Mechanism
Rationale for Hyperparameter Choices
Results of Rolling-Origin Cross-Validation
Comparative Performance Summary Table
Setup and Usage
Key Findings and Insights
Next Steps
1. Technical Report
1.1. Architecture of the Attention Mechanism
The core of this project lies in implementing a custom self-attention mechanism within an LSTM network. The model architecture comprises:

Input Layer: Accepts sequential data (e.g., 52 previous timesteps) for forecasting.
LSTM Layer: A standard LSTM layer processes the input sequence, generating a sequence of hidden states. The return_sequences=True parameter is crucial here, as it allows the attention mechanism to attend over all timesteps' outputs.
Custom Attention Layer: This layer is designed to dynamically weigh the importance of each hidden state from the LSTM output. It calculates a context vector by taking a weighted sum of the LSTM outputs, where the weights are determined by an attention scoring mechanism. This mechanism involves transforming the LSTM outputs and then computing alignment scores, which are then normalized using a softmax function to produce attention weights. The context vector effectively summarizes the most relevant information from the input sequence for making a prediction.
Output Layer: A dense layer with a single unit and linear activation, responsible for producing the final time series forecast.
1.2. Rationale for Hyperparameter Choices
Hyperparameter optimization was crucial for both models. For the attention-based LSTM, keras_tuner.RandomSearch was utilized to find optimal values for:

LSTM Units: Explored a range from 32 to 128 units, in steps of 32, to determine the optimal model complexity.
Learning Rate: Tested discrete learning rates of 1e-2, 1e-3, and 1e-4 to find the most effective convergence speed.
An EarlyStopping callback was implemented during tuning to prevent overfitting, monitoring val_loss with a patience of 3 epochs. The best model found during tuning was then trained more extensively (50 epochs) with a patience of 5 epochs for val_loss and weights restored from the best epoch.

1.3. Results of Rolling-Origin Cross-Validation
Rolling-origin cross-validation (also known as walk-forward validation) was used to provide a robust and unbiased evaluation of both models. This technique simulates a real-world forecasting scenario by iteratively retraining the models on an expanding window of training data and then forecasting on a fixed, subsequent test window. This approach is superior to a single train-test split for time series data as it accounts for the temporal dependency and concept drift.

The evaluation metrics for both models, aggregated across all folds of the rolling-origin cross-validation, are as follows:

2. Comparative Performance Summary Table
Metric	Attention-based LSTM (Rolling CV)	Baseline LSTM (Rolling CV)
RMSE	1.6104	1.2298
MAE	1.2497	0.9950
MAPE	0.3450%	0.2752%
3. Setup and Usage
To run this project, ensure you have Google Colab environment with the necessary libraries installed. The keras-tuner library needs to be installed, which is handled automatically by the provided notebook cells.

The notebook executes the following steps:

Loads and explores the 'CO2' dataset from statsmodels.
Preprocesses the data, including handling missing values, Min-Max scaling, and creating sequential input windows.
Implements a custom Attention layer and integrates it into an LSTM model.
Implements a standard LSTM model as a baseline.
Performs hyperparameter optimization for the attention-based LSTM using KerasTuner.
Trains the best attention-based LSTM model and the baseline LSTM model.
Performs rolling-origin cross-validation for both models and calculates performance metrics.
Visualizes predictions, actuals, and residuals for both models.
4. Key Findings and Insights
Dataset Substitution: The 'Electric Power Consumption' dataset was replaced with the 'CO2' dataset due to unavailability. The 'CO2' dataset exhibits strong trends and seasonality, making it suitable for time series forecasting.
Data Preparation: Missing values were effectively handled through linear interpolation, and Min-Max scaling successfully normalized the data. The windowing process created appropriate input-output sequences for RNN training.
Attention Mechanism Implementation: A custom Keras Attention layer was successfully developed and integrated. An initial IndexError during its implementation was resolved by correctly handling tensor dimensions using K.sum(uit * self.u, axis=-1) instead of K.dot(uit, self.u).
Hyperparameter Optimization: KerasTuner efficiently identified optimal hyperparameters for the attention-based LSTM, enhancing its potential performance.
Comparative Performance: Contrary to initial expectations, the Baseline LSTM model consistently outperformed the Attention-based LSTM model across all evaluated metrics (RMSE, MAE, MAPE) during rolling-origin cross-validation. This indicates that for this particular CO2 dataset, the added complexity of the attention mechanism did not provide a tangible benefit, and a simpler LSTM model was more effective.
5. Next Steps
Attention Mechanism Analysis: Investigate why the attention mechanism did not improve performance. This could involve visualizing attention weights to understand what parts of the sequence the model is attending to, or experimenting with different types of attention mechanisms (e.g., multi-head attention, different scoring functions).
Baseline Hyperparameter Tuning: Although the baseline performed better, its hyperparameters were not as extensively tuned as the attention model. Further optimization of the baseline LSTM (e.g., number of units, learning rate schedules, regularization) could lead to even greater improvements.
Explore Other Advanced Models: Consider other advanced time series models (e.g., Transformers, TCNs) that might be better suited for this dataset or incorporate more complex features.
Feature Engineering: Explore additional features (e.g., lag features, Fourier terms for seasonality, external covariates) that could enhance forecasting accuracy for both models.
