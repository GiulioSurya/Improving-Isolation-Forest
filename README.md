

# Improving Isolation Forest with Context-Behavior Consistency

## Project Overview

This project explores how the integration of **context-behavior consistency assumptions** can improve unsupervised anomaly detection models.
Specifically, we propose a method to enhance **Isolation Forest** by introducing a **regression phase** that models expected behavior from context attributes, and then detects anomalies based on **residual deviations**.

The key idea is simple:
**If two observations are similar in their context attributes, they should also be similar in their behavior attributes.**
Violations of this assumption can be strong indicators of anomalous behavior.

## Methodology

1. **Regression Phase**:

   * A regression model (e.g., Random Forest Regressor) is trained to predict behavioral attributes from contextual attributes.
   * For each observation, the residual (actual value minus predicted value) is computed.

2. **Anomaly Detection Phase**:

   * Residuals are used as input features for an unsupervised anomaly detection model (e.g., Isolation Forest).
   * Larger residuals signal stronger deviations from expected behavior given the context.

3. **Evaluation**:

   * The enhanced model is compared against standard Isolation Forest on carefully constructed test datasets (see Experimental Setup below).

## Experimental Setup

Measuring the quality of unsupervised anomaly detection models is notoriously difficult. To address this, we design an experimental protocol based on two key assumptions:

* **Assumption 1**:
  Most points in a dataset are not anomalies.
  Therefore, a random sample of the dataset should contain relatively few anomalies.

* **Assumption 2**:
  If we perturb a sample of points by swapping indicator attribute values among them (while keeping context attributes intact),
  the resulting dataset should contain a **higher fraction of anomalies** compared to an unperturbed sample.

### Protocol

For each dataset, the following steps are repeated 10 times:

1. **Train/Test Split**:

   * Randomly split the dataset: 80% training data, 20% test data (`testData`).

2. **Outlier Identification**:

   * Identify \~20% of the most extreme points in `testData` using a **Gaussian Mixture Model (GMM)**.
   * Importantly, outliers are selected **only based on environmental (contextual) attributes**, ignoring indicator attributes.

3. **Perturbation**:

   * Randomly split `testData` into two equal subsets: `perturbed` and `nonPerturbed`, ensuring that `outliers` belong to `nonPerturbed`.
   * **Perturb** the `perturbed` set by swapping indicator values among points, thus disrupting context-behavior consistency.

4. **Anomaly Detection**:

   * Apply the anomaly detection model on the full `testData` (`perturbed` + `nonPerturbed`).
   * Evaluate if the model correctly flags perturbed records as anomalies, while maintaining low false positive rates on non-perturbed and outlier points.

### Perturbation Details

* For each point to perturb, select `k = min(50, floor(|D|/4))` other points.
* From the sample, choose the point with the **maximum Euclidean distance** in the indicator space.
* Replace the original indicator attributes with those of the selected point, keeping context attributes unchanged.
* This method ensures that perturbed data remains realistic but breaks context-behavior consistency.

## Expected Behavior of Good Anomaly Detection

A robust model should:

* Flag a **high fraction** of points in the `perturbed` set as anomalies.
* Flag a **low fraction** of points in the `nonPerturbed` set as anomalies.
* Flag a **similarly low fraction** of points in the `outliers` set as anomalies.

A high anomaly rate in `outliers` (contextually extreme but behaviorally typical points) would suggest an undesirable sensitivity to irrelevant context variation.


## Repository Structure

This repository is organized into modular components that support the full pipeline of data generation, perturbation, residual modeling, anomaly detection, and evaluation.

* **`anomaly_det.py`**
  Defines the `AnomalyDetector` class, which supports both standard and context-aware anomaly detection. The context-aware mode uses regression-based residuals before applying Isolation Forest.

* **`perturber.py`**
  Implements the `DatasetPreparer` class, which performs train-test splitting, identifies contextual outliers using GMM, applies perturbation based on behavioral dissimilarity, and assigns anomaly/outlier flags accordingly.

* **`residual.py`**
  Contains the `ComputeResiduals` class. It models each behavioral variable using a Random Forest regressor trained on contextual variables, optionally tuned via Bayesian optimization. Residuals are computed using cross-validation.

* **`run_anomaly_batches_2.py`**
  Runs multiple experiments on real or synthetic datasets, evaluating three detection modes: standard Isolation Forest, contextual Isolation Forest (on residuals), and indicator-only. Outputs are saved as structured Excel files.

* **`syntetic_dataset.py`**
  Provides the `GenerateData` class, a flexible generator of synthetic datasets with configurable numbers of contextual and behavioral variables, noise, and anomaly injection via external shocks.

* **`syntetic_results.py`**
  Runs a grid-based benchmark of anomaly detection accuracy over combinations of context covariance matrices and shock variances. Useful to assess model performance under varying structural assumptions.




