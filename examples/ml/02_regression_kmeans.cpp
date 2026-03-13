/**
 * @file 02_regression_kmeans.cpp
 * @brief Example demonstrating basic usage of Vix.AI.ML for regression and clustering.
 *
 * This example showcases:
 * 1. **Linear Regression** – training on a noisy dataset (y = 2x + 1 + noise)
 * 2. **K-Means Clustering** – identifying two Gaussian blobs in 2D space
 *
 * It also demonstrates the usage of:
 * - `StandardScaler` for normalization
 * - `mse()` for regression evaluation
 * - `KMeans::centers()` and `predict_one()` for clustering visualization
 *
 * ### Run
 * @code
 * $ ./vix-ai-ml-demo
 * [LR] MSE: 0.0382
 * [KMeans] centers:
 *   (0.05, -0.02)
 *   (3.12, 3.01)
 * [KMeans] example label: 0
 * @endcode
 */

#include <vix/ai/ml/Regression.hpp>
#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Metrics.hpp>
#include <iostream>
#include <random>

using namespace vix::ai::ml;

int main()
{
  // 1. Linear Regression Demo  (y = 2x + 1 + noise)

  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.2);

  Mat X;
  Vec y;
  for (int i = 0; i < 200; ++i)
  {
    double x = i * 0.05;
    X.push_back({x});
    y.push_back(2.0 * x + 1.0 + noise(rng));
  }

  // Normalize input features
  StandardScaler scaler;
  Mat Xs = scaler.fit_transform(X);

  // Train regression model
  LinearRegression lr;
  lr.set_hyperparams(0.1, 3000);
  lr.fit(Xs, y);

  // Predict and evaluate
  Vec yhat = lr.predict(Xs);
  std::cout << "[LR] MSE: " << mse(y, yhat) << "\n";

  // 2. K-Means Clustering Demo  (two 2D Gaussian blobs)

  Mat U;
  std::normal_distribution<double> g1x(0.0, 0.5), g1y(0.0, 0.5);
  std::normal_distribution<double> g2x(3.0, 0.5), g2y(3.0, 0.5);

  // Generate synthetic data for two clusters
  for (int i = 0; i < 100; ++i)
    U.push_back({g1x(rng), g1y(rng)});
  for (int i = 0; i < 100; ++i)
    U.push_back({g2x(rng), g2y(rng)});

  // Train KMeans with k=2 clusters
  KMeans km(2, 50);
  km.fit(U);

  std::cout << "[KMeans] centers:\n";
  for (const auto &c : km.centers())
    std::cout << "  (" << c[0] << ", " << c[1] << ")\n";

  std::cout << "[KMeans] example label: " << km.predict_one({0.0, 0.0}) << "\n";

  return 0;
}
