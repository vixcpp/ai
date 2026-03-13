/**
 * @file test_ml.cpp
 * @brief Functional test for Vix.AI.ML:
 *        - LinearRegression (gradient descent + closed-form)
 *        - KMeans clustering
 *
 * This file is used by CTest via:
 *   add_test(NAME vix_ai_ml_test COMMAND vix_ai_ml_test)
 */

#include <vix/ai/ml/Regression.hpp>
#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Metrics.hpp>

#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>

using namespace vix::ai::ml;

int main()
{
  std::cout << "=== [Vix.AI.ML] Functional Test ===\n";

  // 1. Linear Regression test : y = 2x + 1 + noise

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

  // Standardize
  StandardScaler scaler;
  Mat Xs = scaler.fit_transform(X);

  // --- Train via Gradient Descent ---
  LinearRegression lr;
  lr.set_hyperparams(
      /*lr*/ 0.05,
      /*iters*/ 3000,
      /*batch_size*/ 32,
      /*l2*/ 1e-4,
      /*shuffle*/ true,
      /*tol*/ 1e-9,
      /*patience*/ 50,
      /*verbose_every*/ 0);
  lr.fit(Xs, y);

  Vec yhat = lr.predict(Xs);
  double mse_gd = mse(y, yhat);
  std::cout << "[LR:GD] MSE = " << mse_gd << "\n";

  if (!(mse_gd < 0.2))
  {
    std::cerr << "[FAIL] LinearRegression GD MSE too high: " << mse_gd << "\n";
    return EXIT_FAILURE;
  }

  // Train via Closed-form Ridge Regression
  LinearRegression lr_cf;
  lr_cf.fit_closed_form(Xs, y, /*l2=*/1e-4);
  Vec yhat_cf = lr_cf.predict(Xs);
  double mse_cf = mse(y, yhat_cf);
  std::cout << "[LR:ClosedForm] MSE = " << mse_cf << "\n";

  if (!(mse_cf < 0.2))
  {
    std::cerr << "[FAIL] LinearRegression ClosedForm MSE too high: " << mse_cf << "\n";
    return EXIT_FAILURE;
  }

  // --- Check predict_scalar (1D helper) ---
  LinearRegression lr_demo{2.0, 1.0};
  double pred = lr_demo.predict_scalar(4.0);
  if (std::fabs(pred - 9.0) > 1e-9)
  {
    std::cerr << "[FAIL] predict_scalar() incorrect: got " << pred << " expected 9.0\n";
    return EXIT_FAILURE;
  }

  std::cout << "[OK] LinearRegression passed.\n";

  // 2. KMeans test : two 2D Gaussian blobs

  Mat U;
  std::normal_distribution<double> g1x(0.0, 0.5), g1y(0.0, 0.5);
  std::normal_distribution<double> g2x(3.0, 0.5), g2y(3.0, 0.5);

  for (int i = 0; i < 100; ++i)
    U.push_back({g1x(rng), g1y(rng)});
  for (int i = 0; i < 100; ++i)
    U.push_back({g2x(rng), g2y(rng)});

  KMeans km(2, 50);
  km.fit(U);

  const auto &C = km.centers();
  std::cout << "[KMeans] Centers:\n";
  for (const auto &c : C)
    std::cout << "  (" << c[0] << ", " << c[1] << ")\n";

  if (C.size() != 2 || C[0].size() != 2)
  {
    std::cerr << "[FAIL] KMeans centers shape invalid.\n";
    return EXIT_FAILURE;
  }

  // Check label for point near (0,0)
  double lbl = km.predict_one({0.0, 0.0});
  if (!(lbl == 0.0 || lbl == 1.0))
  {
    std::cerr << "[FAIL] KMeans invalid label: " << lbl << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[OK] KMeans passed.\n";
  std::cout << "=== [Vix.AI.ML] All tests passed. ===\n";
  return EXIT_SUCCESS;
}
