/**
 * @file linear_regression.cpp
 * @brief End-to-end example: train a LinearRegression model on synthetic data,
 *        evaluate it with regression metrics, and save / reload the model.
 *
 * Demonstrates:
 *   - StandardScaler (fit on train, transform both splits)
 *   - LinearRegression::set_hyperparams + fit (gradient descent)
 *   - LinearRegression::fit_closed_form (Normal Equation)
 *   - mse / rmse / mae / r2_score
 *   - Model::save + Model::load round-trip
 */

#include <vix/ai/ml/Dataset.hpp>
#include <vix/ai/ml/Metrics.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Regression.hpp>

#include <iostream>
#include <sstream>

static vix::ai::ml::Dataset make_dataset()
{
  using namespace vix::ai::ml;

  // 40 samples, 2 features
  const std::size_t N = 40;
  Mat X(N, Vec(2));
  Vec y(N);

  // Deterministic pseudo-random generator (LCG, no <random> needed here)
  unsigned long state = 12345;
  auto lcg = [&]() -> double
  {
    state = state * 1664525UL + 1013904223UL;
    return static_cast<double>(state & 0xFFFF) / 65535.0; // [0,1]
  };

  for (std::size_t i = 0; i < N; ++i)
  {
    X[i][0] = lcg() * 4.0 - 2.0; // x₀ ∈ [-2, 2]
    X[i][1] = lcg() * 4.0 - 2.0; // x₁ ∈ [-2, 2]
    const double noise = (lcg() - 0.5) * 0.4;
    y[i] = 3.0 * X[i][0] - 2.0 * X[i][1] + 1.0 + noise;
  }

  Dataset ds;
  ds.X = std::move(X);
  ds.y = std::move(y);
  return ds;
}

int main()
{
  using namespace vix::ai::ml;

  std::cout << "=== Linear Regression Example ===\n\n";

  // 1. Load data and split
  Dataset ds = make_dataset();
  auto [train, test] = ds.train_test_split(0.25, 42);

  std::cout << "Train samples : " << train.size_supervised() << "\n";
  std::cout << "Test  samples : " << test.size_supervised() << "\n\n";

  // 2. Standardise features
  //    Fit only on training data to avoid data leakage.
  StandardScaler scaler;
  Mat X_train = scaler.fit_transform(train.X);
  Mat X_test = scaler.transform(test.X);

  // 3a. Gradient descent training
  LinearRegression lr_gd;
  lr_gd.set_hyperparams(
      /*lr*/ 0.05,
      /*iters*/ 2000,
      /*batch_size*/ 0, // full-batch
      /*l2*/ 1e-4,
      /*shuffle*/ true,
      /*tol*/ 1e-9,
      /*patience*/ 50);
  lr_gd.fit(X_train, train.y);

  Vec pred_gd = lr_gd.predict(X_test);

  std::cout << "--- Gradient Descent ---\n";
  std::cout << "  weights : [";
  for (double w : lr_gd.weights())
    std::cout << " " << w;
  std::cout << " ]\n";
  std::cout << "  bias    : " << lr_gd.bias() << "\n";
  std::cout << "  MSE     : " << mse(test.y, pred_gd) << "\n";
  std::cout << "  RMSE    : " << rmse(test.y, pred_gd) << "\n";
  std::cout << "  MAE     : " << mae(test.y, pred_gd) << "\n";
  std::cout << "  R²      : " << r2_score(test.y, pred_gd) << "\n\n";

  // 3b. Closed-form (Normal Equation) training
  LinearRegression lr_cf;
  lr_cf.fit_closed_form(X_train, train.y, /*l2=*/1e-4);

  Vec pred_cf = lr_cf.predict(X_test);

  std::cout << "--- Closed Form (Normal Equation) ---\n";
  std::cout << "  weights : [";
  for (double w : lr_cf.weights())
    std::cout << " " << w;
  std::cout << " ]\n";
  std::cout << "  bias    : " << lr_cf.bias() << "\n";
  std::cout << "  MSE     : " << mse(test.y, pred_cf) << "\n";
  std::cout << "  RMSE    : " << rmse(test.y, pred_cf) << "\n";
  std::cout << "  MAE     : " << mae(test.y, pred_cf) << "\n";
  std::cout << "  R²      : " << r2_score(test.y, pred_cf) << "\n\n";

  // 4. Save / load round-trip
  std::ostringstream oss;
  lr_cf.save(oss);

  LinearRegression lr_loaded;
  std::istringstream iss(oss.str());
  lr_loaded.load(iss);

  Vec pred_loaded = lr_loaded.predict(X_test);
  std::cout << "--- After save/load ---\n";
  std::cout << "  R² (should match CF): " << r2_score(test.y, pred_loaded) << "\n";

  return 0;
}
