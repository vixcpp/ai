/**
 * @file logistic_regression.cpp
 * @brief End-to-end example: binary classification with LogisticRegression.
 *
 * Demonstrates:
 *   - Synthetic linearly-separable dataset (two Gaussian blobs)
 *   - MinMaxScaler
 *   - LogisticRegression::set_hyperparams + fit
 *   - predict (probabilities) vs predict_class (hard labels)
 *   - accuracy01 / precision01 / recall01 / f1_score01
 *   - Model::save + Model::load round-trip
 */

#include <vix/ai/ml/Dataset.hpp>
#include <vix/ai/ml/Metrics.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Regression.hpp>

#include <iostream>
#include <sstream>

// Two Gaussian blobs centred at (-1,-1) [class 0] and (+1,+1) [class 1]
static vix::ai::ml::Dataset make_blobs()
{
  using namespace vix::ai::ml;

  const std::size_t N = 60; // 30 per class
  Mat X(N, Vec(2));
  Vec y(N);

  unsigned long s = 99991;
  auto rnd = [&]() -> double { // uniform [0,1]
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((s >> 33) & 0x7FFFFFFF) / 2147483647.0;
  };
  // Box-Muller for Gaussian noise
  auto gauss = [&](double sigma) -> double
  {
    double u = rnd() + 1e-15, v = rnd();
    return sigma * std::sqrt(-2.0 * std::log(u)) * std::cos(6.28318 * v);
  };

  for (std::size_t i = 0; i < N; ++i)
  {
    const double center = (i < N / 2) ? -1.0 : +1.0;
    X[i][0] = center + gauss(0.6);
    X[i][1] = center + gauss(0.6);
    y[i] = (i < N / 2) ? 0.0 : 1.0;
  }

  Dataset ds;
  ds.X = std::move(X);
  ds.y = std::move(y);
  return ds;
}

int main()
{
  using namespace vix::ai::ml;

  std::cout << "=== Logistic Regression Example ===\n\n";

  // 1. Data + split
  Dataset ds = make_blobs();
  auto [train, test] = ds.train_test_split(0.25, 7);

  std::cout << "Train samples : " << train.size_supervised() << "\n";
  std::cout << "Test  samples : " << test.size_supervised() << "\n\n";

  // 2. Scale features to [0, 1]
  MinMaxScaler scaler;
  Mat X_train = scaler.fit_transform(train.X);
  Mat X_test = scaler.transform(test.X);

  // 3. Train
  LogisticRegression clf;
  clf.set_hyperparams(
      /*lr*/ 0.1,
      /*iters*/ 500,
      /*batch_size*/ 16,
      /*l2*/ 1e-3,
      /*shuffle*/ true,
      /*tol*/ 1e-8,
      /*patience*/ 30,
      /*verbose_every*/ 100);
  clf.fit(X_train, train.y);

  // 4. Evaluate
  Vec probs = clf.predict(X_test); // probabilities

  // Convert probabilities to hard labels for metric functions
  Vec labels(probs.size());
  for (std::size_t i = 0; i < probs.size(); ++i)
    labels[i] = clf.predict_class(X_test[i]);

  std::cout << "\n--- Test-set metrics ---\n";
  std::cout << "  Accuracy  : " << accuracy01(test.y, probs) << "\n";
  std::cout << "  Precision : " << precision01(test.y, probs) << "\n";
  std::cout << "  Recall    : " << recall01(test.y, probs) << "\n";
  std::cout << "  F1 Score  : " << f1_score01(test.y, probs) << "\n";

  // 5. Show per-sample predictions
  std::cout << "\n--- Per-sample predictions (first 10) ---\n";
  std::cout << "  idx | true | prob   | pred\n";
  std::cout << "  ----|------|--------|-----\n";
  for (std::size_t i = 0; i < std::min(std::size_t{10}, test.size_supervised()); ++i)
  {
    std::cout << "   " << i
              << "  |  " << static_cast<int>(test.y[i])
              << "   | " << probs[i]
              << "  |  " << static_cast<int>(labels[i]) << "\n";
  }

  // 6. Save / load
  std::ostringstream oss;
  clf.save(oss);

  LogisticRegression clf2;
  std::istringstream iss(oss.str());
  clf2.load(iss);

  Vec probs2 = clf2.predict(X_test);
  std::cout << "\n--- After save/load ---\n";
  std::cout << "  Accuracy (should match): "
            << accuracy01(test.y, probs2) << "\n";

  return 0;
}
