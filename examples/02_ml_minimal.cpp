/**
 * @file 02_ml_minimal.cpp
 * @brief Minimal demo for Vix.AI.ML LinearRegression API:
 *        - train a model via fit(X,y)
 *        - use the 1D convenience ctor (a,b) + predict_scalar(x)
 */

#include <iostream>
#include <vix/ai/tensor/Engine.hpp>
#include <vix/ai/ml/Dataset.hpp>
#include <vix/ai/ml/Regression.hpp>

int main()
{
  using namespace vix::ai::tensor;
  using namespace vix::ai::ml;

  // Optional tensor demo
  Engine engine{Device{}};
  std::cout << engine.compute({}) << " ";

  // Supervised tiny dataset: y ≈ 2x + 1
  Dataset ds;
  ds.X = {{0.0}, {1.0}, {2.0}, {3.0}}; // one feature column
  ds.y = {1.0, 3.0, 5.0, 7.0};         // targets

  std::cout << "dataset size=" << ds.size_supervised() << " ";

  // 1) Trainable model (gradient descent)
  LinearRegression lr;           // default ctor (no weights yet)
  lr.set_hyperparams(0.1, 3000); // lr, iterations
  lr.fit(ds.X, ds.y);            // train
  Mat Xtest{{4.0}};
  std::cout << "predict(4)=" << lr.predict(Xtest)[0] << " (attendu ~9)\n";

  // 2) Convenience 1D model (y ≈ a*x + b) for quick demos
  LinearRegression lr2{2.0, 1.0}; // initializes w_={2}, b_=1
  std::cout << "predict(4)=" << lr2.predict_scalar(4.0) << " (attendu ~9)\n";

  return 0;
}
