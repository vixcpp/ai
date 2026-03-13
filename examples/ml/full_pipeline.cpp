/**
 * @file full_pipeline.cpp
 * @brief Full end-to-end ML pipeline combining every module of Vix.AI.ML.
 *
 * Pipeline:
 *   Dataset  →  Preprocessing  →  Model  →  Metrics
 *
 * Steps:
 *   1. Generate a synthetic regression dataset and save it to CSV
 *   2. Load it back through Dataset::from_csv
 *   3. Split into train / test with Dataset::train_test_split
 *   4. Scale features with StandardScaler (no leakage)
 *   5. Train LinearRegression (closed-form) and compare with KMeans
 *      used as a sanity check on feature structure
 *   6. Evaluate with mse / rmse / r2_score
 *   7. Save both models and reload them
 *   8. Print a final summary table
 */

#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Dataset.hpp>
#include <vix/ai/ml/Metrics.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Regression.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// Synthetic dataset:  y = 2·x₀ - x₁ + 0.5·x₂ + ε
static void generate_csv(const std::string &path, std::size_t n = 80)
{
  std::ofstream f(path);
  f << "x0,x1,x2,y\n";
  f << std::setprecision(8);

  unsigned long s = 271828;
  auto rnd = [&]() -> double
  {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((s >> 33) & 0x7FFFFFFF) / 2147483647.0;
  };
  auto gauss = [&](double sigma) -> double
  {
    const double u = rnd() + 1e-15, v = rnd();
    return sigma * std::sqrt(-2.0 * std::log(u)) * std::cos(6.28318 * v);
  };

  for (std::size_t i = 0; i < n; ++i)
  {
    const double x0 = rnd() * 6.0 - 3.0;
    const double x1 = rnd() * 6.0 - 3.0;
    const double x2 = rnd() * 6.0 - 3.0;
    const double y = 2.0 * x0 - x1 + 0.5 * x2 + gauss(0.3);
    f << x0 << "," << x1 << "," << x2 << "," << y << "\n";
  }
}

// Pretty-print a metrics summary row
static void print_row(const std::string &name,
                      double mse_v, double rmse_v,
                      double mae_v, double r2_v)
{
  std::cout << std::left << std::setw(22) << name
            << std::right << std::setw(10) << std::setprecision(5) << mse_v
            << std::setw(10) << rmse_v
            << std::setw(10) << mae_v
            << std::setw(10) << r2_v << "\n";
}

int main()
{
  using namespace vix::ai::ml;

  const std::string csv_path = "/tmp/vix_pipeline.csv";

  std::cout << "=== Vix.AI.ML — Full Pipeline Example ===\n\n";

  // 1. Generate + load dataset
  generate_csv(csv_path, 80);

  auto opt = Dataset::from_csv(csv_path, /*has_header=*/true, /*target_col=*/3);
  if (!opt)
  {
    std::cerr << "Failed to load CSV\n";
    return 1;
  }
  Dataset ds = std::move(*opt);

  std::cout << "Dataset : " << ds.size_supervised()
            << " samples, " << ds.n_features() << " features\n\n";

  // 2. Split
  auto [train, test] = ds.train_test_split(0.20, 42);
  std::cout << "Split   : " << train.size_supervised() << " train / "
            << test.size_supervised() << " test\n\n";

  // 3. Preprocessing: StandardScaler (fit on train only)
  StandardScaler scaler;
  Mat X_train = scaler.fit_transform(train.X);
  Mat X_test = scaler.transform(test.X);

  std::cout << "Scaler  : means=[";
  for (double m : scaler.mean)
    std::cout << std::setprecision(3) << m << " ";
  std::cout << "]  stds=[";
  for (double s : scaler.std)
    std::cout << std::setprecision(3) << s << " ";
  std::cout << "]\n\n";

  // 4a. LinearRegression — gradient descent
  LinearRegression lr_gd;
  lr_gd.set_hyperparams(0.05, 3000, 0, 1e-5, true, 1e-10, 50);
  lr_gd.fit(X_train, train.y);
  Vec pred_gd = lr_gd.predict(X_test);

  // 4b. LinearRegression — closed form
  LinearRegression lr_cf;
  lr_cf.fit_closed_form(X_train, train.y, 1e-5);
  Vec pred_cf = lr_cf.predict(X_test);

  // 4c. LogisticRegression (wrong tool for regression — shows R² near 0)
  LogisticRegression clf;
  clf.set_hyperparams(0.05, 300);
  clf.fit(X_train, train.y);
  Vec pred_clf = clf.predict(X_test);

  // 5. Metrics summary
  std::cout << std::left << std::setw(22) << "Model"
            << std::right << std::setw(10) << "MSE"
            << std::setw(10) << "RMSE"
            << std::setw(10) << "MAE"
            << std::setw(10) << "R²" << "\n";
  std::cout << std::string(62, '-') << "\n";

  print_row("LinearReg (GD)",
            mse(test.y, pred_gd), rmse(test.y, pred_gd),
            mae(test.y, pred_gd), r2_score(test.y, pred_gd));

  print_row("LinearReg (CF)",
            mse(test.y, pred_cf), rmse(test.y, pred_cf),
            mae(test.y, pred_cf), r2_score(test.y, pred_cf));

  print_row("LogisticReg (wrong)",
            mse(test.y, pred_clf), rmse(test.y, pred_clf),
            mae(test.y, pred_clf), r2_score(test.y, pred_clf));

  // 6. KMeans on features (unsupervised sanity check)
  std::cout << "\n--- KMeans inertia (k=3, k-means++) ---\n";
  KMeans km(3, 300, 42, true);
  km.fit(X_train);
  std::cout << "  Train inertia : " << km.inertia(X_train) << "\n";
  std::cout << "  Test  inertia : " << km.inertia(X_test) << "\n";

  // 7. Save / reload both regression models
  std::cout << "\n--- Save/load verification ---\n";
  {
    std::ostringstream oss_gd, oss_cf;
    lr_gd.save(oss_gd);
    lr_cf.save(oss_cf);

    LinearRegression lr_gd2, lr_cf2;
    {
      std::istringstream iss(oss_gd.str());
      lr_gd2.load(iss);
    }
    {
      std::istringstream iss(oss_cf.str());
      lr_cf2.load(iss);
    }

    const double r2_gd2 = r2_score(test.y, lr_gd2.predict(X_test));
    const double r2_cf2 = r2_score(test.y, lr_cf2.predict(X_test));
    std::cout << "  GD  R² after reload : " << r2_gd2 << "\n";
    std::cout << "  CF  R² after reload : " << r2_cf2 << "\n";
  }

  std::remove(csv_path.c_str());
  return 0;
}
