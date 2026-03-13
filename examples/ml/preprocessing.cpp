/**
 * @file preprocessing.cpp
 * @brief Demonstrate every preprocessing utility in Preprocessing.hpp.
 *
 * Covers:
 *   - StandardScaler  (fit / transform / fit_transform)
 *   - MinMaxScaler
 *   - MaxAbsScaler
 *   - normalize_l2
 *   - shuffle_dataset
 *   - train_test_split (free function)
 */

#include <vix/ai/ml/Preprocessing.hpp>

#include <iostream>
#include <iomanip>

// Helper: pretty-print a matrix
static void print_mat(const vix::ai::ml::Mat &M,
                      const std::string &label,
                      std::size_t max_rows = 5)
{
  std::cout << label << " (" << M.size() << " rows):\n";
  const std::size_t rows = std::min(M.size(), max_rows);
  for (std::size_t i = 0; i < rows; ++i)
  {
    std::cout << "  [";
    for (std::size_t j = 0; j < M[i].size(); ++j)
      std::cout << std::setw(8) << std::setprecision(4) << M[i][j]
                << (j + 1 < M[i].size() ? ", " : "");
    std::cout << "]\n";
  }
  if (M.size() > max_rows)
    std::cout << "  ... (" << M.size() - max_rows << " more rows)\n";
  std::cout << "\n";
}

int main()
{
  using namespace vix::ai::ml;

  // Raw data:  3 features, deliberately different scales
  //   col 0 ∈ [1, 4]   (small positive)
  //   col 1 ∈ [-100, 100]  (large, centred)
  //   col 2 ∈ [0, 0]   (constant — stress test for zero-variance)
  Mat X = {
      {1.0, -100.0, 0.0},
      {2.0, -50.0, 0.0},
      {3.0, 0.0, 0.0},
      {4.0, 50.0, 0.0},
      {2.5, 100.0, 0.0},
      {1.5, -75.0, 0.0},
      {3.5, 75.0, 0.0},
      {2.0, 25.0, 0.0},
  };
  Vec y = {0, 1, 0, 1, 1, 0, 1, 0};

  print_mat(X, "Raw X");

  // 1. StandardScaler
  std::cout << "=== StandardScaler ===\n";
  StandardScaler ss;
  Mat X_std = ss.fit_transform(X);

  std::cout << "  Means : [";
  for (double m : ss.mean)
    std::cout << std::setprecision(4) << m << " ";
  std::cout << "]\n";
  std::cout << "  Stds  : [";
  for (double s : ss.std)
    std::cout << std::setprecision(4) << s << " ";
  std::cout << "]\n  (col 2 std=1.0 because it is constant)\n\n";

  print_mat(X_std, "Standardised X");

  // 2. MinMaxScaler
  std::cout << "=== MinMaxScaler ===\n";
  MinMaxScaler mm;
  Mat X_mm = mm.fit_transform(X);
  print_mat(X_mm, "MinMax-scaled X");

  // 3. MaxAbsScaler
  std::cout << "=== MaxAbsScaler ===\n";
  MaxAbsScaler ma;
  Mat X_ma = ma.fit_transform(X);
  std::cout << "  MaxAbs per feature: [";
  for (double v : ma.maxabs)
    std::cout << v << " ";
  std::cout << "]\n";
  print_mat(X_ma, "MaxAbs-scaled X");

  // 4. normalize_l2
  std::cout << "=== normalize_l2 ===\n";
  // Use a matrix without the constant column for clarity
  Mat X2 = {
      {3.0, 4.0},
      {1.0, 0.0},
      {0.0, 0.0}, // zero-row — should be left unchanged
      {-6.0, 8.0},
  };
  Mat X2_norm = normalize_l2(X2);
  print_mat(X2_norm, "L2-normalised rows (norms should be 1, except zero-row)");

  // 5. shuffle_dataset (in-place)
  std::cout << "=== shuffle_dataset ===\n";
  Mat Xs = X;
  Vec ys = y;
  std::cout << "  Before shuffle y: [";
  for (double v : ys)
    std::cout << v << " ";
  std::cout << "]\n";

  shuffle_dataset(Xs, ys, 42);

  std::cout << "  After  shuffle y: [";
  for (double v : ys)
    std::cout << v << " ";
  std::cout << "]\n\n";

  // 6. train_test_split (free function)
  std::cout << "=== train_test_split ===\n";
  auto [X_tr, X_te, y_tr, y_te] = train_test_split(X, y, 0.25, true, 42);

  std::cout << "  Train size : " << X_tr.size() << "\n";
  std::cout << "  Test  size : " << X_te.size() << "\n";
  std::cout << "  Train y    : [";
  for (double v : y_tr)
    std::cout << v << " ";
  std::cout << "]\n";
  std::cout << "  Test  y    : [";
  for (double v : y_te)
    std::cout << v << " ";
  std::cout << "]\n";

  return 0;
}
