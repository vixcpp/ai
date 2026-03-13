/**
 * @file dataset_csv.cpp
 * @brief Demonstrate the Dataset utility: CSV generation, loading, slicing,
 *        shuffling, train_test_split, and to_csv.
 *
 * The example writes a synthetic CSV to a temporary file, then exercises
 * every Dataset method.
 */

#include <vix/ai/ml/Dataset.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>

// Write a small numeric CSV to disk so we can demonstrate from_csv.
// Format:  f0, f1, f2, target
static void write_sample_csv(const std::string &path)
{
  std::ofstream f(path);
  f << "feat0,feat1,feat2,label\n"; // header row
  // 20 rows: label = 1 when feat0 > 0, else 0
  const double rows[20][4] = {
      {1.2, 0.5, 3.1, 1.0},
      {-0.3, 1.2, 2.7, 0.0},
      {2.1, -0.4, 1.8, 1.0},
      {-1.5, 0.8, 4.2, 0.0},
      {0.7, 2.3, 0.9, 1.0},
      {-2.0, -1.1, 2.5, 0.0},
      {3.0, 0.0, 1.0, 1.0},
      {-0.5, 0.5, 3.8, 0.0},
      {1.0, 1.0, 1.0, 1.0},
      {-1.0, -1.0, -1.0, 0.0},
      {0.1, 0.2, 0.3, 1.0},
      {-0.1, -0.2, -0.3, 0.0},
      {4.0, 2.0, 0.5, 1.0},
      {-3.0, -2.0, 1.5, 0.0},
      {0.9, -0.9, 2.0, 1.0},
      {-0.9, 0.9, -2.0, 0.0},
      {2.5, 1.5, 0.0, 1.0},
      {-2.5, -1.5, 0.0, 0.0},
      {0.5, 0.5, 0.5, 1.0},
      {-0.5, -0.5, -0.5, 0.0},
  };
  f << std::setprecision(6);
  for (const auto &r : rows)
    f << r[0] << "," << r[1] << "," << r[2] << "," << r[3] << "\n";
}

int main()
{
  using namespace vix::ai::ml;

  const std::string csv_path = "/tmp/vix_sample.csv";
  const std::string out_path = "/tmp/vix_train.csv";

  // 1. Write + load supervised CSV  (target = last column, index 3)
  write_sample_csv(csv_path);

  auto opt = Dataset::from_csv(csv_path, /*has_header=*/true, /*target_col=*/3);
  if (!opt)
  {
    std::cerr << "ERROR: could not load " << csv_path << "\n";
    return 1;
  }
  Dataset ds = std::move(*opt);

  std::cout << "=== Dataset loaded ===\n";
  std::cout << "  is_supervised   : " << ds.is_supervised() << "\n";
  std::cout << "  is_unsupervised : " << ds.is_unsupervised() << "\n";
  std::cout << "  size_supervised : " << ds.size_supervised() << "\n";
  std::cout << "  n_features      : " << ds.n_features() << "\n\n";

  // 2. Load as unsupervised (no target_col)
  auto opt_u = Dataset::from_csv(csv_path, /*has_header=*/true, /*target_col=*/-1);
  if (opt_u)
  {
    std::cout << "=== Unsupervised reload ===\n";
    std::cout << "  size_unsupervised : " << opt_u->size_unsupervised() << "\n";
    std::cout << "  n_features        : " << opt_u->n_features() << "\n\n";
  }

  // 3. slice
  Dataset sliced = ds.slice(0, 5);
  std::cout << "=== slice [0, 5) ===\n";
  std::cout << "  size : " << sliced.size_supervised() << "\n";
  std::cout << "  y    : [";
  for (double v : sliced.y)
    std::cout << v << " ";
  std::cout << "]\n\n";

  // 4. shuffle
  Dataset shuffled = ds.shuffle(123);
  std::cout << "=== shuffle (seed=123) ===\n";
  std::cout << "  y before : [";
  for (double v : ds.y)
    std::cout << v << " ";
  std::cout << "]\n";
  std::cout << "  y after  : [";
  for (double v : shuffled.y)
    std::cout << v << " ";
  std::cout << "]\n\n";

  // 5. train_test_split
  auto [train, test] = ds.train_test_split(0.25, 42);

  std::cout << "=== train_test_split (25% test) ===\n";
  std::cout << "  train size : " << train.size_supervised() << "\n";
  std::cout << "  test  size : " << test.size_supervised() << "\n";
  std::cout << "  train y    : [";
  for (double v : train.y)
    std::cout << v << " ";
  std::cout << "]\n";
  std::cout << "  test  y    : [";
  for (double v : test.y)
    std::cout << v << " ";
  std::cout << "]\n\n";

  // 6. to_csv
  if (train.to_csv(out_path))
    std::cout << "=== to_csv ===\n  Train set saved to: " << out_path << "\n";
  else
    std::cerr << "  ERROR: could not write " << out_path << "\n";

  // Verify reload
  auto reloaded = Dataset::from_csv(out_path, /*has_header=*/false, /*target_col=*/3);
  if (reloaded)
    std::cout << "  Reloaded size : " << reloaded->size_supervised()
              << " (matches train: " << train.size_supervised() << ")\n";

  // Cleanup temp files
  std::remove(csv_path.c_str());
  std::remove(out_path.c_str());

  return 0;
}
