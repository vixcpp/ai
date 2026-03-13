/**
 * @file kmeans_clustering.cpp
 * @brief Unsupervised clustering with KMeans (Lloyd's + k-means++) and
 *        MiniBatchKMeans on a synthetic three-cluster dataset.
 *
 * Demonstrates:
 *   - Dataset unsupervised mode (U)
 *   - MaxAbsScaler
 *   - KMeans with random init vs k-means++ init
 *   - MiniBatchKMeans
 *   - inertia (WCSS) comparison
 *   - Model::save + Model::load round-trip
 */

#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Dataset.hpp>
#include <vix/ai/ml/Preprocessing.hpp>

#include <iostream>
#include <sstream>

// Three well-separated Gaussian blobs in 2D
// Centres: (-3,0), (0,+3), (+3,0)
static vix::ai::ml::Dataset make_clusters()
{
  using namespace vix::ai::ml;

  const std::size_t PER_CLUSTER = 40;
  const std::size_t N = 3 * PER_CLUSTER;

  const double cx[3] = {-3.0, 0.0, +3.0};
  const double cy[3] = {0.0, +3.0, 0.0};

  Mat U(N, Vec(2));

  unsigned long s = 314159;
  auto rnd = [&]() -> double
  {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((s >> 33) & 0x7FFFFFFF) / 2147483647.0;
  };
  auto gauss = [&](double sigma) -> double
  {
    double u = rnd() + 1e-15, v = rnd();
    return sigma * std::sqrt(-2.0 * std::log(u)) * std::cos(6.28318 * v);
  };

  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < PER_CLUSTER; ++i)
    {
      const std::size_t idx = c * PER_CLUSTER + i;
      U[idx][0] = cx[c] + gauss(0.5);
      U[idx][1] = cy[c] + gauss(0.5);
    }

  Dataset ds;
  ds.U = std::move(U);
  return ds;
}

int main()
{
  using namespace vix::ai::ml;

  std::cout << "=== KMeans Clustering Example ===\n\n";

  // 1. Data + scale
  Dataset ds = make_clusters();
  std::cout << "Samples   : " << ds.size_unsupervised() << "\n";
  std::cout << "Features  : " << ds.n_features() << "\n\n";

  MaxAbsScaler scaler;
  Mat U_scaled = scaler.fit_transform(ds.U);

  // 2. KMeans — random initialisation
  KMeans km_rand(3, 300, 42, /*use_kmeanspp=*/false);
  km_rand.fit(U_scaled);

  std::cout << "--- KMeans (random init) ---\n";
  std::cout << "  Inertia : " << km_rand.inertia(U_scaled) << "\n";
  std::cout << "  Centers :\n";
  for (const auto &c : km_rand.centers())
    std::cout << "    [" << c[0] << ", " << c[1] << "]\n";

  // 3. KMeans — k-means++ initialisation
  KMeans km_pp(3, 300, 42, /*use_kmeanspp=*/true);
  km_pp.fit(U_scaled);

  std::cout << "\n--- KMeans (k-means++) ---\n";
  std::cout << "  Inertia : " << km_pp.inertia(U_scaled) << "\n";
  std::cout << "  Centers :\n";
  for (const auto &c : km_pp.centers())
    std::cout << "    [" << c[0] << ", " << c[1] << "]\n";

  // 4. MiniBatchKMeans
  MiniBatchKMeans mb(3, /*batch_size=*/20, /*max_iters=*/200, /*seed=*/7);
  mb.fit(U_scaled);

  std::cout << "\n--- MiniBatchKMeans ---\n";
  std::cout << "  Inertia : " << mb.inertia(U_scaled) << "\n";
  std::cout << "  Centers :\n";
  for (const auto &c : mb.centers())
    std::cout << "    [" << c[0] << ", " << c[1] << "]\n";

  // 5. Per-sample labels (first 12)
  Idxs labels = km_pp.predict_labels(U_scaled);
  std::cout << "\n--- Labels for first 12 samples (k-means++) ---\n  ";
  for (std::size_t i = 0; i < 12; ++i)
    std::cout << labels[i] << " ";
  std::cout << "\n";

  // 6. Elbow method: inertia for k = 1..5
  std::cout << "\n--- Elbow curve ---\n";
  for (std::size_t k = 1; k <= 5; ++k)
  {
    KMeans km_k(k, 300, 42, true);
    km_k.fit(U_scaled);
    std::cout << "  k=" << k
              << "  inertia=" << km_k.inertia(U_scaled) << "\n";
  }

  // 7. Save / load round-trip
  std::ostringstream oss;
  km_pp.save(oss);

  KMeans km_loaded(3);
  std::istringstream iss(oss.str());
  km_loaded.load(iss);

  std::cout << "\n--- After save/load ---\n";
  std::cout << "  Inertia (should match k-means++): "
            << km_loaded.inertia(U_scaled) << "\n";

  return 0;
}
