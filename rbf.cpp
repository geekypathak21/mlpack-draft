#include <mlpack/core.hpp>
#include <chrono>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::kmeans;
using namespace std::chrono;

/**
 * Train the vanilla network on a larger dataset.
 */
int main()
{
  // Load the dataset.
  arma::mat trainData;
  data::Load("/home/himanshu/Desktop/mnist/mnist-dataset/mnist_train.csv", trainData, true);

  arma::mat all;
  data::Load("/home/himanshu/Desktop/mnist/mnist-dataset/mnist_all.csv", all, true);
  all.shed_row(0);

  arma::mat trainLabels = trainData.row(0);
  trainData.shed_row(0);

  arma::mat testData;
  data::Load("/home/himanshu/Desktop/mnist/mnist-dataset/mnist_test.csv", testData, true);

  // Normalize each point since these are images.
  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    trainData.col(i) /= norm(trainData.col(i), 2);
  }

  for (size_t i = 0; i < all.n_cols; ++i)
  {
    all.col(i) /= norm(all.col(i), 2);
  }

  arma::mat testLabels = testData.row(0);
  testData.shed_row(0);

  // Normalize each point since these are images.
  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    testData.col(i) /= norm(testData.col(i), 2);
  }
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         RBF          Activation    Output
   *  Layer         Layer         Layer        Layer
   * +-----+       +-----+       +-----+       +-----+
   * |     |       |     |       |     |       |     |
   * |     +------>|     +------>|     +------>|     |
   * |     |       |     |       |     |       |     |
   * +-----+       +--+--+       +-----+       +-----+
   */

  auto start = high_resolution_clock::now(); 
  arma::mat centroids1;
  arma::Row<size_t> assignments;
  KMeans<> kmeans1;
  kmeans1.Cluster(all, 1000, assignments, centroids1);
  centroids1.save("centroids1.arm");

  FFN<NegativeLogLikelihood<> > model;
  model.Add<RBF<> >(trainData.n_rows, 1000, centroids1);
  model.Add<Linear<> >(1000, 10);
  model.Add<LogSoftMax<> >();

  ens::RMSProp opt(0.01, 32, 0.88, 1e-8, 10 * trainData.n_cols, -1);
  model.Train(trainData, trainLabels, opt);

  // After function call 
  auto stop = high_resolution_clock::now(); 

  arma::mat predictionTemp;
  model.Predict(testData, predictionTemp);
  arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  auto duration = duration_cast<microseconds>(stop - start);

  size_t correct = arma::accu(prediction == testLabels);
  double classificationError = 1 - double(correct) / testData.n_cols;
  std::cout<<classificationError<<std::endl;
  std::cout<<duration.count()<<std::endl; 
}