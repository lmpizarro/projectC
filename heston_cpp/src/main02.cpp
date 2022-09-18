#include "statistics.h"
#include "correlated_snd.h"
#include <iostream>
#include <vector>

int __main__() {

  // Number of values
  int vals = 30;

  /* UNCORRELATED SND */
  /* ================ */

  // Create the Standard Normal Distribution and random draw vectors
  StandardNormalDistribution snd;
  std::vector<double> snd_uniform_draws(vals, 0.0);
  std::vector<double> snd_normal_draws(vals, 0.0);

  // Simple random number generation method based on RAND
  // We could be more sophisticated an use a LCG or Mersenne Twister
  // but we're trying to demonstrate correlation, not efficient
  // random number generation!
  for (uint i=0; i<snd_uniform_draws.size(); i++) {
    snd_uniform_draws[i] = rand() / static_cast<double>(RAND_MAX);
  }

  // Create standard normal random draws
  snd.random_draws(snd_uniform_draws, snd_normal_draws);

  /* CORRELATION SND */
  /* =============== */

  // Correlation coefficient
  double rho = 0.5;

  // Create the correlated standard normal distribution
  CorrelatedSND csnd(rho, &snd_normal_draws);
  std::vector<double> csnd_uniform_draws(vals, 0.0);
  std::vector<double> csnd_normal_draws(vals, 0.0);

  // Uniform generation for the correlated SND
  for (uint i=0; i<csnd_uniform_draws.size(); i++) {
    csnd_uniform_draws[i] = rand() / static_cast<double>(RAND_MAX);
  }

  // Now create the -correlated- standard normal draw series
  csnd.random_draws(csnd_uniform_draws, csnd_normal_draws);

  // Output the values of the standard normal random draws
  for (uint i=0; i<snd_normal_draws.size(); i++) {
    std::cout << snd_normal_draws[i] << ", " << csnd_normal_draws[i] << std::endl;
  }

  return 0;
}