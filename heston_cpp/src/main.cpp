#include <iostream>

#include "payoff.h"
#include "option.h"
#include "correlated_snd.h"
#include "heston_mc.h"

void generate_normal_correlation_paths(double rho,
    std::vector<double>& spot_normals, std::vector<double>& cor_normals) {
  unsigned vals = spot_normals.size();

  // Create the Standard Normal Distribution and random draw vectors
  StandardNormalDistribution snd;
  std::vector<double> snd_uniform_draws(vals, 0.0);

  // Simple random number generation method based on RAND
  for (uint i=0; i<snd_uniform_draws.size(); i++) {
    snd_uniform_draws[i] = rand() / static_cast<double>(RAND_MAX);
  }

  // Create standard normal random draws
  snd.random_draws(snd_uniform_draws, spot_normals);

  // Create the correlated standard normal distribution
  CorrelatedSND csnd(rho, &spot_normals);
  std::vector<double> csnd_uniform_draws(vals, 0.0);

  // Uniform generation for the correlated SND
  for (uint i=0; i<csnd_uniform_draws.size(); i++) {
    csnd_uniform_draws[i] = rand() / static_cast<double>(RAND_MAX);
  }

  // Now create the -correlated- standard normal draw series
  csnd.random_draws(csnd_uniform_draws, cor_normals);
}

int main() {
  // First we create the parameter list
  // Note that you could easily modify this code to input the parameters
  // either from the command line or via a file
  unsigned num_sims = 100000;   // Number of simulated asset paths
  unsigned num_intervals = 252;  // Number of intervals for the asset path to be sampled

  double S_0 = 400.38;    // Initial spot price
  double K = 400.38;      // Strike price
  double r = 0.0329;     // Risk-free rate
  double v_0 = 0.010201; // Initial volatility
  double T = 1.00;       // One year until expiry

  double rho = -0.52;     // Correlation of asset and volatility
  double kappa = 1.65;   // Mean-reversion rate
  double theta = 0.11;  // Long run average volatility
  double xi = 1.00;      // "Vol of vol"

  // Create the PayOff, Option and Heston objects
  PayOff* pPayOffCall = new PayOffCall(K);
  Option* pOption = new Option(K, r, T, pPayOffCall);
  HestonEuler hest_euler(pOption, kappa, theta, xi, rho);

  // Create the spot and vol initial normal and price paths
  std::vector<double> spot_draws(num_intervals, 0.0);  // Vector of initial spot normal draws
  std::vector<double> vol_draws(num_intervals, 0.0);   // Vector of initial correlated vol normal draws
  std::vector<double> spot_prices(num_intervals, S_0);  // Vector of initial spot prices
  std::vector<double> vol_prices(num_intervals, v_0);   // Vector of initial vol prices

  // Monte Carlo options pricing
  double payoff_sum = 0.0;
  for (unsigned i=0; i<num_sims; i++) {
    // std::cout << "Calculating path " << i+1 << " of " << num_sims << std::endl;
    generate_normal_correlation_paths(rho, spot_draws, vol_draws);
    hest_euler.calc_vol_path(vol_draws, vol_prices);
    hest_euler.calc_spot_path(spot_draws, vol_prices, spot_prices);
    payoff_sum += pOption->pay_off->operator()(spot_prices[num_intervals-1]);
  }
  double option_price = (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T);
  std::cout << "Option Price: " << option_price << std::endl;

  // Free memory
  delete pOption;
  delete pPayOffCall;

  return 0;
}
