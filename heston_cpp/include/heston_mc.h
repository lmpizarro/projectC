#ifndef __HESTON_MC_H
#define __HESTON_MC_H

#include <cmath>
#include <vector>
#include "option.h"

// The HestonEuler class stores the necessary information
// for creating the volatility and spot paths based on the
// Heston Stochastic Volatility model.
class HestonEuler {
 public:
  Option* pOption;
  double rho;
  double kappa;
  double theta;
  double xi;

  HestonEuler(Option* _pOption,
              double _kappa, double _theta,
              double _xi, double _rho);

  virtual ~HestonEuler();

  // Calculate the volatility path
  void calc_vol_path(const std::vector<double>& vol_draws,
                     std::vector<double>& vol_path);

  // Calculate the asset price path
  void calc_spot_path(const std::vector<double>& spot_draws,
                      const std::vector<double>& vol_path,
                      std::vector<double>& spot_path);
};

#endif
