#ifndef __OPTION_CPP
#define __OPTION_CPP

#include "option.h"

Option::Option(double _K, double _r,
               double _T, PayOff* _pay_off) :
  pay_off(_pay_off), K(_K),  r(_r), T(_T) {}

Option::~Option() {}

#endif