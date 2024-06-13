// Author: chenxi-wang

#pragma once
#include <torch/extension.h>

at::Tensor dynamic_cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, at::Tensor radius, const float hmin, const float hmax,
                      const int nsample);