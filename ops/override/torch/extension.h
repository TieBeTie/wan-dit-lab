// Stands first on the include path in place of torch's own extension.h: the same C++ API without pybind11 and
// Python.h, so that the kernels of Nunchaku and Sage build into a library that loads without Python.
#pragma once
#include <torch/all.h>
