# lib/CMakeLists.txt

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(lib
  source/right_hand_side.cc
  include/TensorIndexer.h
  include/BoundaryConditions.h
  include/PoroElasticPressureSolver.h
  include/PoroElasticDisplacementSolver.h
  include/ConstitutiveModel.h
  include/StrainProjector.h
  include/InputDataPoroel.h
  )

DEAL_II_SETUP_TARGET(lib)

