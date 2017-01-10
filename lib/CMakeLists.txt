# lib/CMakeLists.txt

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(lib
  source/right_hand_side.cc
  include/TensorIndexer.h
  )

DEAL_II_SETUP_TARGET(lib)

