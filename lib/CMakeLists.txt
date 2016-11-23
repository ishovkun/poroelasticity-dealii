INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(lib
  source/right_hand_side.cc
  )

DEAL_II_SETUP_TARGET(lib)

