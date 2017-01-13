 # code/CMakeLists.txt

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/lib/include
  )

ADD_EXECUTABLE(code source/Runner.cpp)
DEAL_II_SETUP_TARGET(code)

TARGET_LINK_LIBRARIES(code lib)
