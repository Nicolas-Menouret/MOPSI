cmake_minimum_required(VERSION 2.6)
file(TO_CMAKE_PATH "$ENV{IMAGINEPP_ROOT}/CMake" p)
list(APPEND CMAKE_MODULE_PATH "${p}") #For old Imagine++
list(APPEND CMAKE_SYSTEM_FRAMEWORK_PATH /Library/Frameworks) #Mac, why not auto?
find_package(Imagine REQUIRED)

project(MOPSI)
add_executable(MOPSI main.cpp Tools.h Tools.cpp)
ImagineUseModules(MOPSI Graphics)
ImagineUseModules(MOPSI LinAlg)
ImagineUseModules(MOPSI Common)
ImagineUseModules(MOPSI Images)

