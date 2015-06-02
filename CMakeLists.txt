cmake_minimum_required (VERSION 2.8)


include(build/external/LookUp-GreatCMakeCookOff.cmake)
subdirs(gp_emulator/gpu)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/lib/gp_emulator)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests)
message("export PYTHONPATH=${LIBRARY_OUTPUT_PATH}")

