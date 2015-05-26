cmake_minimum_required (VERSION 2.8)


include(build/external/LookUp-GreatCMakeCookOff.cmake)
subdirs(gp_emulator/gpu)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/lib/gp_emulator)

message("export PYTHONPATH=${LIBRARY_OUTPUT_PATH}")

#target_link_libraries(python_predict python)
