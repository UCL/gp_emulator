cmake_minimum_required (VERSION 2.8)


include(cmake/LookUp-GreatCMakeCookOff.cmake)
subdirs(gp_emulator/gpu)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests)
string(ASCII 27 Esc)

set(ColourReset "${Esc}[m")
set(Red  "${Esc}[1;31m")
message("${Red}Type this line or put it in ~/.bashrc or ~/.bash_profile:\nexport PYTHONPATH=$PYTHONPATH:${LIBRARY_OUTPUT_PATH} ${ColourReset}")
