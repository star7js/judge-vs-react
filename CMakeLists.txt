cmake_minimum_required(VERSION 3.10)
project(judge-vs-react C)

set(CMAKE_C_STANDARD 99)

# The deterministic raycaster sim — the only C component. Self-contained
# (SDL2 + libm); also builds standalone from raycaster/.
add_subdirectory(raycaster)
