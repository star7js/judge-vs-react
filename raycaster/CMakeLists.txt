# raycaster/CMakeLists.txt
# Self-contained: SDL2 + libm only. No cJSON / omnispeak dependency (the agent
# bridge speaks JSON-over-socket by hand). Builds standalone or from the root.
cmake_minimum_required(VERSION 3.10)
project(keen-raycaster C)

set(CMAKE_C_STANDARD 99)

# Resolve SDL2 via sdl2-config (Homebrew ships it; no pkg-config needed).
execute_process(COMMAND sdl2-config --cflags
                OUTPUT_VARIABLE SDL2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND sdl2-config --libs
                OUTPUT_VARIABLE SDL2_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(SDL2_CFLAGS)
separate_arguments(SDL2_LIBS)

add_executable(keen-raycaster
    src/main.c
    src/ray.c
    src/map.c
    src/render.c
    src/agent.c
    src/episode.c
    src/aim.c
)
target_compile_options(keen-raycaster PRIVATE ${SDL2_CFLAGS})
target_link_libraries(keen-raycaster PRIVATE ${SDL2_LIBS} m)
