// raycaster/src/map.h
#ifndef MAP_H
#define MAP_H

#include <stdint.h>

#define MAP_MAX_W 64
#define MAP_MAX_H 64

// Special wall type: the exit beacon. Rendered a loud unique color and used as
// the win target. It's a solid wall you navigate toward (you win by getting
// close), which keeps a pure wall-raycaster's "find the exit" task achievable
// without floor sprites.
#define TILE_EXIT 9

// Floor tile types (separate layer from walls).
#define FLOOR_NORMAL 0
#define FLOOR_PIT    1   // a hole — you fall in if you're on it un-jumped

typedef struct {
    int width;
    int height;
    uint8_t walls[MAP_MAX_H][MAP_MAX_W];       // 0 = empty, 1+ = wall type
    uint8_t floor_type[MAP_MAX_H][MAP_MAX_W];  // FLOOR_* per tile
    float spawn_x, spawn_y;                    // spawn position (tile units)
    float spawn_angle;                         // spawn facing (radians)
    int exit_x, exit_y;                        // exit beacon tile
} Map;

// Parametric pit episode config: where/how wide the pit is, how fast the
// player moves, the episode's RNG seed, and a fog/visibility knob. Defines
// the scenario; jump physics (JumpPhysics, in episode.h) is separate.
typedef struct {
    float pit_near;   // tile x where the pit band starts
    int pit_width;    // pit band width in tiles
    float speed;      // per-tick move speed for this episode
    unsigned seed;    // episode RNG seed (echoed; oracle policy is deterministic)
    float fog;        // visibility/fog knob (echoed; unused by headless physics)
} PitConfig;

// Hand-authored test maze with a marked exit. De-risks first-person navigation
// before wiring real Keen tile maps.
void map_load_test(Map *map);

// Same 24x11 room as map_load_test, but with a parametric pit band
// [pit_near, pit_near+pit_width) and an exit beacon placed a fixed gap past
// the far edge of the pit.
void map_load_pit(Map *map, PitConfig cfg);

static inline int map_is_solid(const Map *map, int x, int y)
{
    if (x < 0 || x >= map->width || y < 0 || y >= map->height)
        return 1;
    return map->walls[y][x] > 0;
}

static inline int map_floor(const Map *map, int x, int y)
{
    if (x < 0 || x >= map->width || y < 0 || y >= map->height)
        return FLOOR_NORMAL;
    return map->floor_type[y][x];
}

#endif
