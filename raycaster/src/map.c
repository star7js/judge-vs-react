// raycaster/src/map.c
#include "map.h"
#include <string.h>

void map_load_test(Map *map)
{
    memset(map, 0, sizeof(*map));
    map->width = 24;
    map->height = 11;

    // Border walls
    for (int x = 0; x < map->width; x++) {
        map->walls[0][x] = 1;
        map->walls[map->height - 1][x] = 1;
    }
    for (int y = 0; y < map->height; y++) {
        map->walls[y][0] = 1;
        map->walls[y][map->width - 1] = 1;
    }

    // A pit band across the interior: 2 tiles wide (x = 10, 11). You must jump
    // it. Judging *when* to jump — from first person, where the far edge is hard
    // to place — is the whole point.
    for (int y = 1; y < map->height - 1; y++) {
        map->floor_type[y][10] = FLOOR_PIT;
        map->floor_type[y][11] = FLOOR_PIT;
    }

    // Spawn left of the pit, facing +x toward it and the beacon beyond.
    map->spawn_x = 2.5f;
    map->spawn_y = 5.5f;
    map->spawn_angle = 0.0f;

    // Exit beacon on the far side of the pit.
    map->exit_x = 21;
    map->exit_y = 5;
    map->walls[map->exit_y][map->exit_x] = TILE_EXIT;
}

// Gap (tiles) from the pit's far edge to the exit beacon — matches the
// hand-authored layout in map_load_test (pit far edge 12, exit 21).
#define PIT_EXIT_GAP 9

void map_load_pit(Map *map, PitConfig cfg)
{
    memset(map, 0, sizeof(*map));
    map->width = 24;
    map->height = 11;

    // Border walls
    for (int x = 0; x < map->width; x++) {
        map->walls[0][x] = 1;
        map->walls[map->height - 1][x] = 1;
    }
    for (int y = 0; y < map->height; y++) {
        map->walls[y][0] = 1;
        map->walls[y][map->width - 1] = 1;
    }

    // Parametric pit band: [pit_near, pit_near+pit_width) tiles.
    int near = (int)cfg.pit_near;
    int far = near + cfg.pit_width;
    for (int y = 1; y < map->height - 1; y++) {
        for (int x = near; x < far; x++) {
            if (x > 0 && x < map->width - 1)
                map->floor_type[y][x] = FLOOR_PIT;
        }
    }

    // Spawn left of the pit, facing +x toward it and the beacon beyond.
    map->spawn_x = 2.5f;
    map->spawn_y = 5.5f;
    map->spawn_angle = 0.0f;

    // Exit beacon a fixed gap past the pit's far edge, clamped inside the
    // room's border walls.
    map->exit_x = far + PIT_EXIT_GAP;
    if (map->exit_x > map->width - 2) map->exit_x = map->width - 2;
    map->exit_y = 5;
    map->walls[map->exit_y][map->exit_x] = TILE_EXIT;
}
