// raycaster/src/ray.c
#include "ray.h"
#include <math.h>

RayHit ray_cast(const Map *map, float px, float py,
                float ray_dir_x, float ray_dir_y)
{
    RayHit hit = {0};

    int map_x = (int)px;
    int map_y = (int)py;

    float delta_dist_x = (ray_dir_x == 0.0f) ? 1e30f : fabsf(1.0f / ray_dir_x);
    float delta_dist_y = (ray_dir_y == 0.0f) ? 1e30f : fabsf(1.0f / ray_dir_y);

    float side_dist_x, side_dist_y;
    int step_x, step_y;

    if (ray_dir_x < 0) {
        step_x = -1;
        side_dist_x = (px - map_x) * delta_dist_x;
    } else {
        step_x = 1;
        side_dist_x = (map_x + 1.0f - px) * delta_dist_x;
    }
    if (ray_dir_y < 0) {
        step_y = -1;
        side_dist_y = (py - map_y) * delta_dist_y;
    } else {
        step_y = 1;
        side_dist_y = (map_y + 1.0f - py) * delta_dist_y;
    }

    int side = 0;
    for (int i = 0; i < 256; i++) {
        if (side_dist_x < side_dist_y) {
            side_dist_x += delta_dist_x;
            map_x += step_x;
            side = 0;
        } else {
            side_dist_y += delta_dist_y;
            map_y += step_y;
            side = 1;
        }

        if (map_is_solid(map, map_x, map_y)) {
            hit.hit = 1;
            hit.map_x = map_x;
            hit.map_y = map_y;
            hit.side = side;
            hit.wall_type = map->walls[map_y][map_x];

            if (side == 0)
                hit.distance = side_dist_x - delta_dist_x;
            else
                hit.distance = side_dist_y - delta_dist_y;

            if (hit.distance < 0.0001f) hit.distance = 0.0001f;
            break;
        }
    }

    if (!hit.hit) hit.distance = 1e30f;
    return hit;
}
