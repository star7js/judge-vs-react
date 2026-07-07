// raycaster/src/ray.h
#ifndef RAY_H
#define RAY_H

#include "map.h"

typedef struct {
    float distance;    // perpendicular distance to wall (fisheye-corrected)
    int wall_type;     // which wall type was hit
    int side;          // 0 = vertical wall face (N/S), 1 = horizontal (E/W)
    int map_x, map_y;  // grid cell hit
    int hit;           // 1 if a wall was hit within range, else 0
} RayHit;

// Cast a single ray from (px,py) along (dir_x,dir_y).
RayHit ray_cast(const Map *map, float px, float py,
                float ray_dir_x, float ray_dir_y);

#endif
