// raycaster/src/render.h
#ifndef RENDER_H
#define RENDER_H

#include "map.h"
#include <stdint.h>

#define SCREEN_W 640
#define SCREEN_H 480

// Render one frame into an ARGB8888 pixel buffer (SCREEN_W * SCREEN_H).
// cam_off = vertical camera offset in pixels (0 = standing; > 0 = jumping/up).
// fog     = visibility knob in [0,1]. 0 = baseline (clear); higher fogs the
//           scene faster with distance AND fades the distant pit toward the
//           atmosphere, hiding how far the pit is (the depth cue).
void render_frame(uint32_t *pixels, const Map *map,
                  float px, float py, float dir_x, float dir_y,
                  float plane_x, float plane_y, float cam_off, float fog);

#endif
