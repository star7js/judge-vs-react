// raycaster/src/render.c
#include "render.h"
#include "ray.h"

static uint32_t wall_colors[] = {
    0xFF404040,  // 0 (unused)
    0xFFCC4444,  // 1 red brick
    0xFF44CC44,  // 2 green
    0xFF4488CC,  // 3 blue
    0xFFCCCC44,  // 4 yellow
    0xFF884488,  // 5
    0xFF44CCCC,  // 6
    0xFFCC8844,  // 7
    0xFF8888CC,  // 8
    0xFF00FF66,  // 9 EXIT — loud green beacon
};
#define NUM_WALL_COLORS ((int)(sizeof(wall_colors) / sizeof(wall_colors[0])))

static inline uint32_t shade(uint32_t color, float factor)
{
    if (factor < 0.0f) factor = 0.0f;
    if (factor > 1.0f) factor = 1.0f;
    uint32_t r = (uint32_t)(((color >> 16) & 0xFF) * factor);
    uint32_t g = (uint32_t)(((color >> 8) & 0xFF) * factor);
    uint32_t b = (uint32_t)((color & 0xFF) * factor);
    return 0xFF000000u | (r << 16) | (g << 8) | b;
}

// Linear blend a->b by t in [0,1] (per channel).
static inline uint32_t blend(uint32_t a, uint32_t b, float t)
{
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float u = 1.0f - t;
    uint32_t r = (uint32_t)(((a >> 16) & 0xFF) * u + ((b >> 16) & 0xFF) * t);
    uint32_t g = (uint32_t)(((a >> 8) & 0xFF) * u + ((b >> 8) & 0xFF) * t);
    uint32_t bl = (uint32_t)((a & 0xFF) * u + (b & 0xFF) * t);
    return 0xFF000000u | (r << 16) | (g << 8) | bl;
}

#define ATMOSPHERE 0xFF20243A   // the ceiling/fog color the distance fades toward

void render_frame(uint32_t *pixels, const Map *map,
                  float px, float py, float dir_x, float dir_y,
                  float plane_x, float plane_y, float cam_off, float fog)
{
    const int w = SCREEN_W, h = SCREEN_H;
    const int horizon = h / 2;
    // fog=0 reproduces the baseline coefficients exactly; higher fog multiplies
    // the distance falloff so the scene (and the pit's distance) fades sooner.
    const float fog_mul = 1.0f + 3.0f * (fog < 0.0f ? 0.0f : fog);

    // Ray directions at the left and right screen edges (for floor casting).
    float rdx0 = dir_x - plane_x, rdy0 = dir_y - plane_y;
    float rdx1 = dir_x + plane_x, rdy1 = dir_y + plane_y;

    // Ceiling: flat dark blue above the horizon.
    for (int y = 0; y < horizon; y++)
        for (int x = 0; x < w; x++)
            pixels[y * w + x] = 0xFF20243A;

    // Floor casting below the horizon — this is what makes the pit visible as a
    // real gap in the ground at the correct perspective distance.
    for (int y = horizon; y < h; y++) {
        int p = y - horizon;
        if (p == 0) p = 1;
        float row_dist = (0.5f * h + cam_off) / p;
        float step_x = row_dist * (rdx1 - rdx0) / w;
        float step_y = row_dist * (rdy1 - rdy0) / w;
        float fx = px + row_dist * rdx0;
        float fy = py + row_dist * rdy0;
        float vis = 1.0f / (1.0f + row_dist * 0.10f * fog_mul);

        for (int x = 0; x < w; x++) {
            int cx = (int)fx, cy = (int)fy;
            uint32_t out;
            if (cx < 0 || cy < 0 || cx >= map->width || cy >= map->height) {
                out = shade(0xFF282828, vis);
            } else if (map->floor_type[cy][cx] == FLOOR_PIT) {
                // The pit is black up close; under fog it fades toward the
                // atmosphere with distance, hiding *how far* the pit is. At
                // fog=0 it stays pure black (baseline).
                out = blend(0xFF050505, ATMOSPHERE, fog * (1.0f - vis));
            } else {
                uint32_t base = ((cx + cy) & 1) ? 0xFF3C3C3C : 0xFF2E2E2E;
                out = shade(base, vis);
            }
            pixels[y * w + x] = out;
            fx += step_x;
            fy += step_y;
        }
    }

    // Walls, drawn on top. cam_off shifts the vertical center (perspective
    // parallax: nearer walls shift more), which reads as rising during a jump.
    for (int x = 0; x < w; x++) {
        float camera_x = 2.0f * x / (float)w - 1.0f;
        float ray_dx = dir_x + plane_x * camera_x;
        float ray_dy = dir_y + plane_y * camera_x;

        RayHit hit = ray_cast(map, px, py, ray_dx, ray_dy);
        if (!hit.hit) continue;

        int line_height = (int)(h / hit.distance);
        int center = horizon + (int)(cam_off / hit.distance);
        int draw_start = center - line_height / 2;
        int draw_end = center + line_height / 2;
        if (draw_start < 0) draw_start = 0;
        if (draw_end >= h) draw_end = h - 1;

        int ci = hit.wall_type;
        if (ci < 0 || ci >= NUM_WALL_COLORS) ci = 1;
        uint32_t base = wall_colors[ci];

        int is_exit = (hit.wall_type == TILE_EXIT);
        float fog_k = (is_exit ? 0.06f : 0.15f) * fog_mul;
        float factor = 1.0f / (1.0f + hit.distance * fog_k);
        if (hit.side == 1) factor *= 0.65f;

        uint32_t color = shade(base, factor);
        for (int y = draw_start; y <= draw_end; y++)
            pixels[y * w + x] = color;
    }
}
