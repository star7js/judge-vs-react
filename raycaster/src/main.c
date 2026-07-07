// raycaster/src/main.c
// Prototype: first-person + jump + a pit you must judge and clear.
// The hard part — by design — is judging from first person *when* to jump so you
// clear the gap. No AI yet; this tests whether that challenge is real/fun.
#include <SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "map.h"
#include "render.h"
#include "agent.h"
#include "episode.h"
#include "aim.h"

#define DECISION_INTERVAL 6   // frames between agent decisions (~0.1s at 60fps)

#define ROT_SPEED     0.045f
#define CAM_OFF_SCALE 120.0f   // pixels of camera rise per unit jump height
// MOVE_SPEED, JUMP_V0, GRAVITY, PLAYER_R, WIN_DIST are defined in episode.h
// (shared with the headless episode sim); try_move()/jump_step() live there too.

// Distance (tiles) from the player to the near edge of the next pit straight
// ahead (facing +x). -1 if none within range. This is the agent's cue.
static float pit_ahead(const Map *map, float px, float py)
{
    int ty = (int)py;
    for (int i = 0; i < 16; i++) {
        int tx = (int)px + i;
        if (map_floor(map, tx, ty) == FLOOR_PIT)
            return (float)tx - px;
    }
    return -1.0f;
}

// Headless self-check: (1) screenshot the pit from spawn and mid-jump so we can
// see it renders as a judgeable gap; (2) simulate a well-timed jump and confirm
// it clears the pit to the beacon without falling.
static int run_selftest(const char *bmp_path)
{
    SDL_setenv("SDL_VIDEODRIVER", "dummy", 1);
    SDL_Init(SDL_INIT_VIDEO);

    Map map;
    map_load_test(&map);
    static uint32_t pixels[SCREEN_W * SCREEN_H];

    float dx = cosf(map.spawn_angle), dy = sinf(map.spawn_angle);
    float pxv = -dy * 0.66f, pyv = dx * 0.66f;

    // Shot 1: standing at spawn, facing the pit + beacon.
    render_frame(pixels, &map, map.spawn_x, map.spawn_y, dx, dy, pxv, pyv, 0.0f, 0.0f);
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(
        pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
        SDL_PIXELFORMAT_ARGB8888);
    if (s && bmp_path) { SDL_SaveBMP(s, bmp_path); printf("[selftest] wrote %s\n", bmp_path); }
    if (s) SDL_FreeSurface(s);

    // Shot 2: mid-jump near the pit edge (camera raised).
    render_frame(pixels, &map, 8.8f, map.spawn_y, dx, dy, pxv, pyv, 100.0f, 0.0f);
    SDL_Surface *s2 = SDL_CreateRGBSurfaceWithFormatFrom(
        pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
        SDL_PIXELFORMAT_ARGB8888);
    if (s2 && bmp_path) {
        char p2[512];
        SDL_snprintf(p2, sizeof(p2), "%s.jump.bmp", bmp_path);
        SDL_SaveBMP(s2, p2);
        printf("[selftest] wrote %s\n", p2);
    }
    if (s2) SDL_FreeSurface(s2);

    // Sim: walk right, jump when just before the pit, confirm we clear it.
    float px = map.spawn_x, py = map.spawn_y, jz = 0.0f, jvz = 0.0f;
    float ex = map.exit_x + 0.5f, ey = map.exit_y + 0.5f;
    int jumped = 0, fell = 0, reached = 0;

    for (int step = 0; step < 3000 && !reached && !fell; step++) {
        try_move(&map, &px, &py, dx * MOVE_SPEED, dy * MOVE_SPEED);
        if (!jumped && px >= 9.0f && jz <= 0.0f) { jvz = JUMP_V0; jumped = 1; }
        jump_step(&jz, &jvz, GRAVITY);
        if (jz <= 0.0f && map_floor(&map, (int)px, (int)py) == FLOOR_PIT) fell = 1;
        if (hypotf(ex - px, ey - py) < WIN_DIST) reached = 1;
    }

    SDL_Quit();
    if (reached && !fell) {
        printf("[selftest] PASS — jump cleared the pit; reached exit at (%.2f,%.2f)\n", px, py);
        return 0;
    }
    fprintf(stderr, "[selftest] FAIL — fell=%d reached=%d pos=(%.2f,%.2f)\n", fell, reached, px, py);
    return 1;
}

// Dump a sequence of approach frames (standing, facing the pit) at increasing
// x positions, for the vision experiment. Prints a manifest line per frame.
static int run_approach(const char *prefix)
{
    SDL_setenv("SDL_VIDEODRIVER", "dummy", 1);
    SDL_Init(SDL_INIT_VIDEO);
    Map map;
    map_load_test(&map);
    static uint32_t pixels[SCREEN_W * SCREEN_H];

    float dx = cosf(map.spawn_angle), dy = sinf(map.spawn_angle);
    float pxv = -dy * 0.66f, pyv = dx * 0.66f;
    // Pit is x=[10,12). A jump carries ~3 tiles, so the clear window is ~[8.9,10).
    float xs[] = {3.0f, 5.0f, 6.5f, 7.5f, 8.5f, 9.0f, 9.5f, 9.9f};
    int n = (int)(sizeof(xs) / sizeof(xs[0]));

    for (int i = 0; i < n; i++) {
        render_frame(pixels, &map, xs[i], map.spawn_y, dx, dy, pxv, pyv, 0.0f, 0.0f);
        SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(
            pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
            SDL_PIXELFORMAT_ARGB8888);
        char path[512];
        SDL_snprintf(path, sizeof(path), "%s_%02d.bmp", prefix, i);
        if (s) { SDL_SaveBMP(s, path); SDL_FreeSurface(s); }
        printf("%s\tx=%.2f\n", path, xs[i]);
    }
    SDL_Quit();
    return 0;
}

// Static "frame battery" for the perception experiment: render standing frames
// at a spread of approach positions for a given pit width + fog, and print a
// manifest line per frame with the ground-truth in_window label (whether a jump
// from that exact spot clears). A vision judge reads the frames and decides
// JUMP/WAIT; agreement with in_window is the judgment measurement.
static int run_shoot(int argc, char **argv)
{
    const char *prefix = "shot";
    PitConfig cfg = { 10.0f, 2, MOVE_SPEED, 1u, 0.0f };
    for (int i = 1; i < argc; i++) {
        if (SDL_strcmp(argv[i], "--shoot") == 0 && i + 1 < argc) prefix = argv[++i];
        else if (SDL_strcmp(argv[i], "--pit-near") == 0 && i + 1 < argc) cfg.pit_near = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--pit-width") == 0 && i + 1 < argc) cfg.pit_width = atoi(argv[++i]);
        else if (SDL_strcmp(argv[i], "--fog") == 0 && i + 1 < argc) cfg.fog = (float)atof(argv[++i]);
    }

    SDL_setenv("SDL_VIDEODRIVER", "dummy", 1);
    SDL_Init(SDL_INIT_VIDEO);
    Map map;
    map_load_pit(&map, cfg);
    static uint32_t pixels[SCREEN_W * SCREEN_H];

    JumpPhysics phys = { JUMP_V0, GRAVITY, cfg.speed };
    Window win = pit_window(cfg, phys);

    // Fixed spread of standing positions relative to the pit near-edge, from far
    // approach up to just before the pit (never past the near edge = would fall).
    float offs[] = { -4.0f, -3.0f, -2.25f, -1.75f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, -0.1f };
    int n = (int)(sizeof(offs) / sizeof(offs[0]));

    for (int i = 0; i < n; i++) {
        float px = (float)(int)cfg.pit_near + offs[i];
        render_frame(pixels, &map, px, map.spawn_y, 1.0f, 0.0f, 0.0f, 0.66f, 0.0f, cfg.fog);
        SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(
            pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
            SDL_PIXELFORMAT_ARGB8888);
        char path[512];
        SDL_snprintf(path, sizeof path, "%s_%02d.bmp", prefix, i);
        if (s) { SDL_SaveBMP(s, path); SDL_FreeSurface(s); }
        int in_win = (px >= win.w_lo && px <= win.w_hi) ? 1 : 0;
        printf("{\"path\":\"%s\",\"idx\":%d,\"px\":%.3f,\"pit_width\":%d,\"fog\":%.2f,"
               "\"w_lo\":%.3f,\"w_hi\":%.3f,\"in_window\":%d}\n",
               path, i, px, cfg.pit_width, cfg.fog, win.w_lo, win.w_hi, in_win);
    }
    SDL_Quit();
    return 0;
}

// Save a rendered frame to a numbered BMP (for --record; works headless).
static void save_frame(const uint32_t *pixels, const char *prefix, long frame)
{
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(
        (void *)pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
        SDL_PIXELFORMAT_ARGB8888);
    if (!s) return;
    char path[512];
    SDL_snprintf(path, sizeof path, "%s_%05ld.bmp", prefix, frame);
    SDL_SaveBMP(s, path);
    SDL_FreeSurface(s);
}

// Write the current frame to a fixed path for an external vision agent to read.
// Atomic (write to .tmp then rename) so the agent never reads a half-written BMP.
static void save_frame_latest(const uint32_t *pixels, const char *path)
{
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormatFrom(
        (void *)pixels, SCREEN_W, SCREEN_H, 32, SCREEN_W * sizeof(uint32_t),
        SDL_PIXELFORMAT_ARGB8888);
    if (!s) return;
    char tmp[512];
    SDL_snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (SDL_SaveBMP(s, tmp) == 0) rename(tmp, path);
    SDL_FreeSurface(s);
}

// Per-tick state the --policy agent callback needs that episode_run() itself
// doesn't carry: the render-only Map/pixel buffer, the --frame-out path, and
// the last consulted (tick,px,py,on_ground), so run_episode can send one
// final terminal-state notification to the agent after episode_run() returns
// (episode_run has no notion of "episode just ended, tell the policy").
typedef struct {
    const Map *map;
    uint32_t *pixels;
    const char *frame_out;
    int last_tick;
    float last_px, last_py;
    int last_on_ground;
    float fog;   // visibility knob passed through to render_frame
} AgentCtx;

// decide_fn for --policy agent: renders the current frame, exports it (if
// --frame-out is set) atomically via save_frame_latest, then blocks on the
// external agent's jump/no-jump decision over the Unix socket. Reuses
// pit_ahead's cue so the existing rules-based pit_agent.py backend still
// works unmodified; a vision backend ignores px/py/pit_dist and judges from
// the exported frame alone.
//
// Determinism note: the oracle path (episode_run_oracle) is fully
// deterministic — physics + the analytic window only. This path's
// determinism depends entirely on the connected agent: a deterministic
// rules agent reproduces the same result every run; a VLM does not. That's
// expected, not a regression.
//
// cam_off: the episode player auto-walks with a fixed heading and
// episode_run() doesn't expose the current jump height to the callback, so
// v1 exports the grounded view (cam_off=0). The decision-relevant cue — the
// pit band and its distance — is fully visible grounded; see the task
// report for the tradeoff.
static int agent_decide(void *ctx_, int tick, float px, float py,
                        int on_ground, Window w)
{
    (void)w;
    AgentCtx *ctx = (AgentCtx *)ctx_;
    ctx->last_tick = tick;
    ctx->last_px = px;
    ctx->last_py = py;
    ctx->last_on_ground = on_ground;

    // Fixed heading (+x, plane (0,0.66)) — same as the episode's auto-walker.
    render_frame(ctx->pixels, ctx->map, px, py, 1.0f, 0.0f, 0.0f, 0.66f, 0.0f, ctx->fog);
    if (ctx->frame_out) save_frame_latest(ctx->pixels, ctx->frame_out);

    float pit_dist = pit_ahead(ctx->map, px, py);
    int a = ray_agent_decide(tick, px, py, on_ground, pit_dist, 0, 0);
    return (a == 1) ? 1 : 0;
}

// Headless, deterministic parametric episode(s): no SDL window, no wall-clock
// in the sim path. Parses --episode's own flags out of argv and prints one
// JSON result line per episode.
static int run_episode(int argc, char **argv)
{
    PitConfig cfg = { 10.0f, 2, MOVE_SPEED, 1u, 0.0f };
    const char *policy = "oracle";
    const char *task = "pit";
    const char *frame_out = NULL;
    int delay = 0, repeat = 1, episodes = 1;
    // Sweeping-aim task (see aim.h). Defaults chosen so the aim sweeps up
    // through the window from a run-up below it.
    AimConfig aim = { 1.0f, 0.06f, 0.01f, 0.5f, 1, 1u };

    for (int i = 1; i < argc; i++) {
        if (SDL_strcmp(argv[i], "--policy") == 0 && i + 1 < argc) policy = argv[++i];
        else if (SDL_strcmp(argv[i], "--task") == 0 && i + 1 < argc) task = argv[++i];
        else if (SDL_strcmp(argv[i], "--pit-near") == 0 && i + 1 < argc) cfg.pit_near = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--pit-width") == 0 && i + 1 < argc) cfg.pit_width = atoi(argv[++i]);
        else if (SDL_strcmp(argv[i], "--speed") == 0 && i + 1 < argc) cfg.speed = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--fog") == 0 && i + 1 < argc) cfg.fog = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = (unsigned)strtoul(argv[++i], NULL, 10); aim.seed = cfg.seed;
        }
        else if (SDL_strcmp(argv[i], "--delay") == 0 && i + 1 < argc) delay = atoi(argv[++i]);
        else if (SDL_strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (SDL_strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) episodes = atoi(argv[++i]);
        else if (SDL_strcmp(argv[i], "--frame-out") == 0 && i + 1 < argc) frame_out = argv[++i];
        // aim-task flags
        else if (SDL_strcmp(argv[i], "--omega") == 0 && i + 1 < argc) aim.omega = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--half-width") == 0 && i + 1 < argc) aim.half_width = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--theta") == 0 && i + 1 < argc) aim.theta_t = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--a0") == 0 && i + 1 < argc) aim.a0 = (float)atof(argv[++i]);
        else if (SDL_strcmp(argv[i], "--width-id") == 0 && i + 1 < argc) aim.width_id = atoi(argv[++i]);
    }

    // Sweeping-aim task: a self-contained kinematic episode (oracle only, no
    // rendering) emitting the same JSON schema as the pit — the generality
    // test for the r~1 law. Handled entirely here, before the pit path.
    if (SDL_strcmp(task, "aim") == 0) {
        if (SDL_strcmp(policy, "oracle") != 0) {
            fprintf(stderr, "[episode] --task aim supports --policy oracle only\n");
            return 1;
        }
        for (int e = 0; e < episodes; e++) {
            EpisodeResult res = aim_run_oracle(aim, delay, repeat);
            char buf[512];
            aim_result_json(&res, &aim, "aim-oracle", delay, repeat, buf, sizeof buf);
            printf("%s\n", buf);
        }
        return 0;
    }

    JumpPhysics phys = { JUMP_V0, GRAVITY, cfg.speed };

    if (SDL_strcmp(policy, "agent") == 0 && episodes != 1) {
        fprintf(stderr, "[episode] --policy agent supports --episodes 1 only\n");
        return 1;
    }

    for (int e = 0; e < episodes; e++) {
        EpisodeResult res;
        if (SDL_strcmp(policy, "oracle") == 0) {
            res = episode_run_oracle(cfg, phys, delay, repeat);
        } else if (SDL_strcmp(policy, "agent") == 0) {
            if (SDL_Init(SDL_INIT_VIDEO) != 0) {
                fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
                return 1;
            }
            if (!ray_agent_init(false)) {
                fprintf(stderr, "[agent] init failed\n");
                SDL_Quit();
                return 1;
            }

            Map map;
            map_load_pit(&map, cfg);
            static uint32_t pixels[SCREEN_W * SCREEN_H];
            AgentCtx actx = { &map, pixels, frame_out, -1, 0.0f, 0.0f, 0, cfg.fog };

            res = episode_run(cfg, phys, delay, repeat, agent_decide, &actx);

            // Notify the agent of the terminal state (won/dead) so
            // pit_agent.py (and any other driver) sees a final message and
            // exits cleanly instead of blocking on a socket the game is
            // about to tear down.
            ray_agent_decide(actx.last_tick, actx.last_px, actx.last_py,
                             actx.last_on_ground, -1.0f, res.cleared, res.fell);
            ray_agent_shutdown();
            SDL_Quit();
        } else {
            fprintf(stderr, "[episode] unknown policy '%s'\n", policy);
            return 1;
        }
        char buf[512];
        episode_result_json(&res, &cfg, policy, delay, repeat, buf, sizeof buf);
        printf("%s\n", buf);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int agent_mode = 0, realtime = 0;
    const char *record_prefix = NULL;
    const char *frame_out = NULL;   // write latest frame here for a vision agent
    for (int i = 1; i < argc; i++) {
        if (SDL_strcmp(argv[i], "--test") == 0) {
            const char *out = (i + 1 < argc) ? argv[i + 1] : "frame.bmp";
            return run_selftest(out);
        }
        if (SDL_strcmp(argv[i], "--episode") == 0) {
            return run_episode(argc, argv);
        }
        if (SDL_strcmp(argv[i], "--approach") == 0) {
            const char *pfx = (i + 1 < argc) ? argv[i + 1] : "approach";
            return run_approach(pfx);
        }
        if (SDL_strcmp(argv[i], "--shoot") == 0) {
            return run_shoot(argc, argv);
        }
        if (SDL_strcmp(argv[i], "--agent") == 0) agent_mode = 1;
        if (SDL_strcmp(argv[i], "--realtime") == 0) realtime = 1;
        if (SDL_strcmp(argv[i], "--record") == 0 && i + 1 < argc) record_prefix = argv[++i];
        if (SDL_strcmp(argv[i], "--frame-out") == 0 && i + 1 < argc) frame_out = argv[++i];
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "Keen Raycaster — WASD/arrows move+turn, Space JUMP the pit, R reset, Esc quit",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        SCREEN_W, SCREEN_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Texture *tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STREAMING, SCREEN_W, SCREEN_H);
    static uint32_t pixels[SCREEN_W * SCREEN_H];

    Map map;
    map_load_test(&map);

    float pos_x, pos_y, dir_x, dir_y, plane_x, plane_y, jz, jvz;
    #define RESET() do { \
        pos_x = map.spawn_x; pos_y = map.spawn_y; \
        dir_x = cosf(map.spawn_angle); dir_y = sinf(map.spawn_angle); \
        plane_x = -dir_y * 0.66f; plane_y = dir_x * 0.66f; \
        jz = 0.0f; jvz = 0.0f; \
    } while (0)
    RESET();

    if (agent_mode) {
        if (!ray_agent_init(realtime)) {
            fprintf(stderr, "[agent] init failed\n");
            return 1;
        }
        SDL_SetWindowTitle(win, realtime
            ? "AI playing (realtime) — watch it judge the jump  |  Esc quit"
            : "AI playing (blocking) — watch it judge the jump  |  Esc quit");
    }

    long frame = 0;
    int running = 1, won = 0, won_frames = 0;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) running = 0;
                if (e.key.keysym.sym == SDLK_r) { RESET(); won = 0; SDL_SetWindowTitle(win, "Keen Raycaster — Space to JUMP the pit"); }
                if (e.key.keysym.sym == SDLK_SPACE && jz <= 0.0f && !won) jvz = JUMP_V0;
            }
        }

        const Uint8 *k = SDL_GetKeyboardState(NULL);
        if (!won && !agent_mode) {
            if (k[SDL_SCANCODE_W] || k[SDL_SCANCODE_UP])
                try_move(&map, &pos_x, &pos_y, dir_x * MOVE_SPEED, dir_y * MOVE_SPEED);
            if (k[SDL_SCANCODE_S] || k[SDL_SCANCODE_DOWN])
                try_move(&map, &pos_x, &pos_y, -dir_x * MOVE_SPEED, -dir_y * MOVE_SPEED);
            if (k[SDL_SCANCODE_Q])
                try_move(&map, &pos_x, &pos_y, plane_x * MOVE_SPEED, plane_y * MOVE_SPEED);
            if (k[SDL_SCANCODE_E])
                try_move(&map, &pos_x, &pos_y, -plane_x * MOVE_SPEED, -plane_y * MOVE_SPEED);
            if (k[SDL_SCANCODE_A] || k[SDL_SCANCODE_LEFT]) {
                float r = -ROT_SPEED, odx = dir_x, opx = plane_x;
                dir_x = dir_x * cosf(r) - dir_y * sinf(r);
                dir_y = odx * sinf(r) + dir_y * cosf(r);
                plane_x = plane_x * cosf(r) - plane_y * sinf(r);
                plane_y = opx * sinf(r) + plane_y * cosf(r);
            }
            if (k[SDL_SCANCODE_D] || k[SDL_SCANCODE_RIGHT]) {
                float r = ROT_SPEED, odx = dir_x, opx = plane_x;
                dir_x = dir_x * cosf(r) - dir_y * sinf(r);
                dir_y = odx * sinf(r) + dir_y * cosf(r);
                plane_x = plane_x * cosf(r) - plane_y * sinf(r);
                plane_y = opx * sinf(r) + plane_y * cosf(r);
            }
        }

        // Agent control: auto-walk forward; the agent only decides the jump.
        if (agent_mode && !won) {
            try_move(&map, &pos_x, &pos_y, dir_x * MOVE_SPEED, dir_y * MOVE_SPEED);
            int og = (jz <= 0.0f);
            float pit = pit_ahead(&map, pos_x, pos_y);
            if (realtime) {
                if (frame % DECISION_INTERVAL == 0) {
                    if (frame_out) save_frame_latest(pixels, frame_out);
                    ray_agent_send((int)frame, pos_x, pos_y, og, pit, 0, 0);
                }
                int a = ray_agent_poll();
                if (a == 1 && og) jvz = JUMP_V0;
            } else {
                if (frame % DECISION_INTERVAL == 0) {
                    // Export the frame the agent will act on *before* we block on
                    // its reply, so a vision agent reads the matching image.
                    if (frame_out) save_frame_latest(pixels, frame_out);
                    int a = ray_agent_decide((int)frame, pos_x, pos_y, og, pit, 0, 0);
                    if (a == 1 && og) jvz = JUMP_V0;
                    if (a == -1) agent_mode = 0;  // disconnected → hand back to keyboard
                }
            }
        }

        float cam_off = jump_step(&jz, &jvz, GRAVITY) * CAM_OFF_SCALE;

        if (!won) {
            // Fell in the pit? (on the ground over a pit tile)
            if (jz <= 0.0f && map_floor(&map, (int)pos_x, (int)pos_y) == FLOOR_PIT) {
                SDL_SetWindowTitle(win, "You fell in the pit!  —  R to try again");
                printf("[raycaster] fell in pit at (%.2f,%.2f)\n", pos_x, pos_y);
                RESET();
            }
            float ex = map.exit_x + 0.5f, ey = map.exit_y + 0.5f;
            if (hypotf(ex - pos_x, ey - pos_y) < WIN_DIST) {
                won = 1;
                SDL_SetWindowTitle(win, "CLEARED THE PIT — reached the exit!  R to restart, Esc to quit");
                printf("[raycaster] reached exit at (%.2f,%.2f)\n", pos_x, pos_y);
                if (agent_mode) {  // tell the agent it won so it exits cleanly
                    if (realtime) ray_agent_send((int)frame, pos_x, pos_y, 1, -1.0f, 1, 0);
                    else ray_agent_decide((int)frame, pos_x, pos_y, 1, -1.0f, 1, 0);
                }
            }
        }

        render_frame(pixels, &map, pos_x, pos_y, dir_x, dir_y, plane_x, plane_y, cam_off, 0.0f);
        SDL_UpdateTexture(tex, NULL, pixels, SCREEN_W * sizeof(uint32_t));
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
        if (record_prefix && frame < 1500) save_frame(pixels, record_prefix, frame);
        frame++;

        // In agent mode, hold on the win screen briefly then exit — otherwise a
        // headless (uncapped) run would dump win-screen frames without end.
        if (agent_mode && won && ++won_frames > 40) running = 0;
    }

    if (agent_mode) ray_agent_shutdown();
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
