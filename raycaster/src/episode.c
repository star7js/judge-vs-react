// raycaster/src/episode.c
// Shared physics (try_move/jump_step) plus the deterministic headless
// episode core: analytic pit window, oracle policy, and JSON result output.
#include "episode.h"
#include <math.h>
#include <stdio.h>

// Upper bound on ticks per episode so a misconfigured pit (e.g. unreachable
// exit) can't hang a batch run forever. Well above any reachable episode's
// natural length.
#define MAX_TICKS 6000

// Per-axis sliding collision, checking both leading corners of the player box.
void try_move(const Map *map, float *px, float *py, float dx, float dy)
{
    float nx = *px + dx;
    float ex = nx + (dx > 0 ? PLAYER_R : -PLAYER_R);
    if (!map_is_solid(map, (int)ex, (int)(*py - PLAYER_R)) &&
        !map_is_solid(map, (int)ex, (int)(*py + PLAYER_R)))
        *px = nx;

    float ny = *py + dy;
    float ey = ny + (dy > 0 ? PLAYER_R : -PLAYER_R);
    if (!map_is_solid(map, (int)(*px - PLAYER_R), (int)ey) &&
        !map_is_solid(map, (int)(*px + PLAYER_R), (int)ey))
        *py = ny;
}

// One tick of jump physics. Returns the unscaled jump height jz.
float jump_step(float *jz, float *jvz, float gravity)
{
    *jz += *jvz;
    *jvz -= gravity;
    if (*jz <= 0.0f) { *jz = 0.0f; *jvz = 0.0f; }
    return *jz;
}

// Does a grounded takeoff at x clear the pit? Simulates the *exact* same
// tick sequence episode_run() uses for a delay-0 jump commit (jump applied
// and integrated on the takeoff tick itself, then try_move+jump_step per
// tick thereafter), and uses the same center-point/int-truncated fall check.
// This makes the takeoff window and `in_window` derive from the sim's own
// dynamics instead of a separate analytic formula that can drift from it
// (e.g. by wrongly insetting for PLAYER_R, which the fall check never uses
// since pit floor is non-solid and never gates on the player's radius).
static int takeoff_clears(PitConfig cfg, JumpPhysics phys, float x)
{
    Map map;
    map_load_pit(&map, cfg);
    int py = (int)map.spawn_y; // py never changes: spawn_angle 0 => dy == 0
    float near_tile = (float)(int)cfg.pit_near;
    float far_tile = near_tile + (float)cfg.pit_width;

    // Already standing over the pit (center-based, matching the sim's fall
    // check) — too late to take off, this is a fall in progress.
    if (map_floor(&map, (int)x, py) == FLOOR_PIT) return 0;

    float px = x;
    float jz = 0.0f, jvz = phys.jump_v0;

    for (int t = 0; t < MAX_TICKS; t++) {
        if (t > 0) px += phys.move_speed; // try_move, one tick after takeoff
        jump_step(&jz, &jvz, phys.gravity);
        if (jz <= 0.0f) {
            // Landed this tick, exactly like episode_run's fall check. A
            // jump is a one-shot decision (never reconsidered), so landing
            // short of the pit's near edge is NOT a clear: grounded with no
            // jump left, the walk will inevitably carry it into the pit on
            // a later tick. Only landing at/past the far edge — clearing
            // the whole gap in this one jump — counts.
            if (px < near_tile) return 0;
            return px >= far_tile;
        }
    }
    return 0; // never landed (shouldn't happen for reachable configs)
}

Window pit_window(PitConfig cfg, JumpPhysics phys)
{
    Map map;
    map_load_pit(&map, cfg);
    // Align to the same integer tile boundary map_load_pit uses for the
    // pit's near edge, so the scan's upper bound matches the actual hazard
    // tiles even when cfg.pit_near is fractional.
    float near_tile = (float)(int)cfg.pit_near;

    // Scan the contiguous clear region directly from the sim's own dynamics
    // (via takeoff_clears) rather than an analytic formula that can drift
    // from the discrete integration.
    Window w = { near_tile, -1.0f };
    int found = 0;
    const float step = 0.001f;
    for (float x = map.spawn_x; x < near_tile; x += step) {
        if (takeoff_clears(cfg, phys, x)) {
            if (!found) { w.w_lo = x; found = 1; }
            w.w_hi = x;
        }
    }
    return w;
}

// The oracle policy: jump the first grounded tick the player is at/past the
// analytic takeoff window's low edge. `ctx` unused (oracle is stateless).
static int oracle_decide(void *ctx, int tick, float px, float py,
                          int on_ground, Window w)
{
    (void)ctx; (void)tick; (void)py;
    return (on_ground && px >= w.w_lo) ? 1 : 0;
}

EpisodeResult episode_run(PitConfig cfg, JumpPhysics phys, int delay, int repeat,
                          decide_fn decide, void *ctx)
{
    if (repeat < 1) repeat = 1;

    Map map;
    map_load_pit(&map, cfg);

    Window win = pit_window(cfg, phys);

    float px = map.spawn_x, py = map.spawn_y;
    float dx = cosf(map.spawn_angle), dy = sinf(map.spawn_angle);
    float jz = 0.0f, jvz = 0.0f;
    float ex = (float)map.exit_x + 0.5f, ey = (float)map.exit_y + 0.5f;

    EpisodeResult res;
    res.cleared = 0;
    res.fell = 0;
    res.decision_tick = -1;
    res.decision_x = 0.0f;
    res.in_window = 0;
    res.window = win;
    res.ticks = 0;

    // Decision buffer: a jump is a one-shot event (once decided, never
    // reconsidered), so a single pending slot (jump_exec_tick) is all the
    // "ring buffer keyed by apply-tick" needs to hold. `repeat` gates *when
    // the policy is consulted* (only on ticks where tick % repeat == 0 —
    // between consultations the prior "no jump" decision is implicitly
    // held, since `decided` just stays 0); `delay` gates *when a decision
    // that was made gets applied* to the physics.
    int decided = 0;
    int jump_exec_tick = -1;

    for (int t = 0; t < MAX_TICKS && !res.cleared && !res.fell; t++) {
        try_move(&map, &px, &py, dx * cfg.speed, dy * cfg.speed);

        int on_ground = (jz <= 0.0f);
        if (!decided && (t % repeat == 0) && decide(ctx, t, px, py, on_ground, win)) {
            decided = 1;
            res.decision_tick = t;      // when the decision was MADE
            res.decision_x = px;
            // Ground truth: does a delay-0 takeoff right now actually clear
            // the pit? Same source of truth pit_window() scans with, so
            // in_window can never drift from what "clears" actually means.
            res.in_window = takeoff_clears(cfg, phys, px);
            jump_exec_tick = t + delay; // enqueue: apply `delay` ticks later
        }

        // Apply the buffered decision on its apply-tick.
        if (decided && t == jump_exec_tick && jz <= 0.0f) {
            jvz = phys.jump_v0;
        }

        jump_step(&jz, &jvz, phys.gravity);

        if (jz <= 0.0f && map_floor(&map, (int)px, (int)py) == FLOOR_PIT) res.fell = 1;
        if (hypotf(ex - px, ey - py) < WIN_DIST) res.cleared = 1;

        res.ticks = t + 1;
    }

    return res;
}

EpisodeResult episode_run_oracle(PitConfig cfg, JumpPhysics phys, int delay, int repeat)
{
    return episode_run(cfg, phys, delay, repeat, oracle_decide, NULL);
}

void episode_result_json(const EpisodeResult *res, const PitConfig *cfg,
                          const char *policy, int delay, int repeat,
                          char *buf, int n)
{
    snprintf(buf, (size_t)n,
        "{\"policy\":\"%s\",\"seed\":%u,\"pit_near\":%.3f,\"pit_width\":%d,"
        "\"speed\":%.3f,\"fog\":%.3f,\"delay\":%d,\"repeat\":%d,"
        "\"cleared\":%d,\"fell\":%d,\"decision_tick\":%d,\"decision_x\":%.3f,"
        "\"in_window\":%d,\"w_lo\":%.3f,\"w_hi\":%.3f,\"ticks\":%d}",
        policy, cfg->seed, cfg->pit_near, cfg->pit_width,
        cfg->speed, cfg->fog, delay, repeat,
        res->cleared, res->fell, res->decision_tick, res->decision_x,
        res->in_window, res->window.w_lo, res->window.w_hi, res->ticks);
}
