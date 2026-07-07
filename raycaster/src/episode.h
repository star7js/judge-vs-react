// raycaster/src/episode.h
// Shared jump/collision physics and the deterministic headless episode core.
//
// try_move()/jump_step() live here (not in main.c) so the windowed game loop
// and the headless --episode runner share one copy of the physics — no drift
// between "what the player feels" and "what the sim measures."
#ifndef EPISODE_H
#define EPISODE_H

#include "map.h"

// Physics constants shared by the live game loop (main.c) and the headless
// episode sim (episode.c). Single source of truth.
#define MOVE_SPEED    0.07f    // default per-tick move speed (units/tick)
#define JUMP_V0       0.11f    // jump impulse (jump-height units / tick)
#define GRAVITY       0.005f   // per tick
#define PLAYER_R      0.20f    // player collision box half-width
#define WIN_DIST      1.5f     // proximity (tiles) to the exit beacon that counts as "reached"

// Physics constants used to derive the analytic takeoff window. Defaults
// mirror MOVE_SPEED/JUMP_V0/GRAVITY above.
typedef struct {
    float jump_v0;
    float gravity;
    float move_speed;
} JumpPhysics;

// The analytic takeoff window: [w_lo, w_hi] is the range of grounded x
// positions from which a jump clears the pit. Depends only on the pit
// config + jump physics — never on which policy is under test.
typedef struct {
    float w_lo, w_hi;
} Window;

typedef struct {
    int cleared;          // reached the exit beacon without falling
    int fell;              // fell into the pit
    int decision_tick;     // tick the policy committed to jumping (-1 if never)
    float decision_x;      // player x at the decision tick
    int in_window;         // 1 if decision_x was inside [w_lo, w_hi]
    Window window;         // the oracle-derived window for this config
    int ticks;              // total ticks simulated
} EpisodeResult;

// Per-axis sliding collision, checking both leading corners of the player box.
void try_move(const Map *map, float *px, float *py, float dx, float dy);

// One tick of jump physics: integrates jz/jvz under gravity, clamps to the
// ground. Returns the (unscaled) jump height jz; callers that render apply
// their own pixel scale factor.
float jump_step(float *jz, float *jvz, float gravity);

// Analytic takeoff window for a pit config under given jump physics.
Window pit_window(PitConfig cfg, JumpPhysics phys);

// Policy decision callback: called once per *consulted* tick (i.e. only on
// ticks where tick % repeat == 0 and no decision has been made yet). `ctx` is
// policy-private state (NULL for the oracle). `w` is the policy-independent
// analytic window — the oracle uses it directly; other policies may ignore
// it entirely (e.g. an external agent judging from a rendered frame).
// Returns 1 to jump (commit the decision on this tick), 0 to keep walking.
typedef int (*decide_fn)(void *ctx, int tick, float px, float py,
                         int on_ground, Window w);

// Run one deterministic episode with a caller-supplied decision policy.
// Owns the physics loop, the delay/repeat decision buffer, pit_window
// labeling, and fall/clear detection — shared by every policy so they can't
// drift from one another. `repeat` gates when `decide` is consulted (only
// ticks where tick % repeat == 0; between consultations the prior decision
// is held); `delay` defers applying a made decision by `delay` ticks.
// decision_tick/decision_x always record when the decision was originally
// made, not when it was applied.
EpisodeResult episode_run(PitConfig cfg, JumpPhysics phys, int delay, int repeat,
                          decide_fn decide, void *ctx);

// Run one deterministic episode with the oracle policy (jumps at the first
// grounded tick where x >= w_lo). Thin wrapper over episode_run with the
// built-in oracle_decide callback; kept for API stability.
EpisodeResult episode_run_oracle(PitConfig cfg, JumpPhysics phys, int delay, int repeat);

// Serialize one episode result + its config to a single JSON line in buf
// (truncated to n bytes). Deterministic: no wall-clock, no pointers/addresses.
void episode_result_json(const EpisodeResult *res, const PitConfig *cfg,
                          const char *policy, int delay, int repeat,
                          char *buf, int n);

#endif
