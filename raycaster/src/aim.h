// raycaster/src/aim.h
// A second reflex task — the *sweeping-aim* task — used to test whether the
// pit-jump timing law (r = travel/window collapse near r~1) is specific to
// translational jump-timing or a general property of action-timing-in-a-window.
//
// The player rotates in place at constant angular speed omega (rad/tick). A
// target is centered at theta_t with half-width h, so the angular hit window
// is [theta_t - h, theta_t + h] (width 2h). The aim starts at a0 < theta_t - h
// and sweeps up through the window. The single decision is "fire"; a delay D
// applies the shot D ticks later, by which point the aim has rotated omega*D
// further and may overshoot. This is the angular analog of the pit jump, with
//     r = omega * D / (2h)      (== speed*delay/W in the pit's terms).
#ifndef AIM_H
#define AIM_H

#include "episode.h"   // Window, EpisodeResult

typedef struct {
    float theta_t;      // target center angle (rad)
    float half_width;   // h: target angular half-width (rad); window width = 2h
    float omega;        // angular sweep speed (rad/tick) — the "speed" analog
    float a0;           // starting aim angle (rad); must be < theta_t - h
    int   width_id;     // integer bucket id for grouping in analysis (pit_width analog)
    unsigned seed;      // carried for schema parity (oracle is deterministic)
} AimConfig;

// Angular hit window [theta_t - h, theta_t + h].
Window aim_window(AimConfig cfg);

// Run one deterministic aim episode with the oracle (fire at the first
// consulted tick the aim is at/past the window's near edge), deferring the
// shot by `delay` ticks and consulting every `repeat` ticks.
EpisodeResult aim_run_oracle(AimConfig cfg, int delay, int repeat);

// Serialize an aim result to the SAME JSON schema as episode_result_json, so
// the Python analysis reuses ratio()/window_width() unchanged:
//   speed <- omega,  w_lo/w_hi <- angular window,  pit_width <- width_id,
//   decision_x <- aim angle,  pit_near <- theta_t (unused downstream).
void aim_result_json(const EpisodeResult *res, const AimConfig *cfg,
                     const char *policy, int delay, int repeat, char *buf, int n);

#endif
