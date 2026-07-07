// raycaster/src/aim.c
#include "aim.h"
#include <stdio.h>

// Upper bound on ticks so a misconfigured sweep can't hang a batch run.
#define AIM_MAX_TICKS 6000

Window aim_window(AimConfig cfg)
{
    Window w = { cfg.theta_t - cfg.half_width, cfg.theta_t + cfg.half_width };
    return w;
}

// The oracle: fire the first consulted tick the aim is at/past the window's
// near edge (the angular analog of the pit oracle's `px >= w_lo`). Stateless.
static int aim_oracle(float aim, Window w)
{
    return aim >= w.w_lo;
}

EpisodeResult aim_run_oracle(AimConfig cfg, int delay, int repeat)
{
    if (repeat < 1) repeat = 1;

    Window win = aim_window(cfg);

    EpisodeResult res;
    res.cleared = 0;
    res.fell = 0;
    res.decision_tick = -1;
    res.decision_x = 0.0f;
    res.in_window = 0;
    res.window = win;
    res.ticks = 0;

    int decided = 0;
    int fire_exec_tick = -1;
    float aim = cfg.a0;

    for (int t = 0; t < AIM_MAX_TICKS && !res.cleared && !res.fell; t++) {
        // Aim advances one omega step per tick. Ordering mirrors the pit's
        // episode_run (advance the world, then consult the policy): t=0 uses
        // a0 as-is, every later tick has already rotated by omega.
        if (t > 0) aim += cfg.omega;

        if (!decided && (t % repeat == 0) && aim_oracle(aim, win)) {
            decided = 1;
            res.decision_tick = t;      // when the decision was MADE
            res.decision_x = aim;       // aim angle at the decision
            // Ground truth: a delay-0 shot right now hits iff the aim is
            // inside the window — the same window "cleared" is judged against.
            res.in_window = (aim >= win.w_lo && aim <= win.w_hi) ? 1 : 0;
            fire_exec_tick = t + delay; // enqueue: apply `delay` ticks later
        }

        // Apply the buffered shot on its apply-tick. The aim at that moment
        // (= decision aim + omega*delay) decides hit vs. overshoot.
        if (decided && t == fire_exec_tick) {
            if (aim >= win.w_lo && aim <= win.w_hi) res.cleared = 1;
            else res.fell = 1;
        }

        res.ticks = t + 1;
    }

    // A decided-but-unresolved shot (apply tick past the plateau) counts as a
    // miss — shouldn't happen for reachable configs, but keep it definite.
    if (decided && !res.cleared && !res.fell) res.fell = 1;
    return res;
}

void aim_result_json(const EpisodeResult *res, const AimConfig *cfg,
                     const char *policy, int delay, int repeat, char *buf, int n)
{
    // Same key set/order as episode_result_json so the Python analysis is
    // schema-identical: speed<-omega, w_lo/w_hi<-angular window, pit_width<-
    // width_id, decision_x<-aim angle, pit_near<-theta_t.
    snprintf(buf, (size_t)n,
        "{\"policy\":\"%s\",\"seed\":%u,\"pit_near\":%.4f,\"pit_width\":%d,"
        "\"speed\":%.4f,\"fog\":%.3f,\"delay\":%d,\"repeat\":%d,"
        "\"cleared\":%d,\"fell\":%d,\"decision_tick\":%d,\"decision_x\":%.4f,"
        "\"in_window\":%d,\"w_lo\":%.4f,\"w_hi\":%.4f,\"ticks\":%d}",
        policy, cfg->seed, cfg->theta_t, cfg->width_id,
        cfg->omega, 0.0f, delay, repeat,
        res->cleared, res->fell, res->decision_tick, res->decision_x,
        res->in_window, res->window.w_lo, res->window.w_hi, res->ticks);
}
