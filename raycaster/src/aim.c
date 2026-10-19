// raycaster/src/aim.c
#include "aim.h"
#include <math.h>
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
static int aim_oracle(void *ctx, int tick, float aim, Window w)
{
    (void)ctx; (void)tick;
    return aim >= w.w_lo;
}

EpisodeResult aim_run_oracle(AimConfig cfg, int delay, int repeat)
{
    return aim_run(cfg, delay, repeat, aim_oracle, NULL);
}

AimConfig aim_scene_config(AimConfig base, int target_x, float run_up)
{
    // Player at (target_x + 0.5, AIM_PLAYER_Y); the tile's face is the
    // segment y = 1, x in [target_x, target_x + 1]. Angles via atan2 so the
    // window is exactly what the center ray sees.
    float cx = (float)target_x + 0.5f, cy = AIM_PLAYER_Y;
    float lo = atan2f(1.0f - cy, (float)target_x - cx);
    float hi = atan2f(1.0f - cy, (float)target_x + 1.0f - cx);
    AimConfig c = base;
    c.theta_t = 0.5f * (lo + hi);
    c.half_width = 0.5f * (hi - lo);
    c.a0 = c.theta_t - run_up;
    return c;
}

EpisodeResult aim_run(AimConfig cfg, int delay, int repeat, aim_decide_fn decide, void *ctx)
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

        if (!decided && (t % repeat == 0) && decide(ctx, t, aim, win)) {
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

        // No decision and the aim has swept a full window past the far edge:
        // the shot can no longer land, so end the episode as a miss.
        if (!decided && aim > win.w_hi + (win.w_hi - win.w_lo)) res.fell = 1;

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
