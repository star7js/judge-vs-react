// raycaster/src/agent.h
#ifndef AGENT_H
#define AGENT_H

#include <stdbool.h>

// Unix-socket bridge for an external agent to drive the jump. The player
// auto-walks forward; the agent only decides *when* to jump.
//   blocking mode  — game waits for each decision (latency = slow-mo, never fails)
//   realtime mode  — game runs at speed; late decisions can miss the jump
bool ray_agent_init(bool realtime);
void ray_agent_shutdown(void);

// Blocking: send state, wait for one action. Returns 1=jump, 0=no, -1=disconnected.
int ray_agent_decide(int tick, float px, float py, int on_ground,
                     float pit_dist, int won, int dead);

// Realtime: fire state (non-blocking), poll for the latest action separately.
void ray_agent_send(int tick, float px, float py, int on_ground,
                    float pit_dist, int won, int dead);
int ray_agent_poll(void);   // 1=jump, 0=none, -1=nothing new / disconnected

#endif
