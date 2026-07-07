// raycaster/src/agent.c
#include "agent.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

// Default socket path; override with KEEN_AGENT_SOCK so parallel episodes
// (one game+agent pair each) don't collide on a single socket.
#define SOCK_PATH_DEFAULT "/tmp/keen_ray_agent.sock"
static const char *sock_path(void)
{
    const char *p = getenv("KEEN_AGENT_SOCK");
    return (p && *p) ? p : SOCK_PATH_DEFAULT;
}

static int sfd = -1, cfd = -1;
static char rbuf[8192];
static int rlen = 0;

static void send_state(int tick, float px, float py, int og,
                       float pit, int won, int dead)
{
    if (cfd < 0) return;
    char b[256];
    int n = snprintf(b, sizeof b,
        "{\"tick\":%d,\"px\":%.3f,\"py\":%.3f,\"on_ground\":%d,"
        "\"pit_dist\":%.3f,\"won\":%d,\"dead\":%d}\n",
        tick, px, py, og, pit, won, dead);
    if (send(cfd, b, n, 0) < 0) { close(cfd); cfd = -1; }
}

bool ray_agent_init(bool realtime)
{
    unlink(sock_path());
    sfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd < 0) return false;

    struct sockaddr_un a;
    memset(&a, 0, sizeof a);
    a.sun_family = AF_UNIX;
    strncpy(a.sun_path, sock_path(), sizeof a.sun_path - 1);

    if (bind(sfd, (struct sockaddr *)&a, sizeof a) < 0) return false;
    if (listen(sfd, 1) < 0) return false;
    signal(SIGPIPE, SIG_IGN);

    printf("[agent] waiting for agent on %s (%s) ...\n",
           sock_path(), realtime ? "realtime" : "blocking");
    fflush(stdout);

    cfd = accept(sfd, NULL, NULL);      // block until the agent connects
    if (cfd < 0) return false;
    if (realtime) fcntl(cfd, F_SETFL, O_NONBLOCK);

    printf("[agent] connected\n");
    fflush(stdout);
    return true;
}

void ray_agent_shutdown(void)
{
    if (cfd >= 0) close(cfd);
    if (sfd >= 0) close(sfd);
    unlink(sock_path());
    sfd = cfd = -1;
}

int ray_agent_decide(int tick, float px, float py, int og,
                     float pit, int won, int dead)
{
    if (cfd < 0) return -1;
    send_state(tick, px, py, og, pit, won, dead);
    char b[256];
    ssize_t r = recv(cfd, b, sizeof b - 1, 0);
    if (r <= 0) { close(cfd); cfd = -1; return -1; }
    b[r] = '\0';
    return strstr(b, "jump") ? 1 : 0;
}

void ray_agent_send(int tick, float px, float py, int og,
                    float pit, int won, int dead)
{
    send_state(tick, px, py, og, pit, won, dead);
}

int ray_agent_poll(void)
{
    if (cfd < 0) return -1;

    // Drain everything currently available (non-blocking).
    for (;;) {
        if (rlen >= (int)sizeof rbuf - 1) rlen = 0;   // overflow guard
        ssize_t r = recv(cfd, rbuf + rlen, sizeof rbuf - 1 - rlen, 0);
        if (r > 0) rlen += r;
        else if (r == 0) { close(cfd); cfd = -1; break; }
        else break;   // EAGAIN / no data
    }

    // Return the most recent complete action (later actions supersede older).
    int result = -1;
    char *nl;
    while (rlen > 0 && (nl = memchr(rbuf, '\n', rlen)) != NULL) {
        int consumed = (int)(nl - rbuf) + 1;
        *nl = '\0';
        result = strstr(rbuf, "jump") ? 1 : 0;
        memmove(rbuf, rbuf + consumed, rlen - consumed);
        rlen -= consumed;
    }
    return result;
}
