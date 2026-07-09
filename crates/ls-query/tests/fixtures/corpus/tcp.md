TCP congestion control regulates how fast a sender injects packets into the
network. The sender maintains a congestion window that limits unacknowledged
data in flight, growing it when the network looks healthy and shrinking it when
loss suggests congestion.

Slow start is the opening phase: the congestion window begins at a few segments
and doubles every round-trip time, growing exponentially until it reaches the
slow-start threshold or a loss occurs. Despite the name, slow start is the
fastest-growing phase of TCP.

After the threshold, congestion avoidance takes over with additive increase:
the window grows by roughly one segment per round trip. On packet loss the
window is cut multiplicatively — the classic AIMD scheme, additive increase,
multiplicative decrease — which lets many flows share a bottleneck fairly.

When a packet is dropped during slow start, the threshold is set to half the
current window and the window resets, re-entering slow start or fast recovery
depending on how the loss was detected. Modern stacks use CUBIC or BBR, but the
AIMD intuition still explains their behavior under sustained loss.
