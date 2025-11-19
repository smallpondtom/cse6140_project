import time

def two_opt(tour, dist, cutoff_seconds, start_time):
    """
    Simple 2-opt hill climbing with time cutoff.

    tour: initial tour (list of city indices)
    dist: distance matrix
    cutoff_seconds: max allowed runtime (float or int, in seconds)
    start_time: time.time() value when algorithm started

    Returns:
        (best_tour, best_cost)
    """
    n = len(tour)

    def tour_cost(t):
        c = 0
        for i in range(n):
            c += dist[t[i]][t[(i+1) % n]]
        return c

    best_tour = tour[:]   # copy
    best_cost = tour_cost(best_tour)

    improved = True
    while improved:
        # Check cutoff at top of each outer iteration
        if time.time() - start_time >= cutoff_seconds:
            break

        improved = False
        for i in range(1, n - 2):
            # Check cutoff periodically in inner loop too
            if time.time() - start_time >= cutoff_seconds:
                break

            for j in range(i + 1, n - 1):
                # Again, keep it safe
                if time.time() - start_time >= cutoff_seconds:
                    break

                # 2-opt swap: reverse segment [i, j]
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[j], best_tour[(j + 1) % n]

                old_cost = dist[a][b] + dist[c][d]
                new_cost = dist[a][c] + dist[b][d]

                if new_cost < old_cost:
                    # apply 2-opt move
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    best_cost = best_cost - old_cost + new_cost
                    improved = True
                    break  # restart scanning with new tour

            if improved:
                break

    return best_tour, best_cost
