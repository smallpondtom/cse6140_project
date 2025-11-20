import os
import time
import argparse
import numpy as np
import pandas as pd
from brute_force import brute_force
from approx import approx
from local_search.local_search import local_search
from write_output import write_output


def is_full_tour(tour):
    """Return True if `tour` is a closed tour visiting unique nodes exactly once.

    `tour` is expected to be a sequence (numpy array or list) of vertex IDs where
    the first element is repeated at the end to close the tour.
    """
    if tour is None:
        return False
    tour = list(tour)
    if len(tour) <= 1:
        return False
    # Check closure
    if tour[0] != tour[-1]:
        return False
    # All nodes except the final closing node should be unique
    return len(set(tour[:-1])) == (len(tour) - 1)


def main():
    parser = argparse.ArgumentParser(description="Run TSP algorithms over datasets and collect results")
    parser.add_argument("--data-dir", default="../data", help="Directory containing .tsp files")
    parser.add_argument("--runtimes", nargs="+", type=float, default=[20], help="Runtime cutoff(s) in seconds")
    parser.add_argument("--seed-count", type=int, default=10, help="Number of random seeds to try for LS")
    parser.add_argument("--rand-seed", type=int, default=12345, help="Random seed for reproducible seed generation")
    parser.add_argument("--output-csv", default="../result/results.csv", help="CSV file to write summary results")

    args = parser.parse_args()

    data_dir = args.data_dir
    runtimes = args.runtimes
    seed_count = args.seed_count
    rng = np.random.RandomState(int(args.rand_seed))
    seeds = [int(s) for s in rng.randint(1, 10_000, size=seed_count)]
    output_csv = args.output_csv

    # Get the list of dataset files (deterministically sorted)
    dataset_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tsp")])

    total_datasets = len(dataset_files)
    print(f"Found {total_datasets} dataset(s) in {data_dir}")

    results = []
    for idx, dataset in enumerate(dataset_files, start=1):
        dataset_path = os.path.join(data_dir, dataset)
        instance_name = os.path.splitext(dataset)[0]
        print(f"Starting dataset {idx}/{total_datasets}: {instance_name}")

        # Run LS multiple times to get a best distance and average stats
        best_distance = None
        ls_times = []
        ls_qualities = []
        ls_full_count = 0

        for runtime in runtimes:
            for seed in seeds:
                try:
                    t0 = time.perf_counter()
                    tour, distance = local_search(dataset_path, runtime, seed)
                    elapsed = time.perf_counter() - t0

                    # Persist solution
                    write_output(instance_name, "LS", runtime, tour, distance, seed)

                    # update stats
                    ls_times.append(elapsed)
                    ls_qualities.append(distance)
                    if is_full_tour(tour):
                        ls_full_count += 1

                    if best_distance is None or distance < best_distance:
                        best_distance = distance

                except Exception as e:
                    print(f"Error running LS for {dataset} runtime={runtime} seed={seed}: {e}")
                    continue

        avg_ls_time = float(np.mean(ls_times)) if ls_times else 0.0
        avg_ls_quality = float(np.mean(ls_qualities)) if ls_qualities else 0.0
        ls_full_tour_frac = (ls_full_count / max(1, len(ls_times))) if ls_times else 0.0

        print(
            f"LS finished for {instance_name}: best_distance={best_distance}, "
            f"avg_time={avg_ls_time:.4f}s, full_tour_frac={ls_full_tour_frac:.2%}"
        )

        row = {
            "Dataset": dataset,
            "LS Avg Time(s)": avg_ls_time,
            "LS Avg Sol.Quality": avg_ls_quality,
            "LS Full Tour Fraction": ls_full_tour_frac,
            "LS Best Distance": best_distance if best_distance is not None else None,
        }

        # Run BF and Approx using the best LS distance for relative error
        for alg in ["BF", "Approx"]:
            for runtime in runtimes:
                try:
                    t0 = time.perf_counter()
                    if alg == "BF":
                        tour, distance = brute_force(dataset_path, runtime)
                        write_output(instance_name, alg, runtime, tour, distance)
                    else:
                        tour, distance = approx(dataset_path, runtime)
                        write_output(instance_name, alg, runtime, tour, distance)
                    elapsed = time.perf_counter() - t0

                    rel_error = None
                    if best_distance is not None and best_distance != 0:
                        rel_error = ((distance - best_distance) / best_distance) * 100.0

                    full_tour = "Yes" if is_full_tour(tour) else "No"

                    if alg == "BF":
                        row.update({
                            "Brute Force Time(s)": elapsed,
                            "Brute Force Sol.Quality": distance,
                            "Brute Force Full Tour": full_tour,
                            "Brute Force RelError(%)": rel_error,
                        })
                    else:
                        row.update({
                            "Approx Time(s)": elapsed,
                            "Approx Sol.Quality": distance,
                            "Approx Full Tour": full_tour,
                            "Approx RelError(%)": rel_error,
                        })

                    print(f"Completed: {instance_name} with {alg} (runtime={runtime})")
                except Exception as e:
                    print(f"Error running {alg} for {instance_name} runtime={runtime}: {e}")
                    continue
            # log when an algorithm finishes all runtimes for this dataset
            print(f"{alg} completed for {instance_name} (all runtimes done)")

        results.append(row)
        print(f"Finished dataset {idx}/{total_datasets}: {instance_name}")

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
