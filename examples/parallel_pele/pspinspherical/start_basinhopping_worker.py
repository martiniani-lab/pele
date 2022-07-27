import argparse

from start_server import (create_system, get_database_params_worker,
                          get_server_uri)

from pele.concurrent import BasinhoppingWorker


def main():
    parser = argparse.ArgumentParser(description="connect worker queue")
    parser.add_argument("p", type=int, help="p-spin")
    parser.add_argument("nspins", type=int, help="number of spins")
    parser.add_argument(
        "--nsteps", type=int, help="number of basin hopping steps", default=1000
    )
    args = parser.parse_args()

    nspins = args.nspins
    p = args.p

    interactions = get_database_params_worker(nspins, p)
    system = create_system(nspins, p, interactions)

    uri = get_server_uri(nspins, p)
    worker = BasinhoppingWorker(uri, system=system)
    worker.run(args.nsteps)


if __name__ == "__main__":
    main()
