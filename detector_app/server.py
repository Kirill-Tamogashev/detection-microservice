import multiprocessing as mp

from grpc_server import serve
from http_server import app


def run_server(rank):
    if rank == 0:
        app.run(host='0.0.0.0', port=8080)
    else:
        serve()


if __name__ == '__main__':
    workers = []
    for rank in range(2):
        worker = mp.Process(
            target=run_server, args=(rank,)
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()