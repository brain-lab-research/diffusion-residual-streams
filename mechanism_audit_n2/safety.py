"""Exclusive use of a GPU that is idle and has no other compute process."""
import csv
import fcntl
import io
import os
import subprocess
import time
from pathlib import Path


def query(arguments):
    result = subprocess.run(['nvidia-smi'] + arguments, check=True, capture_output=True, text=True, timeout=15)
    return list(csv.DictReader(io.StringIO(result.stdout), skipinitialspace=True))


class GPUReservation:
    def __init__(self, gpu, policy):
        if gpu in policy['never_use_training_gpus']:
            raise RuntimeError('This GPU is reserved for an existing training job')
        self.gpu, self.policy = str(gpu), policy
        self.lock = Path('/tmp/mhc_mechanism_gpu' + str(gpu) + '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observations = []
        for i in range(2):
            rows = query(['--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,nounits'])
            row = next(x for x in rows if x['index'].strip() == str(gpu))
            self.uuid = row['uuid'].strip()
            applications = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,nounits'])
            others = [x for x in applications if x['gpu_uuid'].strip() == self.uuid]
            if others or float(row['utilization.gpu [%]']) > policy['max_idle_utilization_percent'] or float(row['memory.used [MiB]']) > policy['max_idle_memory_mib']:
                raise RuntimeError('Requested GPU is not idle; no model was allocated')
            observations.append(row)
            if i == 0:
                time.sleep(policy['two_observations_seconds'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        self.observations = observations
        self.last_checked = 0

    def check(self, force=False):
        if not force and time.monotonic() - self.last_checked < 5:
            return
        applications = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,nounits'])
        if any(x['gpu_uuid'].strip() == self.uuid and int(x['pid']) != os.getpid() for x in applications):
            raise RuntimeError('Another compute process appeared on the analysis GPU; stopping at a batch boundary')
        self.last_checked = time.monotonic()
