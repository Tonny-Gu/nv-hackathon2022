from math import log10
import os
import json

CMD = "srun -N {N} -n {n} ./main.{lib}.bin --normal --num_elems={ne} --num_result={nr} > {logname}"
PREFIX = "exp1"

def gen_param():
    for lib in ["base"]:
    # for lib in ["nv.mpi", "nv.mgdr", "nv.nccl"]:
        for n, N in [(1, 1),]:
            for ne in [100000, 1000000, 10000000, 100000000, 1000000000]:
                yield {
                    "N": N,
                    "n": n,
                    "lib": lib,
                    "ne": ne,
                    "nr": int(ne * 0.01),
                    "logname": "%s_%s_%s_%s_E%s.log" % (PREFIX, lib, n, N, int(log10(ne)))
                }

for conf in gen_param():
    cmd = CMD.format(**conf)
    print(cmd)
    os.system(cmd)
    # with open(conf["logname"] + ".json", "w") as f:
    #     f.write(json.dumps(conf))