import json
import os
import re
import csv

results = {}
for fn in os.listdir("."):
    print(fn)
    if "json" in fn:
        with open(fn, "r") as f:
            j = json.loads(f.readline())
            results[j["logname"]] = j

for k, v in results.items():
    with open(k, "r") as f:
        lines = f.readlines()
        r = re.compile("(\[.*\])?(?P<metric>.*)=(?P<value>.*)")
        for l in lines:
            match = r.findall(l)
            if len(match) > 0:
                m = match[0]
                if "time" in m[1]:
                    metric = m[1].strip()
                    value = m[2].strip()
                    v.update({metric: value})

with open("result.csv", "w") as f:
    c = csv.writer(f)
    keys = None
    for k, v in results.items():
        if not keys:
            keys = list(v.keys())
            c.writerow(keys)
        c.writerow([v[k] for k in keys])


