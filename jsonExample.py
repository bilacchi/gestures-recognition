import json
from datetime import datetime, timedelta
import numpy as np

n = 10
dates = [(datetime.now()-timedelta(days=7*i)).strftime('%y%m%d') for i in range(n)]
path = ['mesh/brenda2.ply', 'mesh/brenda4.ply']


jsonstr = {
    "Gordura":{
        "date":dates,
        "file":np.random.choice(path, n).tolist()
    },
    "Postura":{
        "date":dates,
        "file":np.random.choice(path, n).tolist()
    }
}

with open('timeline.json', 'w') as jsonf:
    json.dump(jsonstr, jsonf)