import json
from time import sleep
for i in range(100):
    for j in range(100):
        data = {
                'x':i,
                'y':j
                }
        with open('media/t.json', 'w') as f:
            json.dump(data, f,indent=4)
        sleep(0.2)