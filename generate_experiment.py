import os
from utils.data import ALL_DATASETS

"""Create a command to run all the experiments"""

configurations = os.listdir("configurations")
configurations.remove("default.json")
configurations = [config[:-5] for config in configurations if config.endswith(".json")]
configurations.sort()
configurations = sorted(configurations, key=len)

commands = []

for dataset in ALL_DATASETS.keys():
    for config in configurations:
        commands.append(f"python train.py -d {dataset} -c {config} -b 2 -e 100 -l 0.01")
    

cmd = " && ".join(commands) + " && echo \"\" && echo \"\" && echo \"All experiments finished\" && exit"
print(cmd)