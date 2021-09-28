import json
import pandas as pd

json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata.json", 'r'))

fnames = []
for train_test in json_file.keys():
        for samples in json_file[train_test].keys():
          if "PROBAND" in samples:
                  #fnames.append(train_test+samples)
            fnames.append(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/predictions/{train_test}/{samples}/combined_predictions_100.csv")#, sep=':')
#print(fnames)
model = pd.concat((pd.read_csv(f, sep=':') for f in fnames), ignore_index=True)
model.to_csv("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto_model_1.txt", index=False, sep=':')
