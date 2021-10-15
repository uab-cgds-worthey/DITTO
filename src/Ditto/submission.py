import json
import pandas as pd

json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json", 'r'))

fnames = []
for samples in json_file['test'].keys():
#for train_test in json_file.keys():
#   if "TEST" in train_test:
        #for samples in json_file[train_test].keys():
          if "PROBAND" in samples:
                  #fnames.append(train_test+samples)
            fnames.append(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/annotated_vcf/test/{samples}/combined_predictions_100.csv")#, sep=':')
#print(fnames)
model = pd.concat((pd.read_csv(f, sep=':') for f in fnames), ignore_index=True)
model['SD']=0
model['C']='*'
model.to_csv("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/ditto_model_debugged_annotated_vcf.txt", index=False, sep=':')
