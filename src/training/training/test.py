import ray 
# Start Ray.
ray.init(ignore_reinit_error=True)
from functools import partial
from ray.util.multiprocessing import Pool
import time
print(f'''This cluster consists of
    {len(ray.nodes())} nodes in total
    {ray.cluster_resources()['CPU']} CPU resources in total
''')



def my_function (train, test, vars):
    #sums = train+test
    for key, value in vars.items():
        time.sleep(2)
        return train+test,key, value

pool = Pool()

variants = [{0:{1:2}},{0:{2:2}},{0:{3:3}}]
train =1
test =2
func = partial(my_function,train,test)
for result in pool.map(func, variants):
    print(result[0],result[1],result[2])
