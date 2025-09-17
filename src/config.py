from pydantic import BaseModel
from typing import List
import json





class PoolConfig(BaseModel):
    size : int

class Config(BaseModel):
    gpu_list : List[str]
    model_name : str
    batch_size : int
    file_batch_size : int
    beam_size : int
    enable_auto_scale : bool
    max_auto_scale_limit : int
    enable_access_token : bool
    pool_config : PoolConfig



_config : Config = None
def get_config(refresh=False):
    global _config
    if _config == None or refresh:
        with open("config.json",'r') as file:
            json_config = json.load(file)
        _config = Config(**json_config)
    return _config

