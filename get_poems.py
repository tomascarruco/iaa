import json
import os
from typing import List
import urllib3
from pydantic import BaseModel, FieldSerializationInfo

POEMS_API_URL = "http://poetrydb.org/random"

http_client_pool = urllib3.PoolManager()

class Poem(BaseModel):
    # Attributes
    title: str
    author: str
    lines: List[str]
    linecount: int
   
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def many_from_json_array(cls, json_arr):
        if isinstance(json_arr, str):
            items = json.loads(json_arr)
        else:
            items = json_arr
        return [cls.model_validate(item) for item in items]


# RETURNS: Poem class
def get_random_poems(poem_count = 1) -> List[Poem]:
    response = http_client_pool.request("GET", f"{POEMS_API_URL}/{poem_count}")

    contents = response.data.decode('utf-8')
    json_contents = json.loads(contents)

    return Poem.many_from_json_array(json_contents)
    

if __name__ == "__main__":
    fd = os.path.exists("./poems")
    if not fd :
        os.mkdir("./poems")

    poems = get_random_poems(100)

    for poem in poems:
        poem.lines = [line + '\n' for line in poem.lines]

    for poem in poems:
        with open(f"./poems/{poem.title} - {poem.author}", "w") as f:
            f.writelines(poem.lines)
