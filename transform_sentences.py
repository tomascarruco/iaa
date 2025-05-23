# conda activate project_venv
 
from sentence_transformers import SentenceTransformer

import numpy as np
import os
import torch
import duckdb
import urllib.parse

MAX_CORPUS_SIZE = 1_000


create_table = '''
create table if not exists poem_embeddings (
  title varchar,
  embedding FLOAT[128]
);
'''

if __name__ == "__main__":
    model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1",
        # model_kwargs={"torch_dtype": "float16"}
    )

    con = duckdb.connect(config= {'threads': 1}, database="file.db")
    con.sql(create_table)

    embeddings: dict[str, torch.Tensor] = {}
    for filename in os.listdir("./poems"):
        with open("./poems/" + filename ,"r") as f:
            content_embedding = model.encode(f.read(), show_progress_bar=True, convert_to_tensor=True, precision="binary")
            embeddings[filename] = content_embedding

    for key, val in embeddings.items():
        # embeding = val.numpy().tolist()
        # title = urllib.parse.quote(key)
        # string = f"insert into poem_embeddings values ('{title}',{embeding});"
        # con.sql(string)
        emb_fl = f"{key}_embedding.npy"
        np.save(os.path.join("./embeddings/", emb_fl), val)

    # con.table("poem_embeddings").show()

    # query = [ "holy dreams, of lovely beams" ]
    # query_embeding = model.encode(query[0], show_progress_bar=True, convert_to_tensor=True, precision="binary")
    # embeding_q = query_embeding.numpy().tolist()

    # string = f'''
    # select title, score from (
    #     select *, array_cosine_similarity(embedding, Cast({embeding_q} as Float[128])) as score
    #     from poem_embeddings) sq
    #     where score is not null
    #     order by score Desc limit 5
    # ; 
    # '''
    # rel = con.sql(string)
    # rel.show()
    
    
    # queries = [
    #     "Fatherhood",
    # ]

    # q_embedded = model.encode(queries[0], convert_to_tensor=True)

    # print(content_embedding)
    # print(binary_c_embeding)

    # hits = util.semantic_search(query_embeddings=q_embedded, corpus_embeddings=content_embedding, score_function=util.dot_score)


