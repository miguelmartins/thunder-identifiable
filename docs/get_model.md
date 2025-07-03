In addition to benchmarking, we also provide a [`get_model_from_name`](https://mics-lab.github.io/thunder/api/#thunder.models.get_model_from_name) function through our API to extract embeddings using any foundation model we support on your own data. It returns a Pytorch model callable, associated transforms (to apply to images to extract embeddings from) and a function to get embeddings from a batch of transformed images, the model callable, and the type of embeddings (pooled or spatial) to return.

Below is an example if you want to get the Pytorch callable, transforms and function to extract embeddings for `uni2h`:

```python
from thunder.models import get_model_from_name

model, transform, get_embeddings = get_model_from_name("uni2h", device="cuda")
```

You can then extract the embedding of a PIL image `im`  as follows:

```python
pooled_emb = get_embeddings(transform(im), model, pooled_emb=True)
spatial_emb = get_embeddings(transform(im), model, pooled_emb=False)
```
