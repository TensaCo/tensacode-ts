---
library_name: tensorcode
tags:
- tensorcode
---

# Classify

This artifact stores the `tensorcode.ops.vec.classify.Classify` architecture configuration and model weights.

## Loading

Install the compatible TensorCode library and model dependencies, then load this local directory or its Hugging Face repository ID:

```python
from tensorcode.ops.vec.classify import Classify

model = Classify.from_pretrained("./model")
```

## Training and evaluation

This generated card does not establish training provenance, task competence, evaluation results, or license. The publisher should document these before distributing a trained model. Session history and optimizer state are not included in the model artifact.
