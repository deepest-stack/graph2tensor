Model
=============================

.. currentmodule:: graph2tensor.model.data


data
---------------------------

EgoTensorGenerator Class
```````````````````````````

.. autoclass:: EgoTensorGenerator
    :members:
    :special-members: __call__


SkipGramGenerator4DeepWalk Class
``````````````````````````````````

.. autoclass:: SkipGramGenerator4DeepWalk
    :members:
    :special-members: __call__


SkipGramGenerator4Node2Vec Class
``````````````````````````````````

.. autoclass:: SkipGramGenerator4Node2Vec
    :members:
    :special-members: __call__


SkipGramGenerator4MetaPath2Vec Class
`````````````````````````````````````````

.. autoclass:: SkipGramGenerator4MetaPath2Vec
    :members:
    :special-members: __call__



.. currentmodule:: graph2tensor.model.layers


layers
---------------------------

GCNConv Class
```````````````````````````

.. autoclass:: GCNConv
    :members: __init__


GINConv Class
```````````````````````````

.. autoclass:: GINConv
    :members: __init__

GATConv Class
```````````````````````````

.. autoclass:: GATConv
    :members: __init__

RGCNConv Class
```````````````````````````

.. autoclass:: RGCNConv
    :members: __init__

UniMP Class
```````````````````````````

.. autoclass:: UniMP
    :members: __init__


EmbeddingEncoder Class
```````````````````````````

.. autoclass:: EmbeddingEncoder
    :members: __init__

OnehotEncoder Class
```````````````````````````

.. autoclass:: OnehotEncoder
    :members: __init__

IntegerLookupEncoder Class
```````````````````````````

.. autoclass:: IntegerLookupEncoder
    :members: __init__

StringLookupEncoder Class
```````````````````````````

.. autoclass:: StringLookupEncoder
    :members: __init__


MeanPathReduce Class
```````````````````````````

.. autoclass:: MeanPathReduce
    :members: __init__


MaxPathReduce Class
```````````````````````````

.. autoclass:: MaxPathReduce
    :members: __init__


SumPathReduce Class
```````````````````````````

.. autoclass:: SumPathReduce
    :members: __init__


ConcatPathReduce Class
```````````````````````````

.. autoclass:: ConcatPathReduce
    :members: __init__



.. currentmodule:: graph2tensor.model.explainer


explainers
---------------------------

IntegratedGradients Class
```````````````````````````

.. autoclass:: IntegratedGradients
    :members: explain



.. currentmodule:: graph2tensor.model.models


models
---------------------------

MessagePassing Class
```````````````````````````

.. autoclass:: MessagePassing
    :members: __init__


DeepWalk Class
```````````````````````````

.. autoclass:: DeepWalk
    :members: get_node_embedding, most_similar
    :special-members: __init__



Node2Vec Class
```````````````````````````

.. autoclass:: Node2Vec
    :members: get_node_embedding, most_similar
    :special-members: __init__


MetaPath2Vec Class
```````````````````````````

.. autoclass:: MetaPath2Vec
    :members: get_node_embedding, most_similar
    :special-members: __init__