Quickstart
==========

You can :doc:`install <install>` ``ffp`` through:

.. code-block:: bash

   pip install ffp

And use embeddings by:

.. code-block:: python

   import ffp
   # load finalfusion embeddings
   embeddings = ffp.load_finalfusion("/path/to/embeddings.fifu")
   # embedding lookup
   embedding = embeddings["Test"]
   # embedding lookup with default value
   embedding = embeddings.embedding("Test", default=0)
   # access storage and calculate dot product with an embedding
   storage = embedding.dot(embeddings.storage)
   # print 10 first vocab items
   print(embeddings.vocab.words[:10])
   # print embeddings metadata
   print(embeddings.metadata)

``ffp`` exports most common-use functions and types in the top level.
See :doc:`Top-Level Exports <modules/re-exports>` for an overview.

These re-exports are also available in their respective sub-packages and modules.
The full API documentation can be foud :doc:`here <modules/api>`.