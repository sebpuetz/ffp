.. ffp documentation master file, created by
   sphinx-quickstart on Fri May  1 12:35:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Finalfusion in Python
=====================

``ffp`` is a Python package to interface with `finalfusion <https://finalfusion.github.io>`__
embeddings. ``ffp`` supports all common embedding formats, including finalfusion,
fastText, word2vec binary, text and textdims.

``ffp`` integrates nicely with ``numpy`` since its :class:`ffp.storage.Storage` types can be
treated as ndarrays.

The ``finalfusion`` format revolves around :class:`ffp.io.Chunk`\ s, these are specified in the
`finalfusion spec <https://finalfusion.github.io/spec>`__. Each component class in ``ffp``
implements the :class:`ffp.io.Chunk` interface which specifies serialization and deserialization.
Any unique combination of chunks can make up :class:`ffp.Embeddings`.

Contents
--------
.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   quickstart
   install
   modules/re-exports
   modules/api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
