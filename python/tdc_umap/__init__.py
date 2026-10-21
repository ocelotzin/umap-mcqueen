"""``tdc_umap``: el clasificador de umap-mcqueen y el motor ``engine='umap'``
para el Peeler de tridesclous.

* ``Clasificador`` viene del módulo compilado (``tdc_umap._core``, Rust): carga
  un modelo que entrenó ``umap-mcqueen`` y clasifica lotes de formas de onda.
* ``registrar()`` da de alta el motor en tridesclous. Es explícito: importar
  ``tdc_umap`` no toca tridesclous, y tridesclous no sabe que este paquete existe.

    import tdc_umap
    tdc_umap.registrar()
    peeler.change_params(engine='umap', catalogue=..., umap_modelo='modelos/s1')
"""

from ._core import Clasificador

__all__ = ['Clasificador', 'registrar']


def registrar():
    """Da de alta ``engine='umap'`` en ``tridesclous.peeler.peeler_engines``.

    ``Peeler.change_params`` busca el motor en ese diccionario, así que basta con
    ponerlo ahí; tridesclous no se modifica. Llamarlo dos veces no hace nada
    nuevo. Devuelve la clase del motor.
    """
    from tridesclous import peeler
    from .peeler_engine_umap import PeelerEngineUmap
    peeler.peeler_engines['umap'] = PeelerEngineUmap
    return PeelerEngineUmap
