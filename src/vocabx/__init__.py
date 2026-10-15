"""vocabx: vocabulary expansion vs. plain continued pretraining, compared fairly.

Extension points (each a ``Registry``; register a new entry and name it in a config):

* ``vocabx.data.sources.DATA_SOURCES``           - corpora (train pools / eval sets)
* ``vocabx.tokenization.extenders.EXTENDERS``    - how new tokens are created
* ``vocabx.modeling.init.INITIALIZERS``          - how their embeddings are initialised
* ``vocabx.evaluation.downstream.TASKS``         - zero-shot cloze tasks
* training stages / parameter groups             - ``train.stages`` in the config
"""

__version__ = "0.1.0"


def load_all_plugins() -> None:
    """Import every module that registers built-in plugins."""
    from .data import sources  # noqa: F401
    from .evaluation import downstream  # noqa: F401
    from .modeling import init  # noqa: F401
    from .tokenization import extenders  # noqa: F401
