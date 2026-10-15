"""Registry of text sources.

A source yields raw documents for a split. ``split="train"`` is the stream the
training pool (and, for sources without an official test split, the held-out
docs) is carved from. ``split="test"`` is an *official* held-out split; sources that
set ``has_official_test`` use it for their test set.

Add a source::

    @DATA_SOURCES.register("my_corpus_el")
    class MyCorpus(HFSource):
        language = "el"
        path, config, text_field = "me/my_corpus", None, "text"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from ..registry import Registry
from ..utils import LOG, text_hash

DATA_SOURCES = Registry("data source")


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


class DataSource:
    language: str = "?"
    has_official_test: bool = False
    eval_only: bool = False
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})

    def iter_docs(self, split: str, seed: int, shuffle_buffer: int) -> Iterator[Doc]:  # pragma: no cover
        raise NotImplementedError


class HFSource(DataSource):
    """A Hugging Face dataset read in streaming mode."""

    path: str = ""
    config: Optional[str] = None
    text_field: str = "text"
    id_field: Optional[str] = "id"
    train_split: str = "train"
    test_split: Optional[str] = None

    def _load(self, split: str):
        from datasets import load_dataset

        hf_split = self.test_split if split == "test" else self.train_split
        if hf_split is None:
            raise ValueError(f"{type(self).__name__} has no '{split}' split")
        LOG.info("streaming %s [%s] split=%s", self.path, self.config, hf_split)
        return load_dataset(self.path, self.config, split=hf_split, streaming=True)

    def iter_docs(self, split: str, seed: int, shuffle_buffer: int) -> Iterator[Doc]:
        ds = self._load(split)
        if shuffle_buffer and shuffle_buffer > 1:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        for row in ds:
            text = row.get(self.text_field) or ""
            did = str(row.get(self.id_field)) if self.id_field and row.get(self.id_field) is not None else text_hash(text)
            yield Doc(id=f"{self.name_prefix()}:{did}", text=text, meta={k: row[k] for k in self.meta_fields() if k in row})

    def name_prefix(self) -> str:
        return type(self).__name__

    def meta_fields(self) -> List[str]:
        return []


# ----------------------------------------------------------------------------- Greek


@DATA_SOURCES.register("wikipedia_el")
class WikipediaEl(HFSource):
    language = "el"
    description = "Greek Wikipedia, 2023-11-01 dump (CC-BY-SA)."
    path, config = "wikimedia/wikipedia", "20231101.el"
    meta_fields = lambda self: ["title", "url"]  # noqa: E731


@DATA_SOURCES.register("fineweb2_el")
class FineWeb2El(HFSource):
    language = "el"
    has_official_test = True
    description = "FineWeb-2 ell_Grek (ODC-By); its official test split is used for held-out web text."
    path, config, test_split = "HuggingFaceFW/fineweb-2", "ell_Grek", "test"
    meta_fields = lambda self: ["url", "dump"]  # noqa: E731


@DATA_SOURCES.register("culturax_el")
class CulturaXEl(HFSource):
    language = "el"
    description = "CulturaX el (gated: accept terms on the Hub and set HF_TOKEN)."
    path, config, id_field = "uonlp/CulturaX", "el", None
    meta_fields = lambda self: ["url"]  # noqa: E731


# ----------------------------------------------------------------------------- English


@DATA_SOURCES.register("fineweb_edu_en")
class FineWebEduEn(HFSource):
    language = "en"
    description = "FineWeb-Edu sample-10BT: SmolLM2's own pretraining distribution (replay + 'seen-distribution' eval)."
    path, config = "HuggingFaceFW/fineweb-edu", "sample-10BT"
    meta_fields = lambda self: ["url"]  # noqa: E731


@DATA_SOURCES.register("fineweb_edu_2025_en")
class FineWebEdu2025En(HFSource):
    language = "en"
    description = "FineWeb-Edu crawl CC-MAIN-2025-26 (crawled after SmolLM2's release)."
    path, config = "HuggingFaceFW/fineweb-edu", "CC-MAIN-2025-26"
    meta_fields = lambda self: ["url"]  # noqa: E731


@DATA_SOURCES.register("bbc_news_2025_en")
class BBCNews2025(DataSource):
    language = "en"
    eval_only = True
    has_official_test = True
    description = "BBC News articles published Jan-Jun 2025 (post-cutoff English held-out)."

    def iter_docs(self, split: str, seed: int, shuffle_buffer: int) -> Iterator[Doc]:
        from datasets import load_dataset

        months = self.params.get("months", ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"])
        rows = []
        for m in months:
            LOG.info("loading RealTimeData/bbc_news_alltime %s", m)
            for r in load_dataset("RealTimeData/bbc_news_alltime", m, split="train"):
                rows.append(r)
        import random

        random.Random(seed).shuffle(rows)
        for r in rows:
            text = r.get("content") or ""
            yield Doc(id=f"bbc:{r.get('link') or text_hash(text)}", text=text, meta={"date": r.get("published_date")})


class _Belebele(DataSource):
    eval_only = True
    has_official_test = True
    lang_code = ""

    def iter_docs(self, split: str, seed: int, shuffle_buffer: int) -> Iterator[Doc]:
        """Unique FLORES passages, in dataset order, with ids aligned across languages."""
        from datasets import load_dataset

        seen: Dict[str, int] = {}
        texts_seen = set()
        for r in load_dataset("facebook/belebele", self.lang_code, split="test"):
            passage = r["flores_passage"]
            if passage in texts_seen:
                continue
            texts_seen.add(passage)
            k = seen.get(r["link"], 0)
            seen[r["link"]] = k + 1
            yield Doc(id=f"belebele:{r['link']}|{k}", text=passage, meta={"link": r["link"]})


@DATA_SOURCES.register("belebele_el")
class BelebeleEl(_Belebele):
    language = "el"
    lang_code = "ell_Grek"
    description = "Belebele/FLORES passages in Greek (parallel with belebele_en)."


@DATA_SOURCES.register("belebele_en")
class BelebeleEn(_Belebele):
    language = "en"
    lang_code = "eng_Latn"
    description = "Belebele/FLORES passages in English (parallel with belebele_el)."


def make_source(name: str, params: Optional[Dict[str, Any]] = None) -> DataSource:
    return DATA_SOURCES.get(name)(params)
