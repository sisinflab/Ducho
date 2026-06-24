from huggingface_hub import HfApi
from typing import Literal

DUCHO_TASK_MAP = {
    # Ducho modality → HF pipeline_tag(s)
    "visual":          ["image-feature-extraction", "image-classification"],
    "textual":         ["feature-extraction", "sentence-similarity", "text-classification"],
    "audio":           ["audio-classification", "automatic-speech-recognition"],
    "visual-textual":  ["image-to-text", "visual-question-answering", "zero-shot-image-classification"],
}

def search_models(
    modality: Literal["visual", "textual", "audio", "visual-textual"],
    query: str = None,
    library: str = None,
    limit: int = 10,
    sort: str = "downloads",
    task: str = None,
) -> list[dict]:
    api = HfApi()
    tasks = DUCHO_TASK_MAP.get(modality, [])
    if task is not None:
        tasks = [task]

    results = []
    for task in tasks:
        models = api.list_models(
            pipeline_tag=task,
            filter=library,
            search=query,
            sort=sort,
            limit=limit,
        )
        for m in models:
            results.append({
                "model_id": m.modelId,
                "task": task,
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
                "last_modified": str(m.lastModified),
                "tags": m.tags,
                "ducho_config_snippet": _make_config_snippet(modality, m.modelId),
            })
    sort_key = {"downloads": "downloads", "likes": "likes", "lastModified": "last_modified"}.get(sort, "downloads")
    results.sort(key=lambda m: m[sort_key], reverse=True)
    return results[:limit]

def _make_config_snippet(modality: str, model_id: str) -> str:
    """Returns a ready-to-use YAML snippet for Ducho config."""
    return (
        f"{modality}:\n"
        f"  items:\n"
        f"    - model: {model_id}\n"
        f"      backend: transformers\n"
        f"      output_layers: 1\n"
        f"      batch_size: 32\n"
    )