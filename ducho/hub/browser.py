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
    library: str = None,          # "transformers", "sentence-transformers", etc.
    limit: int = 10,
    sort: str = "downloads",       # or "likes", "lastModified"
) -> list[dict]:
    api = HfApi()
    tasks = DUCHO_TASK_MAP.get(modality, [])
    
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
                "task": task, # there is also m.pipeline_tag
                "downloads": m.downloads or 0, # to catch None values
                "likes": m.likes or 0, # to catch None values
                "last_modified": str(m.lastModified),
                "tags": m.tags,
                # Ready-to-paste Ducho YAML snippet:
                "ducho_config_snippet": _make_config_snippet(modality, m.modelId),
            })
    return results

def _make_config_snippet(modality: str, model_id: str) -> str:
    """Returns a ready-to-use YAML snippet for Ducho config."""
    return (
        f"{modality}:\n"
        f"  items:\n"
        f"    - model: {model_id}\n"
        f"      backend: transformers\n"
        f"      output_layers: 1\n"
        f"      batch_size: \n"
    )