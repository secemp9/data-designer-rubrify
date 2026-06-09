# data-designer-rubrify

A [DataDesigner](https://github.com/gretelai/data-designer) plugin that adds a `rubrify-judge` column type. It evaluates generated text against a compiled [rubrify](https://github.com/secemp9/rubrify) `RubricBundle`, producing a normalized score, per-criterion judgment details, and a pass/fail decision for every row in a DataDesigner pipeline. The plugin bridges rubrify's `Judge` engine to DataDesigner's model provider system so that no separate LLM configuration is required beyond what DataDesigner already manages.

## Installation

```bash
pip install data-designer-rubrify
```

The package registers itself via the `data_designer.plugins` entry point. No manual registration is needed -- DataDesigner discovers it automatically on import.

## Creating a rubric bundle

The plugin requires a **compiled (locked) rubric bundle** serialized as JSON. Use rubrify's compiler to produce one:

```python
from rubrify import Rubric, compile_rubric

rubric = Rubric.from_yaml("my_rubric.yaml")
result = compile_rubric(rubric)

if not result.ok:
    print("Audit issues:", result.issues)

# Serialize the locked bundle to JSON
bundle_json = result.bundle.model_dump_json(indent=2)
with open("my_rubric_bundle.json", "w") as f:
    f.write(bundle_json)
```

The resulting `my_rubric_bundle.json` file is what you pass to `rubric_path` in the column config.

## YAML config example

```yaml
columns:
  # ... upstream columns that produce 'prompt' and 'response' ...

  quality_score:
    column_type: rubrify-judge
    target_column: response
    context_column: prompt
    model_alias: judge_model
    rubric_path: ./my_rubric_bundle.json
    judge_temperature: 0.0
    judge_max_tokens: 2048
    parallel_criteria: false
```

`model_alias` must match an alias defined in the `DataDesignerConfigBuilder` model provider setup. The plugin reads the provider's endpoint, model ID, and API key from DataDesigner's model registry and constructs the rubrify `Judge` internally.

## Config fields

All fields on `RubrifyColumnConfig`:

| Field | Type | Default | Description |
|---|---|---|---|
| `column_type` | `Literal["rubrify-judge"]` | `"rubrify-judge"` | Discriminator. Must be `"rubrify-judge"`. |
| `target_column` | `str` | *required* | Name of the column whose cell values are evaluated against the rubric. |
| `model_alias` | `str` | *required* | Alias of the model configuration to use as the judge LLM. Must match an alias in the DataDesigner model registry. |
| `rubric_path` | `str \| None` | `None` | File-system path to a compiled rubric bundle JSON file. Relative paths are resolved against cwd. Mutually exclusive with `rubric_json`. |
| `rubric_json` | `str \| None` | `None` | Inline compiled rubric bundle as a JSON string. Mutually exclusive with `rubric_path`. |
| `context_column` | `str \| None` | `None` | Optional column supplying additional context for the judge (e.g. the original prompt). |
| `genre` | `str \| None` | `None` | Optional genre tag to filter applicable criteria within the rubric. |
| `judge_temperature` | `float` | `0.0` | Sampling temperature for the judge model. |
| `judge_max_tokens` | `int` | `2048` | Maximum tokens the judge model may generate per evaluation. |
| `parallel_criteria` | `bool` | `False` | If `True`, evaluate criteria concurrently rather than sequentially. |

Exactly one of `rubric_path` or `rubric_json` must be provided. Supplying both or neither raises a `ValueError`.

## Output columns

For a column named `quality_score`, the generator produces:

| Column | Type | Content |
|---|---|---|
| `quality_score` | `float \| None` | Normalized aggregate score from the rubric evaluation. |
| `quality_score__judgments` | `str \| None` | JSON-serialized list of per-criterion judgment dicts. |
| `quality_score__decision` | `str \| None` | Overall pass/fail decision string. |

All three columns are set to `None` when the target cell is empty/null or when evaluation raises an exception.

## How the judge model is resolved

The plugin does not accept raw API keys or model IDs directly. Instead, it reads model configuration from DataDesigner's model registry using the `model_alias` field:

1. `model_alias` is looked up in DataDesigner's `ModelRegistry` to obtain the provider name, model ID, endpoint URL, and API key.
2. The plugin tries to find the model in `harn_ai`'s built-in catalog via `harn_ai.models.get_model(provider, model_id)`.
3. If the model is not in the catalog (e.g. a custom or private endpoint), a minimal `harn_ai.types.Model` is constructed using a provider-to-API-format mapping that covers OpenAI, Anthropic, Google, Mistral, DeepSeek, Groq, Cerebras, xAI, OpenRouter, Fireworks, and Together.
4. For the API key, the plugin first checks the DataDesigner provider config; if no key is set there, it falls back to environment-based discovery via `harn_ai.env_api_keys.get_env_api_key`.
5. A rubrify `Judge` is constructed with the resolved model, API key, and the `judge_temperature` / `judge_max_tokens` / `parallel_criteria` settings from the column config.

## Requirements

- Python >= 3.12
- `rubrify >= 0.1.4`
- `data-designer-config`
- `data-designer-engine`
