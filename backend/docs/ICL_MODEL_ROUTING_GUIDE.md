# NL→GraphQL Translation – ICL & Model-Routing Guide

> This file explains **how in-context learning (ICL) examples flow through the backend**, how the **model & prompts are built**, and gives you a **step-by-step recipe** to rebuild the pipeline from scratch.

---

## 1. In-Context Learning (ICL) Examples

| Location | Purpose |
|----------|---------|
| `backend/config/icl_examples.py` | Stores seed pairs of **natural language → GraphQL**. |

```python
INITIAL_ICL_EXAMPLES = [
    {"natural": "Get all users…", "graphql": "query { users { name email } }"},
    …
]
```

* `get_initial_icl_examples()` returns these as **formatted strings**:
  ```text
  Natural: Get all users…
  GraphQL: query { users { … } }
  ```
* `TranslationService` adds the **first 3** examples to every request *(override by passing your own `icl_examples` list to the endpoint).*  
* Edit the list or remove the `[:3]` slice to change quantity.

---

## 2. Prompt Assembly & Model Routing

**Files involved**

* `backend/services/translation_service.py` – builds prompts.
* `backend/services/ollama_service.py` – talks to Ollama (or any LLM).
* `backend/config/settings.py` – holds defaults (model, temperature…).

### Prompt stack

```mermaid
graph TD;
  A[System Prompt]
  B[User Prompt]
  A -->|messages[0]| LLM
  B -->|messages[1]| LLM
```

1. `_build_system_prompt()`
   * Fixed rules (`You are a GraphQL expert… return JSON`).
   * Optional **Schema Context** (if caller provides it).
   * **Examples** section – the ICL list above.
2. User prompt wraps the natural-language query.
3. Passed to `ollama_service.chat_completion()` as:
   ```python
   messages = [
       {"role": "system", "content": system_prompt},
       {"role": "user",   "content": user_prompt},
   ]
   ```

### Model selection

* Endpoint may send `model` – otherwise `settings.ollama.default_model` is used (`.env: OLLAMA_DEFAULT_MODEL=`).
* To swap providers create a new service with the same `chat_completion()` interface and inject it.

---

## 3. Re-implement From Scratch (10-step recipe)

```text
my-nl2graphql/
├─ backend/
│  ├─ app.py               # FastAPI entry
│  ├─ config/
│  │   ├─ settings.py      # env vars & defaults
│  │   └─ icl_examples.py  # seed pairs
│  ├─ services/
│  │   ├─ llm_service.py   # wrapper over Ollama / OpenAI / …
│  │   └─ translation_service.py
│  └─ routes/translation.py
└─ requirements.txt
```

1. **Settings** – store `base_url`, `default_model`, `temperature`, etc.
2. **ICL seeds** – list pairs, export helper `get_examples()`.
3. **LLM wrapper** – small async function calling provider's REST API.
4. **TranslationService** – assemble prompts, parse JSON result.
5. **FastAPI route** – `POST /translate` that calls the service.
6. **Run** – `uvicorn app:app --reload`.

Swap the provider → change only `llm_service.py`.  
Tune outputs → edit `_build_system_prompt()` or ICL list.

---

## 4. Quick-Reference: Where to Tweak

| Want to change… | Edit This |
|-----------------|-----------|
| Default model / temp / tokens | `.env` → `OLLAMA_*` or `config/settings.py` |
| Example pairs | `config/icl_examples.py` |
| Prompt rules / JSON schema | `_build_system_prompt()` in `translation_service.py` |
| Per-request model / examples | Front-end payload: `{ model, icl_examples }` |

---

**Happy hacking!** 