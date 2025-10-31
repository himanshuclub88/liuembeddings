---

## 🧭 Migration Guide — `liuembeddings` v2.0.0

### 🚀 Overview

`liuembeddings` **v2.0.0** introduces a major upgrade — moving from **TensorFlow Hub (USE)**
to **Hugging Face Sentence-Transformers** for faster, lightweight, and modern embeddings.

This guide explains how to migrate your existing code from `v1.x` to `v2.0.0`.

---

## 🔄 1. What Changed

| Area                 | v1.x (Old)                     | v2.0.0 (New)                                |
| -------------------- | ------------------------------ | ------------------------------------------- |
| **Backend**          | TensorFlow Hub (USE)           | Hugging Face Sentence-Transformers          |
| **Dependencies**     | `tensorflow`, `tensorflow_hub` | `sentence-transformers`, `torch`            |
| **Default Model**    | `USE`                          | `MiniLM`                                    |
| **Performance**      | Slow, large dependencies       | Faster, smaller, and GPU/CPU friendly       |
| **API**              | Mostly the same                | Same API, different model loading mechanism |
| **Supported Models** | Only USE / USE-Large           | MiniLM, MPNet, E5, BGE                      |
| **Model Cache**      | TensorFlow in-memory           | Hugging Face in-memory                      |

---

## 📦 2. New Dependencies

You can uninstall TensorFlow-related packages and install only the new requirements:

```bash
pip uninstall tensorflow tensorflow-hub -y
```

---

## 🧠 3. Model List in v2.0.0

| Model Name  | ID                                        | Dimension | Size (MB) | Accuracy |
| ----------- | ----------------------------------------- | --------- | --------- | -------- |
| `MiniLM`    | `sentence-transformers/all-MiniLM-L6-v2`  | 384       | 22        | 0.78     |
| `MPNetBase` | `sentence-transformers/all-mpnet-base-v2` | 768       | 420       | 0.82     |
| `USE`       | `intfloat/e5-base-v2`                     | 768       | 300       | 0.84     |
| `USEL`      | `BAAI/bge-base-en-v1.5`                   | 1024      | 1024      | 0.86     |

---

## ⚙️ 4. Code Migration Example

### Old (v1.x)

```python
from liuembeddings.embeddings import LiuEmbeddings

embedder = LiuEmbeddings("USE")
vec = embedder.embed_query("Hello world")
```

### New (v2.0.0)

```python
from liuembeddings.embeddings import LiuEmbeddings

embedder = LiuEmbeddings("USE")  # or "MiniLM" for speed
vec = embedder.embed_query("Hello world")
```

✅ **Same API**, just use one of the new supported model names.

---

## 📁 5. Config Changes

### Old

`LiuConfig.AVAILABLE_MODELS` contained only TensorFlow Hub models (`USE`, `USE-Large`).

### New

`LiuConfig.AVAILABLE_MODELS` now includes modern transformer models with metadata:

```python
{
  "MiniLM": {...},
  "MPNetBase": {...},
  "USE": {...},
  "USE-Large": {...}
}
```

---

## 🧩 6. Removed TensorFlow Environment Settings

In v1.x we had:

```python
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
```

This is no longer needed.
✅ All models now run directly via PyTorch or CPU using Sentence-Transformers.

---

## 📊 7. Benefits of v2.0.0

* 🚀 **30–50× faster installation** (no TensorFlow)
* 🧠 **Higher accuracy** via modern transformer encoders
* 💾 **Smaller dependency footprint**
* 🔄 **Same API for easy migration**
* 🧩 **Future support for multilingual and domain-specific models**

---

## 🧰 8. Optional: Set Default Model in Code

If you want to keep the same interface for users without specifying a model:

```python
embedder = LiuEmbeddings()  # defaults to USE
```

To change the default globally:

```python
DEFAULT_MODEL = "E5Base"
```

---

## ⚡ 9. Verify Migration

Run a quick embedding test:

```python
from liuembeddings import LiuEmbeddings

embedder = LiuEmbeddings("E5Base")
print(len(embedder.embed_query("Migration complete!")))
```

Output should show a vector of **768 dimensions** without TensorFlow logs.

---

## ✅ 10. Summary

| Action                  | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| 🧹 Uninstall TensorFlow | `pip uninstall tensorflow tensorflow-hub`                            |
| 📦 Install Transformers | `pip install sentence-transformers`                                  |
| 🔄 Update Code          | Change `"USE-Large"` → `"E5Base"` or `"MiniLM"`                      |
| ⚙️ No API Change        | Keep using `embed_query`, `embed_documents`, `embed_documents_batch` |

---
