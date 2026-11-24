# grouille-inkling

Specialized Metal inference for [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), in the same spirit as [grouille](https://github.com/franckverrot/grouille).

## Usage

Requirements: it needs at least Rust 1.87.

```sh
# optional: build the mmap pack
python3 spike/convert_gpk.py

cargo build -p grouille-inkling --release
./target/release/inkling-generate --bench 3 --max 64 "Write something long."
```

Arguments:
* `--weights` accepts the `.gpk` or the HF `.safetensors`.  Default prefers `weights/inkling2mini.gpk` when present.
* `--draft` turns the drafter on.  It needs greedy decoding, which is the default, and `mtp.safetensors` next to the checkpoint.  With `--timing` it also prints how many blocks it checked and how many proposals were kept.
* `INKLING_ATTN_Q8=1` moves the attention projections and the unembed to 8-bit weights during decode.  It is faster and it is lossy, so it stays off unless you ask for it.
* `INKLING_KV_F16=1` keeps the history a full-attention layer holds past the
window in f16.  It halves what a long context costs in memory, and it is
lossy too.


## Some numbers

On an M2 Max 96GB, greedy for 300 tokens after a short prompt:

| | tokens a second |
|---|---|
| mlx-vlm | 23 |
| `inkling-generate` | **38** |
| with `--draft`, Python code | **50** |
| with `--draft`, JSON | **61** |
| with `--draft`, prose | 36 |

Memory peaks at around 13 GB whereas `mlx-vlm` peaks at near 21GB.

With 133k tokens of context, memory will go as high as 19 GB, and decode will still run at about 19 tokens per second.

After comparing with `mlx-vlm`, prompt processing is pretty much on par.


## Yet another inference engine...

Sure, but most of the ones I've seen in the days following Inkling 2's release barely produced valid outputs, so this repo checks against the reference.

```sh
python3 spike/regress.py
python3 spike/ref_check.py goldens/regress/*.logits.json
INKLING_PROMPT_CHUNK=200 python3 spike/regress.py
cargo test -p grouille-inkling --release -- --test-threads=1
```



# LICENSE

[MIT](https://opensource.org/licenses/MIT).