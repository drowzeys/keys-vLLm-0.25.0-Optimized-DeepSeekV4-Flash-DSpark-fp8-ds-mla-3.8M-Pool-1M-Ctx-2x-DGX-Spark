# Fresh bench/eval — 1M nvfp4_ds_mla DSV4 DSpark (2× Spark)

**Timestamp (UTC):** 2026-07-09T23:19:57Z

**API:** `http://10.100.10.3:8000/v1` · model `deepseek-v4-flash-dspark` · max_model_len 1,048,576

## C1 pure decode (256 tok, temp 0, chat)

| run | pure tok/s | wall tok/s | out_tok |
|---:|---:|---:|---:|
| 1 | 44.42 | 42.46 | 253 |
| 2 | 32.67 | 31.74 | 256 |
| 3 | 48.83 | 46.89 | 256 |
| 4 | 56.60 | 54.52 | 256 |
| 5 | 35.69 | 34.37 | 256 |

**mean / peak pure:** 43.64 / 56.60

**mean / peak wall:** 42.00 / 54.52

## C4 aggregate: **64.5 tok/s** (512 tok / 7.9s)

## Math eval

| q | expect | got | ok |
|---|---|---|---|
| `12*11` | 132 | 132 | ✓ |
| `100-37` | 63 | 63 | ✓ |
| `847*293` | 248171 | 248171 | ✓ |
| `15+27` | 42 | 42 | ✓ |
| `2**10` | 1024 | 1024 | ✓ |

## Tools: ✓ finish_reason=`tool_calls`

## Code smoke: ✓

```
```python
s[::-1]
```
```
