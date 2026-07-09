#!/usr/bin/env python3
"""Reproduce C1 / C4 throughput + math / tools / code smoke against a live
DeepSeek-V4-Flash-DSpark OpenAI-compatible endpoint.

Example (rank0 API on :8000):

  python3 benchmarks/bench_eval.py --api http://10.100.10.3:8000/v1 \\
      --out benchmarks/results/my-run.json

Methodology matches RESULTS-nvfp4-1m.md:
  pure tok/s = (completion_tokens - 1) / (t_end - t_first_content)
  wall tok/s = completion_tokens / (t_end - t_request_start)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path


def stream_chat(api: str, messages, max_tokens=256, temperature=0.0, model="deepseek-v4-flash-dspark"):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    text = []
    finish = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            d = choices[0].get("delta") or {}
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
            c = d.get("content")
            if c:
                if t_first is None:
                    t_first = time.perf_counter()
                text.append(c)
    t_end = time.perf_counter()
    full = "".join(text)
    out_tok = (usage or {}).get("completion_tokens") or max(1, len(full.split()))
    pure = wall = None
    if t_first is not None and out_tok > 1:
        pure = (out_tok - 1) / (t_end - t_first)
        wall = out_tok / (t_end - t0)
    return {
        "out_tokens": out_tok,
        "pure_tok_s": pure,
        "wall_tok_s": wall,
        "ttft_s": (t_first - t0) if t_first else None,
        "elapsed_s": t_end - t0,
        "finish_reason": finish,
        "text": full[:2000],
        "usage": usage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--out", default="benchmarks/results/bench_eval.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    results = {
        "meta": {
            "api": args.api,
            "model": args.model,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stack": "1M nvfp4_ds_mla stage-c B12X DSpark k=5 TP=2",
        },
        "c1": [],
        "evals": {},
    }

    print("warm...", flush=True)
    try:
        stream_chat(args.api, [{"role": "user", "content": "Say hi in one word."}], max_tokens=8, model=args.model)
    except Exception as e:
        print("warm failed:", e)

    prompts = [
        "Write a Python function that implements binary search on a sorted list. Include docstring and edge cases. No markdown fences.",
        "Explain how NCCL all-reduce works over RoCE in two short paragraphs.",
        "Write a bash script that rsyncs a directory with retries and logging.",
        "Implement a minimal LRU cache class in Python with get/put O(1).",
        "Describe the tradeoffs of nvfp4 vs fp8 KV cache for long-context MLA decode.",
    ]
    print("C1 x5...", flush=True)
    for i, p in enumerate(prompts, 1):
        r = stream_chat(args.api, [{"role": "user", "content": p}], max_tokens=args.max_tokens, model=args.model)
        row = {"run": i, "prompt": p[:80], **{k: r[k] for k in r if k != "text"}}
        row["text_preview"] = (r["text"] or "")[:300]
        results["c1"].append(row)
        print(f"  run{i}: pure={r['pure_tok_s']:.2f} wall={r['wall_tok_s']:.2f} tok={r['out_tokens']}", flush=True)

    pures = [x["pure_tok_s"] for x in results["c1"] if x.get("pure_tok_s")]
    walls = [x["wall_tok_s"] for x in results["c1"] if x.get("wall_tok_s")]
    results["c1_summary"] = {
        "n": len(pures),
        "pure_mean": statistics.mean(pures) if pures else None,
        "pure_peak": max(pures) if pures else None,
        "wall_mean": statistics.mean(walls) if walls else None,
        "wall_peak": max(walls) if walls else None,
    }
    print("C1 summary", results["c1_summary"], flush=True)

    print("C4...", flush=True)

    def one(i):
        return stream_chat(
            args.api,
            [{"role": "user", "content": f"[{i}] Write 3 bullet points about CUDA graphs vs eager."}],
            max_tokens=128,
            model=args.model,
        )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        c4 = list(ex.map(one, range(4)))
    t_wall = time.perf_counter() - t0
    agg_tok = sum(x["out_tokens"] for x in c4)
    results["c4"] = {
        "streams": [{k: x[k] for k in ("out_tokens", "pure_tok_s", "wall_tok_s", "ttft_s", "elapsed_s")} for x in c4],
        "aggregate_tok_s": agg_tok / t_wall if t_wall else None,
        "wall_s": t_wall,
        "total_tokens": agg_tok,
    }
    print("C4 agg", results["c4"]["aggregate_tok_s"], flush=True)

    print("math...", flush=True)
    math_cases = [("12*11", "132"), ("100-37", "63"), ("847*293", "248171"), ("15+27", "42"), ("2**10", "1024")]
    math_out = []
    for q, expect in math_cases:
        r = stream_chat(
            args.api,
            [{"role": "user", "content": f"Compute exactly: {q}. Reply with only the integer result, no other text."}],
            max_tokens=32,
            model=args.model,
        )
        m = re.search(r"-?\d+", r["text"] or "")
        got = m.group(0) if m else (r["text"] or "").strip().split()[0:1]
        got = got if isinstance(got, str) else (got[0] if got else "")
        ok = got == expect
        math_out.append({"q": q, "expect": expect, "got": got, "ok": ok, "text": (r["text"] or "")[:200]})
        print(f"  {q}: got={got} expect={expect} ok={ok}", flush=True)
    results["evals"]["math"] = math_out

    print("tools...", flush=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 128,
        "temperature": 0.0,
    }
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            args.api.rstrip("/") + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            tr = json.loads(r.read())
        ch = tr["choices"][0]
        msg = ch.get("message") or {}
        results["evals"]["tools"] = {
            "finish_reason": ch.get("finish_reason"),
            "has_tool_calls": bool(msg.get("tool_calls")),
            "tool_calls": msg.get("tool_calls"),
            "content_preview": (msg.get("content") or "")[:200],
            "ok": ch.get("finish_reason") == "tool_calls" or bool(msg.get("tool_calls")),
        }
        print("tools", results["evals"]["tools"]["finish_reason"], results["evals"]["tools"]["ok"], flush=True)
    except Exception as e:
        results["evals"]["tools"] = {"ok": False, "error": str(e)}
        print("tools fail", e, flush=True)

    print("code smoke...", flush=True)
    r = stream_chat(
        args.api,
        [{"role": "user", "content": "Write only a Python one-liner that reverses a string s."}],
        max_tokens=64,
        model=args.model,
    )
    results["evals"]["code_smoke"] = {
        "text": (r["text"] or "")[:300],
        "ok": "[::-1]" in (r["text"] or ""),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
