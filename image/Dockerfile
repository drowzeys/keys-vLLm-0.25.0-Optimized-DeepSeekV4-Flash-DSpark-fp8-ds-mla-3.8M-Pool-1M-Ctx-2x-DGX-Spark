FROM vllm/vllm-openai:nightly-aarch64
# vLLM nightly DSV4-Flash sm120 dual-cache sparse-MLA needs flashinfer 0.6.14 API
# (swa_topk_lens/extra_sparse_indices); image ships 0.6.13.
RUN pip install --no-cache-dir       flashinfer-python==0.6.14 flashinfer-cubin==0.6.14 flashinfer-jit-cache==0.6.14+cu130       --find-links https://flashinfer.ai/whl/flashinfer-cubin/       --find-links https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/
