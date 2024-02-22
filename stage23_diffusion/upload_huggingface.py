from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/cpfs01/user/hejingwen/AdaptBIR/stage2.ckpt",
    path_in_repo="stage2.ckpt",
    repo_id="yqliu/AdaptBIR",
    token="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",
    repo_type="model",
)