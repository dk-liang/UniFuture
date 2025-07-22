from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "unifuture.modules.GeneralConditioner",
    "params": {"emb_models": list()}
}
