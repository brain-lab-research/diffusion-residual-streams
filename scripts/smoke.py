"""Small CPU forward/backward check of the released model implementations."""
import os

os.environ.update(MODEL_VOCAB="128", MODEL_MASK_ID="127", MODEL_SEQ="16", MODEL_DIM="32",
                  MODEL_LAYERS="2", MODEL_HEADS="4", MODEL_KV_HEADS="2",
                  MODEL_INTERMEDIATE="64", MHC_STREAMS="2", MHC_PAPER="1",
                  MHC_PROPER_INIT="1", MHC_RESMIX="learned")

import torch  # noqa: E402
from model_and_train import Model, mdlm_loss  # noqa: E402
from mechanism_audit_n2.engine import Engine  # noqa: E402


def main():
    torch.manual_seed(7)
    ids = torch.randint(0, 127, (2, 16))
    mask = torch.rand(2, 16) < 0.5
    corrupted = ids.masked_fill(mask, 127)
    t = torch.full((2,), 0.5)
    for arm in ("baseline", "mhc", "frr"):
        model = Model(arm)
        logits = model(corrupted)
        assert logits.shape == (2, 16, 128)
        loss = mdlm_loss(logits, ids, mask, t)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.emb.weight.grad).all()
        if arm != "frr":
            engine = Engine(model.eval(), arm, 2 if arm == "mhc" else 1, 0)
            state, _ = engine.run(corrupted, mask)
            torch.testing.assert_close(engine.original_logits(state), logits.detach(), rtol=0, atol=0)
        print(f"{arm}: forward, backward, instrumented logits OK")


if __name__ == "__main__":
    main()
