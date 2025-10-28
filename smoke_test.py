import torch
from time import perf_counter

# Minimal smoke test to verify the environment on a headless server.
# - Builds a model
# - Loads one batch from the dataset
# - Runs one forward/backward step on CPU
# - Prints a short summary and exits

from model import Transformer
from dataset import get_dataset, cmn_words, eng_words, seq_len, pad

def main():
    torch.manual_seed(9527)
    device = torch.device("cpu")  # Always use CPU for the smoke test

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    train_loader, _ = get_dataset(batch_size=8)
    input_size = len(cmn_words) + 3
    output_size = len(eng_words) + 3

    model = Transformer(input_size, output_size, max_len=seq_len, padding_idx=pad).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)

    model.train()
    start = perf_counter()

    # Take a single batch
    source, target = next(iter(train_loader))
    source, target = source.to(device), target.to(device)

    logits = model(source, target[:, :-1])
    loss = loss_fn(logits.view(-1, logits.size(-1)), target[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    dur = perf_counter() - start

    print("Smoke test OK")
    print({
        "batch": tuple(source.shape),
        "seq_len": seq_len,
        "loss": float(loss.detach().cpu()),
        "time_sec": round(dur, 3),
    })


if __name__ == "__main__":
    main()
