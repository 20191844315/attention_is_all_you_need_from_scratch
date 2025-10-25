# train_test.py
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from time import perf_counter, ctime
from datetime import timedelta
from tqdm import tqdm  # 进度条

from model import Transformer
from dataset import get_dataset, sos, eos, cmn_words, eng_words, seq_len, pad

# ---------------------
# 设备选择
# ---------------------
torch.manual_seed(9527)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# 数据加载
# ---------------------
train_loader, test_loader = get_dataset()
input_size = len(cmn_words) + eos + 1
output_size = len(eng_words) + eos + 1

# ---------------------
# 模型、优化器、损失
# ---------------------
model = Transformer(input_size, output_size, max_len=seq_len, padding_idx=pad).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)


def backward(logits, target):
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


# ---------------------
# 训练函数
# ---------------------
def train(model, dataloader, backward, epochs=20, save_path="best_model.pt"):
    losses = []
    best_loss = float('inf')
    model.train()
    label, start_time = model.__class__.__name__, perf_counter()
    print(f"{label} @ {ctime()}\n")

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for source, target in progress_bar:
            source, target = source.to(device), target.to(device)

            logits = model(source, target[:, :-1])
            loss = backward(logits.view(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss_val = loss.cpu().item()

            losses.append(loss_val)
            total_loss += loss_val
            progress_bar.set_postfix(loss=total_loss / len(dataloader))

        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Time {timedelta(seconds=int(perf_counter() - start_time))}, "
            f"Loss {avg_loss:.4f}"
        )

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with loss {best_loss:.4f}")

    print(f"\n{label} total time: {timedelta(seconds=int(perf_counter() - start_time))}\n")
    return losses


# ---------------------
# 测试函数
# ---------------------
@torch.no_grad()
def eval(model, dataloader, print_result, max_len):
    model.eval()
    for x, Y in dataloader:
        x = x.to(device)
        y = torch.full((x.size(0), 1), sos, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = model(x, y)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            y = torch.cat([y, next_token], dim=1)
            if (next_token == eos).all():
                break

        print_result(
            [i - eos - 1 for i in x.view(-1).tolist() if i > eos],
            [i - eos - 1 for i in Y.view(-1).tolist() if i > eos],
            [i - eos - 1 for i in y.view(-1).tolist() if i > eos]
        )


# ---------------------
# 打印函数
# ---------------------
def print_result(source, target, output):
    print("source:", " ".join(cmn_words[i] for i in source))
    print("target:", " ".join(eng_words[i] for i in target))
    print("output:", " ".join(eng_words[i] for i in output))
    print()


# ---------------------
# 训练
# ---------------------
losses = train(model, train_loader, backward, epochs=20, save_path="best_model.pt")

# ---------------------
# 绘制 loss 曲线
# ---------------------
plt.figure(figsize=(6, 4))
plt.plot(losses, label="loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# ---------------------
# 加载最佳模型进行测试
# ---------------------
model.load_state_dict(torch.load("best_model.pt"))
eval(model, test_loader, print_result, seq_len)
