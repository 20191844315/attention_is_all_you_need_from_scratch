# train_test.py
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for servers/headless envs
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
#数据集大小，为开始符，结束符，占位符留3个位置
input_size = len(cmn_words) + 3
output_size = len(eng_words) + 3

# ---------------------
# 模型、优化器、损失
# ---------------------
model = Transformer(input_size, output_size, max_len=seq_len, padding_idx=pad).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)


def backward(logits, target):
    #logits 是神经网络模型最后一层的原始输出，在进入 Softmax 之前的值
    loss = loss_fn(logits, target)#计算损失函数
    loss.backward()#反向传播
    optimizer.step()#根据梯度更新参数
    optimizer.zero_grad()#清空梯度
    return loss


# ---------------------
# 训练函数
# ---------------------
def train(model, dataloader, backward, epochs=20, save_path="best_model.pt"):
    # 初始化
    losses = []
    best_loss = float('inf')
    model.train()

    label, start_time = model.__class__.__name__, perf_counter()
    print(f"{label} @ {ctime()}\n")

    for epoch in range(epochs):
        #总损失
        total_loss = 0
        #显示进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for source, target in progress_bar:
            source, target = source.to(device), target.to(device)
            # 前向传播，decoder错位输入，少输入一位eos，保证模型一直预测到eos
            logits = model(source, target[:, :-1])
            # 计算损失并反向传播
            loss = backward(logits.view(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            # 获取，记录损失数值
            loss_val = loss.cpu().item()
            losses.append(loss_val)
            total_loss += loss_val
            # 更新进度条显示
            progress_bar.set_postfix(loss=total_loss / len(dataloader))
        #计算平均损失
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
# 告诉 PyTorch 不要构建梯度计算图，节省内存、提高速度
@torch.no_grad()
def eval(model, dataloader, print_result, max_len):
    model.eval()#评估模式
    for x, Y in dataloader: # x: 源语言, Y: 真实目标语言（用于对比）
        x = x.to(device)
        #创建一个形状为 [batch_size, 1] 的张量，全部用 sos（start-of-sequence） token填充
        y = torch.full((x.size(0), 1), sos, dtype=torch.long, device=device)

        for _ in range(max_len - 1):# 最多生成max_len-1个token（因为已有<sos>）
            logits = model(x, y)#源序列 x + 当前已生成的部分目标序列 y
            #取出当前模型认为 “最有可能成为下一个词” 的 token 。
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            #拼接
            y = torch.cat([y, next_token], dim=1)
            if (next_token == eos).all():#如果当前batch中的所有序列都生成了 <eos>，就提前结束
                break

        print_result(
            #i > eos过滤掉012三个特殊索引
            [i - 3 for i in x.view(-1).tolist() if i > 2],
            [i - 3 for i in Y.view(-1).tolist() if i > 2],
            [i - 3 for i in y.view(-1).tolist() if i > 2]
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
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
print("Saved training loss plot to training_loss.png")

# ---------------------
# 加载最佳模型进行测试
# ---------------------
model.load_state_dict(torch.load("best_model.pt", map_location=device))
eval(model, test_loader, print_result, seq_len)
