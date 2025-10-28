##  Attention is all you need
Simple Transformer Implemented by Pytorch.
基于谷歌原论文的transformer实现，在舍弃了数据预处理和工程优化等内容，旨在帮助初学者快速复现并理解transformer和pytorch的内容架构，同时聚焦于自己复现时遇到的各种问题，尽量详细的做到适合中国宝宝体质的无障碍理解。

PS:感谢[ngolin](https://github.com/ngolin)提供了框架和数据集，本项目使用的数据来自于[AttentionIsAllYouNeed](https://github.com/ngolin/AttentionIsAllYouNeed)

## 快速开始
配置环境
```bash
    pip install -r requirements.txt
```
快速开始
训练模型
```bash
    $ python main.py
```
利用训练好的 best_model.pt进行测试
```bash
    $ python test.py
```

## 在服务器上运行（CPU/GPU）
本仓库已适配无显示的服务器环境：
- 已将 Matplotlib 切换为非交互后端（Agg），训练/测试会将损失曲线保存为图片文件（training_loss.png / training_loss_test.png）。
- 加载模型使用 map_location，避免 GPU/CPU 不匹配导致的加载失败。

1) 基础依赖
- Python 3.9+（建议与你本地一致）
- 安装依赖
```bash
pip install -r requirements.txt
```

2) PyTorch 安装说明
- CPU 服务器（最简单，适合无 GPU）
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- GPU 服务器（根据服务器 CUDA 版本选择）
参考 https://pytorch.org/get-started/locally/ 选择匹配 CUDA 的安装命令。例如（以 CUDA 12.x 为例）：
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
注意：requirements.txt 中的版本是参考，若与你的服务器 CUDA 不匹配，请以官网命令为准安装可用版本。

3) 数据与目录
- 数据文件已包含在 `./dataset/cmn-eng.txt`。
- 训练过程中会在项目根目录保存 `best_model.pt` 与损失曲线图片。

4) 先做一次环境冒烟测试（强烈推荐）
```bash
python smoke_test.py
```
输出包含 batch 形状、一次前后向的损失和用时，看到“Smoke test OK”即可说明环境可跑。

5) 启动训练/测试
- 训练
```bash
python main.py
```
输出日志中会打印每个 epoch 的平均 loss，并在 loss 改善时更新 `best_model.pt`，训练结束会生成 `training_loss.png`。
- 使用最佳模型测试
```bash
python test.py
```


6) 后台运行（可选）
- Linux 常用：tmux/screen/nohup。例如：
```bash
nohup python main.py > train.out 2>&1 &
```
- 或在 tmux 内启动：
```bash
tmux new -s tf
python main.py
```

7) 常见问题
- 找不到 CUDA 或驱动不匹配：请用 CPU 版本的 torch 先跑通，或按 PyTorch 官网选择正确 CUDA 版本安装。
- 服务器无显示导致报错：本项目已将绘图改为保存到文件，不需要显示。
- best_model.pt 在 CPU/GPU 间加载错误：我们已使用 map_location 自动适配当前设备。

## 文件说明
main.py：训练代码
model.py：模型结构与具体实现
best_model.pt: 训练过程中得到的最好的模型
test.py 实验模型训练效果（中译英）
## 实验结果
![alt text](image.png)
## 参考资料
[论文原文](https://arxiv.org/abs/1706.03762)
[论文精讲](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
[AttentionIsAllYouNeed](https://github.com/ngolin/AttentionIsAllYouNeed)
