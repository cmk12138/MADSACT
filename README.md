# MADSACT 多无人机协同路径规划运行说明书

## 1. 项目概述

本项目实现的是面向多无人机协同路径规划的 MADSACT 算法（Multi-Agent DSAC-T / MADSACT），核心训练入口在：

- marl_dsact_path_env_3_smooth.py
- run_multiseed_ci_dsact.py
- path_env.py

该程序采用多智能体强化学习框架，其中：

- 领导者智能体和跟随者智能体共享/分离策略网络
- 采用连续动作空间，动作通过 tanh 裁剪并进行 SAC 风格的策略更新
- 使用分布式 Critic（DSAC-T 风格）进行价值估计和 TD bound 约束
- 输出训练奖励曲线、检查点以及测试评估指标

适用场景包括：

- 多无人机协同路径规划
- 形成保持与协同避障
- 目标到达与任务完成率评估

---

## 2. 目录结构说明

在当前工作区中，核心代码文件如下：

- marl_dsact_path_env_3_smooth.py：主训练程序，包含环境构建、网络结构、ReplayBuffer、训练循环和奖励可视化
- run_multiseed_ci_dsact.py：多随机种子训练与评估脚本，用于分析平均结果和 95% 置信区间
- test_dsact_marl_mainplot.py：测试脚本，可加载保存的模型并进行评估
- path_env.py：无人机路径规划环境
- checkpoints_path_dsact：训练中间模型保存目录
- runs：训练日志和结果输出目录
- multiseed_runs：多种子实验结果目录

如果需要查看奖励图，可以在训练完成后查看：

- checkpoints_path_dsact
- runs

---

## 3. 运行环境要求

建议使用如下环境：

- Python 3.9 及以上
- PyTorch 1.10 及以上
- NumPy
- Matplotlib
- Gym / 依赖环境中的仿真模块

安装依赖命令示例：

```bash
pip install numpy matplotlib torch
```

如果项目中需要额外依赖，请在运行前检查是否已经安装好环境中所需的库；如果报错为缺失模块，先安装对应依赖，再继续训练。

> 注意：该项目的训练入口已经不是旧版 MASAC 的 main_SAC.py，而是 DSAC-T 型多智能体训练脚本。请优先使用 MADSACT 相关脚本进行训练和测试。

---

## 4. 训练步骤

### 4.1 直接训练 MADSACT

进入项目根目录后，执行：

```bash
python marl_dsact_path_env_3_smooth.py
```

该脚本会：

1. 创建环境
2. 初始化多个智能体策略网络和 Critic 网络
3. 进行强化学习训练
4. 每隔一段回合保存模型
5. 训练完成后生成奖励曲线图

训练输出默认保存在：

- ./runs/r1/
- ./checkpoints_path_dsact/

模型文件示例：

- dsact_path_ep50000.pt
- reward_curves.png
- reward_total_only.png

---

### 4.2 通过配置修改训练参数

主训练脚本中的 Config 类中定义了关键参数，位置在 marl_dsact_path_env_3_smooth.py。关键参数如下：

- N_AGENT：领导者数量，通常为 1
- M_ENEMY：跟随者数量，可根据任务调整
- RENDER：是否显示可视化界面
- EP_MAX：最大训练回合数
- EP_LEN：每回合最大步数
- GAMMA：折扣因子
- ACTOR_LR / CRITIC_LR / ALPHA_LR：学习率
- BATCH：训练批次大小
- REPLAY_SIZE：经验回放容量
- SAVE_DIR：模型保存路径
- SAVE_EVERY_EP：每多少回合保存一次检查点

推荐先用较小的回合数快速验证，如：

```python
EP_MAX = 200
SAVE_EVERY_EP = 50
```

验证通过后再扩大训练规模。

---

### 4.3 多随机种子实验

如果需要统计多次实验的平均结果和置信区间，可使用：

```bash
python run_multiseed_ci_dsact.py
```

也可以指定种子和跟随者数量，例如：

```bash
python run_multiseed_ci_dsact.py --seeds 0,1,2,3,4 --followers 2 --ep-max 5000 --test-episodes 100
```

输出目录：

- ./multiseed_runs/
- ./multiseed_runs/summary_across_seeds.csv

该脚本会自动完成：

- 多种子重复训练
- 模型评估
- 统计 MCR、FKR、JT、JS、JC 等指标
- 计算均值和 95% 置信区间

---

## 5. 模型测试步骤

训练完成后，可使用测试脚本进行评估：

```bash
python test_dsact_marl_mainplot.py --ckpt ./checkpoints_path_dsact/dsact_path_ep50000.pt --episodes 100 --render
```

参数说明：

- --ckpt：指定模型路径
- --episodes：测试回合数
- --render：是否打开可视化界面

如果不指定 --ckpt，则脚本会自动在保存目录中搜索最新的模型。

测试时，程序会：

- 加载训练好的 actor
- 在环境中执行多个测试回合
- 统计奖励、任务完成情况、编队保持率等指标
- 可选进行可视化渲染

---

## 6. 关键代码说明

### 6.1 训练主入口

训练主函数在 marl_dsact_path_env_3_smooth.py 中定义：

```python
def main():
    cfg = Config()
    train(cfg)
```

真正的训练逻辑在：

```python
def train(cfg: Config):
```

它负责：

- 创建环境
- 初始化智能体
- 采样动作
- 进行策略更新
- 保存权重
- 生成奖励曲线

### 6.2 环境接口

环境在 path_env.py 和 rl_env/path_env.py 中实现，检测步骤返回值时，脚本使用：

```python
def unwrap_step(step_out):
```

这个函数兼容不同版本的环境返回格式，保证训练代码能稳定运行。

---

## 7. 常见问题与处理建议

### 7.1 程序报错：找不到模块

如果出现类似：

- ModuleNotFoundError
- No module named 'rl_env'
- No module named 'path_env'

请先检查当前工作目录是否正确，并确认脚本与环境文件处于同一目录层级，必要时使用以下方式：

```bash
python -c "import os; print(os.getcwd())"
```

并确认你运行的是项目根目录下的脚本。

### 7.2 训练很慢

建议：

- 先设置更短的 EP_MAX
- 把 RENDER 设置为 False
- 使用更小的 BATCH 或更少的测试回合

### 7.3 训练后没有图片或模型

通常是因为：

- 训练未跑完整
- SAVE_DIR 路径不正确
- 训练过程中出现异常退出

请检查终端输出和保存目录，确认是否已生成 checkpoint。

---

## 8. 推荐运行顺序

为了保证训练流程稳定，建议按以下顺序执行：

1. 安装依赖
2. 运行：
   ```bash
   python marl_dsact_path_env_3_smooth.py
   ```
3. 等待训练保存模型和奖励曲线
4. 若需要评估，执行：
   ```bash
   python test_dsact_marl_mainplot.py --ckpt ./checkpoints_path_dsact/dsact_path_ep50000.pt --episodes 50
   ```
5. 若需要统计多种子结果，执行：
   ```bash
   python run_multiseed_ci_dsact.py
   ```

---

## 9. 总结

本项目的核心运行入口是 MADSACT 多智能体路径规划训练流程。最关键的训练脚本是：

- marl_dsact_path_env_3_smooth.py

如果你想复现实验效果，建议先用较短训练回合验证环境和参数，再逐步扩大到正式训练规模。

在训练完成后，使用测试脚本加载 checkpoint 即可进行算法效果验证和结果分析。
