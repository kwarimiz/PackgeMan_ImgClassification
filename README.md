
# **做个调包侠，把图像分类流程化繁为简！**

本仓库为你提供了一个全面的图像分类工具包，你只需要准备好数据，即可通过一行脚本开启训练。无论是模型、优化器还是损失函数的选择，你都可以在不频繁修改代码的情况下，轻松切换配置。从简单的 CNN 到复杂的 Transformer，这里应有尽有。

你可以方便地查看被错误分类的图像，比较它们的正确标签和错误标签，从而进行分析和改进。所有的实验过程和结果都将记录在网页上，你可以清晰地对比分析。

我们提供了单卡和多卡的训练和推理方式，高效完成任务。所有 PyTorch 代码都已封装好，你只需要成为一个无情的调包侠！

## **安装**

使用以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## **数据准备**

按照 PyTorch DataLoader 的方式组织数据：

```
--data root
    --train
        --class1
        --class2
        --class3
        ...
    --val
        --class1
        --class2
        --class3
        ...
    --test
        --class1
        --class2
        --class3
        ...
```

在 `config/root_path` 中，定义你的数据集根目录 `DATA_ROOT` 和权重存放目录 `WEIGHT_ROOT`。

## **训练**

- **单卡训练：**
  ```bash
  python train_wandb.py -n [net] -r [result_folder]
  ```

- **多卡训练：**
  ```bash
  accelerate launch --num_processes=[gpu numbers] train_wandb.py -n [net] -r [result_folder]
  ```

### **调参**

想要更换模型？只需修改 `-n` 参数，将 `net` 替换为所需模型名称。支持的模型请查看 `models/model.py`。在 `train_components` 中可以查看支持的损失函数、优化器和调度器。

更多参数定义可在 `config/cmd_args.py` 中查看。

## **评估**

在 `predict.ipynb` 中调用 `evaluator`，可以查看混淆矩阵等各类指标。

## **推理**

- **单卡推理：**
  在 `predict.ipynb` 中设置 `evaluator` 的 `gpu_num=1`。

- **多卡推理：**
  ```bash
  accelerate launch --num_processes=4 predict_faster.py -r [result_folder] --result_root [result_root]
  ```

## **日志记录**

登录你的 Weights & Biases (W&B) 账号，并添加 API 密钥，即可自动记录和可视化实验过程。

---



---

# **Be a Package Man , Simplify Your Image Classification Workflow!**

This repository provides everything you need for image classification. Just prepare your data, and you can start training with a single script. You can switch models, optimizers, loss functions, and more by making minimal changes to the script without frequently modifying the code. From simple CNNs to complex Transformers, it’s all here.

Easily view misclassified images, compare their correct and incorrect labels, and make informed analyses to improve results. All your experiments and results are logged on a web interface, allowing for clear comparisons.

We offer both single-GPU and multi-GPU training and inference options, ensuring efficient execution. All PyTorch code is pre-packaged, so you just need to be a relentless package tinkerer!

## **Installation**

Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## **Data Preparation**

Organize your data according to the PyTorch DataLoader format:

```
--data root
    --train
        --class1
        --class2
        --class3
        ...
    --val
        --class1
        --class2
        --class3
        ...
    --test
        --class1
        --class2
        --class3
        ...
```

Define your dataset root directory `DATA_ROOT` and weights storage directory `WEIGHT_ROOT` in `config/root_path`.

## **Training**

- **Single-GPU Training:**
  ```bash
  python train_wandb.py -n [net] -r [result_folder]
  ```

- **Multi-GPU Training:**
  ```bash
  accelerate launch --num_processes=[gpu numbers] train_wandb.py -n [net] -r [result_folder]
  ```

### **Hyperparameter Tuning**

Want to change the model? Simply modify the `-n` parameter and replace `net` with your desired model name. Supported models can be found in `models/model.py`. Check out `train_components` for available loss functions, optimizers, and schedulers.

You can find more parameter definitions in `config/cmd_args.py`.

## **Evaluation**

Use `evaluator` in `predict.ipynb` to view confusion matrices and various metrics.

## **Inference**

- **Single-GPU Inference:**
  In `predict.ipynb`, set `evaluator`'s `gpu_num=1`.

- **Multi-GPU Inference:**
  ```bash
  accelerate launch --num_processes=4 predict_faster.py -r [result_folder] --result_root [result_root]
  ```

## **Logging**

Log in to your Weights & Biases (W&B) account and add your API key to automatically track and visualize your experiments.

---

With these simple steps, you’ll become a package tinkerer, effortlessly navigating the world of image classification!