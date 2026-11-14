# VirusHost3Class：病毒宿主三分类模型

[English](README.md) · [安装模型](docs/INSTALL.md) · [方法细节](docs/METHODS.md) · [复现说明](docs/REPRODUCIBILITY.md)

VirusHost3Class 根据病毒核酸序列，将病毒划分为 **Insect（昆虫）**、**Plant（植物）** 和 **Other（其他宿主）** 三类。程序从三份全长 FASTA 开始，生成四个长度组，重新提取序列特征，完成五折交叉验证，并训练可用于新病毒序列预测的模型。

- 长度组：1000 bp、3000 bp、5000 bp、全长。
- 特征：1-mer、2-mer、3-mer、4-mer、ESM-2（320维）、LucaVirus（2560维）。
- 分类器：随机森林、KNN、SVM、GaussianNB。
- 总计：**4 × 6 × 4 = 96组**独立模型评估。

ESM-2 和 LucaVirus 的 embedding **直接从输入 FASTA 重新推理**，不导入历史特征。

## 1. 安装

推荐 Linux + Conda。LucaVirus 需要官方 [LucaVirusTasks](https://github.com/LucaOne/LucaVirusTasks) 源码及预训练权重。

```bash
conda env create -f environment.yml
conda activate virushost3class
cp config.example.env config.env
bash scripts/setup_models.sh
```

模型权重获取和环境校验见 [安装说明](docs/INSTALL.md)。

## 2. 先运行示例

```bash
python scripts/example_run.py
```

示例包含可运行的**人工核酸序列**，使用 3-mer + GaussianNB 演示切片、特征提取、五折训练、模型保存和新序列预测，不需要下载大型模型。输出保存在 `work/example/`，**不代表论文结果**。

## 3. 完整研究流程

将已选好代表序列的三份全长 FASTA 放到：

```text
input/full_length/Insect_full_length.fasta
input/full_length/Plant_full_length.fasta
input/full_length/Other_full_length.fasta
```

Other 也支持 `.fasta.gz`。FASTA 每条序列标题以唯一的数字病毒 TaxID 开头，例如 `>123456|virus=example`。分节段病毒需在输入前按研究选定的片段顺序拼接。详细要求见 [DATA](docs/DATA.md)。

编辑 `config.env` 中的输入路径和切片参数。确认模型源码、权重能够访问后执行：

```bash
bash run_all.sh --check
mkdir -p logs
nohup bash run_all.sh --run > logs/reproduce_all.log 2>&1 &
tail -f logs/reproduce_all.log
```

运行流程：全长 FASTA → 四个长度组 → 六种特征重新提取 → 96组五折训练 → pooled OOF 统一评估 → 保存4个长度的 LucaVirus+SVM 模型。

中断后直接重复 `bash run_all.sh --run`，程序会核验并跳过已完成步骤，从未完成的 embedding 批次或模型组合继续。若原始输入或配置改变，程序会拒绝混用旧结果。

## 4. 预测新病毒

```bash
python predict.py \
  --input queries.fasta \
  --model-dir work/models/full_length/LucaVirus_SVM \
  --length full_length \
  --lucavirus-tasks external/LucaVirusTasks \
  --output predictions.tsv
```

输出三类的预测概率以及预测标签。固定长度模型会优先读取模型保存的切片种子；若未保存，需手动传入一致的 `--slice-seed`。不同长度之间不自动选择模型，避免引入未经验证的长度阈值。

## 5. 输出

| 位置 | 内容 |
|---|---|
| `work/sequences/` | 四个长度的FASTA和TaxID表 |
| `work/features/` | 各类新生成的特征矩阵和模型参数记录 |
| `work/results_parts/` | 逐特征、逐模型的可恢复五折结果 |
| `work/final/` | 96组汇总表、混淆矩阵、Figure 5 数值核对 |
| `work/models/` | 独立拟合的LucaVirus+SVM预测模型 |

本仓库不随代码分发原始全长数据、第三方模型权重或已训练好的分类器。论文数值的严格复现还需匹配原始切片及checkpoint；见 [复现说明](docs/REPRODUCIBILITY.md)。

## License

本项目源码采用 [MIT License](LICENSE)；ESM-2、LucaVirus及其权重受各自项目许可证约束。
