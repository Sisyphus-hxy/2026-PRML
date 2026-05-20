# ai辅助了部分文件/绘图代码
import math
import random
import argparse
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ReverseDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=20):
        self.num_samples=num_samples
        self.seq_len=seq_len
        self.vocab_size=vocab_size

        self.data=[]
        for _ in range(num_samples):
            x=torch.randint(1,vocab_size,(seq_len,))
            y=torch.flip(x,dims=[0])
            self.data.append((x,y))
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx]
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe=torch.zeros(max_len, d_model)
        
        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        pos_emb = self.pos_embedding(positions)

        return x + pos_emb

class FixedAbsolutePositionalEncoding(nn.Module):
    """
    一个简单的绝对位置编码，用来和 sin/cos 编码做对照。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        if max_len > 1:
            position = position / (max_len - 1)
        basis = torch.linspace(0.5, 2.0, d_model).unsqueeze(0)
        pe = position * basis
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class NoPositionalEncoding(nn.Module):
    def forward(self, x):
        return x

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_head = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        output = self.W_o(attn_output)

        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=64, d_ff=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attn_weights

class TransformerReverseModel(nn.Module):
    def __init__(
        self,
        vocab_size=20,
        seq_len=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        pos_encoding="sinusoidal",
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        self.embedding_scale = math.sqrt(d_model)

        if pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        elif pos_encoding == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=seq_len)
        elif pos_encoding == "fixed_absolute":
            self.pos_encoding = FixedAbsolutePositionalEncoding(d_model, max_len=seq_len)
        elif pos_encoding == "none":
            self.pos_encoding = NoPositionalEncoding()
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) * self.embedding_scale
        x = self.dropout(self.pos_encoding(x))

        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x)

        logits = self.classifier(x)
        return logits, last_attn

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            pred = logits.argmax(dim=-1)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            correct_tokens += (pred == y).sum().item()
            correct_sequences += (pred == y).all(dim=1).sum().item()
            total_sequences += y.size(0)

    return {
        "loss": total_loss / total_tokens,
        "token_acc": correct_tokens / total_tokens,
        "seq_acc": correct_sequences / total_sequences,
    }

def train_one_experiment(args, pos_encoding, device, train_dataset, val_dataset):
    set_seed(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = TransformerReverseModel(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pos_encoding=pos_encoding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        train_loss = total_loss / total_tokens
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "pos_encoding": pos_encoding,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_token_acc": val_metrics["token_acc"],
            "val_seq_acc": val_metrics["seq_acc"],
        }
        history.append(row)

        print(
            f"[{pos_encoding:14s}] epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"token_acc={val_metrics['token_acc']:.4f} seq_acc={val_metrics['seq_acc']:.4f}"
        )

    sample_x, sample_y = next(iter(val_loader))
    sample_x = sample_x[:5].to(device)
    sample_y = sample_y[:5].to(device)
    model.eval()
    with torch.no_grad():
        sample_pred = model(sample_x)[0].argmax(dim=-1)

    return history, sample_x.cpu(), sample_y.cpu(), sample_pred.cpu()

def save_history(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "position_encoding_results.csv"
    fieldnames = [
        "pos_encoding",
        "epoch",
        "train_loss",
        "val_loss",
        "val_token_acc",
        "val_seq_acc",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return csv_path

def save_plot(history, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图，只保存 CSV。")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "position_encoding_comparison.png"
    groups = {}
    for row in history:
        groups.setdefault(row["pos_encoding"], []).append(row)

    plt.figure(figsize=(8, 5))
    for name, rows in groups.items():
        rows = sorted(rows, key=lambda item: item["epoch"])
        epochs = [row["epoch"] for row in rows]
        acc = [row["val_token_acc"] for row in rows]
        plt.plot(epochs, acc, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation token accuracy")
    plt.title("Position Encoding Comparison on Reverse Sequence Task")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reverse sequence experiment with several positional encodings."
    )
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-encodings",
        nargs="+",
        default=["sinusoidal", "learnable", "fixed_absolute", "none"],
        choices=["sinusoidal", "learnable", "fixed_absolute", "none"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("4/results"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(
        "任务：输入一个整数序列，输出它的反转结果。"
        "这里比较几种位置编码对结果的影响。"
    )

    train_dataset = ReverseDataset(args.train_samples, args.seq_len, args.vocab_size)
    val_dataset = ReverseDataset(args.val_samples, args.seq_len, args.vocab_size)

    all_history = []
    samples = {}
    for pos_encoding in args.pos_encodings:
        print(f"\n=== Experiment: {pos_encoding} positional encoding ===")
        history, sample_x, sample_y, sample_pred = train_one_experiment(
            args,
            pos_encoding,
            device,
            train_dataset,
            val_dataset,
        )
        all_history.extend(history)
        samples[pos_encoding] = (sample_x, sample_y, sample_pred)

    csv_path = save_history(all_history, args.output_dir)
    plot_path = save_plot(all_history, args.output_dir)

    print("\n=== Sample predictions ===")
    for pos_encoding, (sample_x, sample_y, sample_pred) in samples.items():
        print(f"\n{pos_encoding}:")
        for i in range(sample_x.size(0)):
            print(
                f"input={sample_x[i].tolist()} "
                f"target={sample_y[i].tolist()} "
                f"pred={sample_pred[i].tolist()}"
            )

    best_rows = {}
    for row in all_history:
        name = row["pos_encoding"]
        if name not in best_rows or row["val_token_acc"] > best_rows[name]["val_token_acc"]:
            best_rows[name] = row

    print("\n=== Best validation result ===")
    for name, row in sorted(best_rows.items(), key=lambda item: item[1]["val_token_acc"], reverse=True):
        print(
            f"{name:14s} token_acc={row['val_token_acc']:.4f} "
            f"seq_acc={row['val_seq_acc']:.4f} at epoch={row['epoch']}"
        )

    print(f"\nSaved CSV: {csv_path}")
    if plot_path is not None:
        print(f"Saved plot: {plot_path}")

    print(
        "\n结论：反转序列需要模型知道每个 token 的位置。"
        "sinusoidal 和 learnable 编码一般能学到镜像位置的对应关系；"
        "没有位置编码时，模型缺少顺序信息，准确率通常会低很多。"
        "fixed_absolute 的结果可以作为一个简单对照。"
    )

if __name__ == "__main__":
    main()
