import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def find_data_file(file_name=None):
    if file_name is not None:
        data_file = Path(file_name)
        if not data_file.exists():
            raise FileNotFoundError(f"cannot find data file: {data_file}")
        return data_file

    work_dir = Path(__file__).resolve().parent
    names = [
        "pollution.csv",
        "AirQuality.csv",
        "air_quality.csv",
        "PRSA_data_2010.1.1-2014.12.31.csv",
    ]

    for name in names:
        p = work_dir / name
        if p.exists():
            return p

    files = sorted(work_dir.glob("*.csv"))
    if files:
        return files[0]

    raise FileNotFoundError("put the air quality csv in folder 3, or pass --data")


def load_data(csv_file):
    df = pd.read_csv(csv_file)

    rename_cols = {
        "pollution": "pm2.5",
        "dew": "DEWP",
        "temp": "TEMP",
        "press": "PRES",
        "wnd_dir": "cbwd",
        "wnd_spd": "Iws",
        "snow": "Is",
        "rain": "Ir",
    }
    df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

    if "pm2.5" not in df.columns:
        raise ValueError("the csv should contain column 'pm2.5' or 'pollution'")

    df["pm2.5"] = df["pm2.5"].ffill().bfill()

    if "cbwd" in df.columns:
        df["cbwd"] = LabelEncoder().fit_transform(df["cbwd"].astype(str))

    drop_cols = [c for c in ["No", "date", "year", "month", "day", "hour"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()

    cols = ["pm2.5"] + [c for c in df.columns if c != "pm2.5"]
    return df[cols]


def make_samples(values, look_back=24):
    if len(values) <= look_back:
        raise ValueError("data is too short for this look_back")

    x, y = [], []
    for i in range(len(values) - look_back):
        x.append(values[i:i + look_back])
        y.append(values[i + look_back, 0])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


class PollutionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def inverse_pm25(y, scaler, feature_num):
    tmp = np.zeros((len(y), feature_num))
    tmp[:, 0] = y.reshape(-1)
    return scaler.inverse_transform(tmp)[:, 0]


def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            pred = model(bx)
            loss = criterion(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(bx)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history.append([epoch, train_loss, val_loss])

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d} | train loss {train_loss:.6f} | val loss {val_loss:.6f}")

    return pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"])


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            pred = model(bx).cpu().numpy()
            ps.append(pred)
            ys.append(by.numpy())

    return np.vstack(ys), np.vstack(ps)


def plot_loss(history, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("LSTM training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction(y_true, y_pred, save_path, n=300):
    n = min(n, len(y_true))
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:n], label="real")
    plt.plot(y_pred[:n], label="predicted")
    plt.xlabel("hour")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 forecasting result")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--look_back", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = Path(__file__).resolve().parent
    csv_file = find_data_file(args.data)
    print("data:", csv_file)

    df = load_data(csv_file)
    split = int(len(df) * 0.8)

    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(df.iloc[:split])
    test_values = scaler.transform(df.iloc[split - args.look_back:])

    x_train, y_train = make_samples(train_values, args.look_back)
    x_test, y_test = make_samples(test_values, args.look_back)

    val_size = max(1, int(len(x_train) * 0.1))
    if val_size >= len(x_train):
        raise ValueError("training data is too short, use a smaller look_back")

    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PollutionLSTM(input_size=x_train.shape[2]).to(device)

    print("train:", x_train.shape, y_train.shape)
    print("test :", x_test.shape, y_test.shape)
    print("device:", device)

    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    y_real, y_hat = predict(model, test_loader, device)
    y_real = inverse_pm25(y_real, scaler, df.shape[1])
    y_hat = inverse_pm25(y_hat, scaler, df.shape[1])

    rmse = mean_squared_error(y_real, y_hat) ** 0.5
    mae = mean_absolute_error(y_real, y_hat)
    r2 = r2_score(y_real, y_hat)

    print("\ntest result")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    result = pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2}])
    result.to_csv(base_dir / "lstm_result.csv", index=False)
    history.to_csv(base_dir / "lstm_loss.csv", index=False)

    torch.save(model.state_dict(), base_dir / "lstm_pm25.pth")
    plot_loss(history, base_dir / "lstm_loss.png")
    plot_prediction(y_real, y_hat, base_dir / "lstm_prediction.png")


if __name__ == "__main__":
    main()
