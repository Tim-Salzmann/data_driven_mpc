from torch.utils.data import Dataset
import deep_casadi.torch as dc


class GPToMLPDataset(Dataset):
    def __init__(self, gp_dataset):
        super().__init__()
        self.x = gp_dataset.x
        self.y = gp_dataset.y

    def stats(self):
        return self.x.mean(axis=0), self.x.std(axis=0), self.y.mean(axis=0), self.y.std(axis=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class NormalizedMLP(dc.TorchDeepCasadiModule):
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.input_size = self.model.input_size
        self.output_size = self.model.output_size
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)

    def forward(self, x):
        return (self.model((x - self.x_mean) / self.x_std) * self.y_std) + self.y_mean

    def cs_forward(self, x):
        return (self.model((x - self.x_mean.numpy()) / self.x_std.numpy()) * self.y_std.numpy()) + self.y_mean.numpy()
