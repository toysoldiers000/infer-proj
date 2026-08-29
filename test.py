import torch, torch.fx as fx

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn   = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))

m = Toy()
x = torch.randn(1, 3, 8, 8)

# 导出 ATEN dialect 的图
ep = torch.export.export(m, (x,)).run_decompositions()
g = ep.graph
print("dialect =", ep.dialect)          # 打印 dialect 名字
print("FX nodes =", len(list(g.nodes))) # 数节点数
for node in g.nodes:
    if node.op == "call_function":
        print("  node:", node.target)   # 每个原子算子的名字
