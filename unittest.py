#! /usr/bin/python3

import torch
import torch.nn as nn
import unittest
import lobs_utils

# 使用简单神经网络进行测试
class SimpleDnn(lobs_utils.LobsDnnModel):
    def __init__(self):
        super(SimpleDnn, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2, 1)
        self.withReLUs = set(["fc1"])
        self.fc1.weight.data = torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float32)
        self.fc2.weight.data = torch.tensor([[0.4, -0.3]], dtype=torch.float32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TestLOBS(unittest.TestCase):
    def test_simple(self):
        model = SimpleDnn()
        model.resetHessianStats()
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        lobs_utils.updateHessianStats(model, inputs)
        layers = list(model.named_children())
        self.assertEqual(layers[0][0], "fc1")
        lobs_utils.calcHessiansAndPinvs(model)

        # 第一层删除第一个权重和第四个、第五个权重
        layer = layers[0][1]
        hessian_block = model.hessians[0]
        self.assertTrue(torch.equal(hessian_block, torch.tensor([[2, 4, 6],[4, 8, 12],[6, 12, 18]], dtype=torch.float32)))
        hpinv_block = model.hpinvs[0]
        print(hpinv_block)
        self.assertTrue(torch.allclose(hpinv_block, torch.tensor([[0.0026, 0.0051, 0.0077], [0.0051, 0.0102, 0.0153], [0.0077, 0.0153, 0.0230]], dtype=torch.float32), rtol=1e-6, atol=1e-4))
        indices = torch.tensor([0, 4, 5])
        original_weight = layer.weight.data.clone()
        with torch.no_grad():
            weight, _, original_delta = lobs_utils.optimal_brain_surgeon(layer, indices, hessian_block, hpinv_block)
            print(original_delta)
            self.assertTrue(torch.allclose(original_delta, torch.tensor([[-0.00052], [-0.00102], [-0.00154], [-0.0318], [-0.06323999999999999], [-0.09504]], dtype=torch.float32), rtol=1e-4, atol=1e-3))
            print(weight)
            self.assertTrue(torch.allclose(weight, torch.tensor([[0.0, 0.199, -0.1015], [0.2684, 0.0, 0.0]], dtype=torch.float32), rtol=1e-6, atol=1e-4))
        inputs = inputs.view(3, 1)
        print(original_weight)
        outputs_delta = torch.matmul(original_weight, inputs) - torch.matmul(weight, inputs)
        print(outputs_delta)
        actualMSE = torch.norm(outputs_delta, p=2)
        print(actualMSE)
        self.assertAlmostEqual(actualMSE.item(), 0.83844, places=5)

if __name__ == '__main__':
    unittest.main()
