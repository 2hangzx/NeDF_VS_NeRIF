import torch

checkpoint = torch.load('E:/github_upload/py-project/torch-ngp-bos-test-nomask-NGU-RIintegral-3Dmask/case1_3x128_2^19_hashencode_auto&disc_mask/checkpoints/ngp_ep0834.pth', map_location='cpu')
print(checkpoint.keys())
print(checkpoint['stats'])


import matplotlib.pyplot as plt

# 提取训练损失和验证损失
train_loss_list = checkpoint['stats']['loss']
valid_loss_list = checkpoint['stats']['valid_loss'] * len(train_loss_list)  # 目前只有一个 valid_loss，可以假设它对所有 epoch 相同

# 画图
plt.plot(train_loss_list, label='Train Loss')

# 如果 valid_loss 只有一个值，你可以在图上绘制一条水平线表示
# plt.axhline(y=valid_loss_list[0], color='r', linestyle='--', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve from Checkpoint')
plt.legend()
plt.grid(True)
plt.show()