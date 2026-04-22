import torch
import torch.nn as nn

class F110_YOLO(torch.nn.Module):
    def __init__(self):
        super(F110_YOLO, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, padding=1, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.batchnorm6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.batchnorm7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.batchnorm8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(32, 5, kernel_size=1, padding=0, stride=1)
        self.relu9 = nn.ReLU()

    def forward(self, x):
        debug = 0  # change this to 1 if you want to check network dimensions
        if debug == 1:
            print(0, x.shape)
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        if debug == 1:
            print(1, x.shape)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        if debug == 1:
            print(2, x.shape)
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        if debug == 1:
            print(3, x.shape)
        x = torch.relu(self.batchnorm4(self.conv4(x)))
        if debug == 1:
            print(4, x.shape)
        x = torch.relu(self.batchnorm5(self.conv5(x)))
        if debug == 1:
            print(5, x.shape)
        x = torch.relu(self.batchnorm6(self.conv6(x)))
        if debug == 1:
            print(6, x.shape)
        x = torch.relu(self.batchnorm7(self.conv7(x)))
        if debug == 1:
            print(7, x.shape)
        x = torch.relu(self.batchnorm8(self.conv8(x)))
        if debug == 1:
            print(8, x.shape)
        x = self.conv9(x)
        if debug == 1:
            print(9, x.shape)
        x = torch.cat([x[:, 0:3, :, :], torch.sigmoid(x[:, 3:5, :, :])], dim=1)

        return x

    def get_loss(self, result, truth, lambda_coord=5, lambda_noobj=1):
        x_loss = (result[:, 1, :, :] - truth[:, 1, :, :]) ** 2
        y_loss = (result[:, 2, :, :] - truth[:, 2, :, :]) ** 2
        w_loss = (
            torch.sqrt(result[:, 3, :, :].clamp(min=1e-6))
            - torch.sqrt(truth[:, 3, :, :].clamp(min=1e-6))
        ) ** 2
        h_loss = (
            torch.sqrt(result[:, 4, :, :].clamp(min=1e-6))
            - torch.sqrt(truth[:, 4, :, :].clamp(min=1e-6))
        ) ** 2
        class_loss_obj = (
            truth[:, 0, :, :] * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        )
        class_loss_noobj = (
            (1 - truth[:, 0, :, :])
            * lambda_noobj
            * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        )

        total_loss = torch.sum(
            lambda_coord * truth[:, 0, :, :] * (x_loss + y_loss + w_loss + h_loss)
            + class_loss_obj
            + class_loss_noobj
        )

        return total_loss


model = F110_YOLO()
state_dict = torch.load("model_200.pt",map_location='cuda')
model.load_state_dict(state_dict)

model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 540, 960).cuda()

torch.onnx.export(model, 
                  dummy_input, 
                  "model.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'])

"""
Convert to .engine:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_200.engine --fp16
"""