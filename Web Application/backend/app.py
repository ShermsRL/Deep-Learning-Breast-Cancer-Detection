from flask import Flask, render_template, request, send_file, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import numpy as np
import torch.nn as nn
import pydicom
import tempfile
import os
import skimage
import base64

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')


# U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Down-Sample Portion (Left side of U-Net)
        self.enc1 = DoubleConv(in_channels, 32)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = DoubleConv(32, 64)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = DoubleConv(64, 128)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.enc4 = DoubleConv(128, 256)
        self.down4 = nn.MaxPool2d(kernel_size=2)

        # Bridge to Up-Sample Portion
        self.bridge = DoubleConv(256, 512)

        # Up-Sample Portion (Right side of U-Net)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(64, 32)

        # Output
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bridge(self.down4(e4))
        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return self.output(d4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1)

try:
    checkpoint = torch.load("./model/UNetFinal", map_location=device)
    if torch.cuda.is_available():
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")

model.to(device)
model.eval()

# Classifier model
classifier_model = models.vgg16(pretrained=False)
num_features = classifier_model.classifier[6].in_features
classifier_model.classifier[6] = nn.Linear(num_features, 2)  # Adjust output classes
classifier_model.load_state_dict(torch.load("./model/vgg16_best_fold5.pth", weights_only=True))
classifier_model.to(device)
classifier_model.eval()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/original', methods=['POST'])
def get_original_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        dicom_data = pydicom.dcmread(temp_path)
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        input_image = torch.from_numpy(pixel_array).float().unsqueeze(0)

        transform = transforms.Compose([
            transforms.Resize((512, 512))  # Standardizes to [-1,1]
        ])

        input_tensor = transform(input_image).squeeze().numpy()

        segmented_image = Image.fromarray(input_tensor.astype(np.uint16))

        img_buffer = io.BytesIO()
        segmented_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        app.logger.error(e)
        return f'Error processing DICOM image: {str(e)}', 400

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as f:
            f.write(file.read())
            temp_path = f.name

        dicom = pydicom.dcmread(temp_path)
        img_arr = dicom.pixel_array.astype(float)
        norm_img = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-6)  # 0-1 float

        display_img = (norm_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(display_img).convert('L')

        input_img = pil_img.resize((512, 512))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        input_tensor = transform(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            mask_prob = torch.sigmoid(output).squeeze().cpu().numpy()

        mask_resized = skimage.transform.resize(mask_prob, img_arr.shape, order=0, preserve_range=True)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        ys, xs = np.where(mask_bin)
        if len(ys) == 0 or len(xs) == 0:
            return jsonify({"error": "No lesion detected."}), 200
        margin = 10
        min_y, max_y = max(0, ys.min() - margin), min(img_arr.shape[0], ys.max() + margin)
        min_x, max_x = max(0, xs.min() - margin), min(img_arr.shape[1], xs.max() + margin)
        roi = img_arr[min_y:max_y, min_x:max_x]

        roi_norm = (roi - roi.min()) / (roi.max() - roi.min() + 1e-6)
        roi_img = Image.fromarray((roi_norm * 255).astype(np.uint8)).convert('L').resize((224, 224))

        classifier_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        roi_tensor = classifier_transform(roi_img).unsqueeze(0).to(device)
        roi_tensor = roi_tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            clf_output = classifier_model(roi_tensor)
            probs = torch.softmax(clf_output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        labels = ['Benign', 'Malignant']
        diagnosis = labels[pred_class]

        roi_b64 = pil_to_b64(roi_img)
        mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8)).convert('L')
        mask_img = mask_img.resize((224, 224))
        mask_b64 = pil_to_b64(mask_img)
        orig_img_resized = pil_img.resize((224, 224)).convert('RGBA')
        colored_mask = Image.new('RGBA', mask_img.size, color=(255, 0, 0, 100))
        mask_alpha = mask_img.point(lambda p: p > 0 and 100)
        colored_mask.putalpha(mask_alpha)
        overlay = Image.alpha_composite(orig_img_resized, colored_mask)
        overlay_b64 = pil_to_b64(overlay)

        return jsonify({
            "roi_image": roi_b64,
            "mask_image": mask_b64,
            "overlay_image": overlay_b64,
            "diagnosis": diagnosis,
            "confidence": f"{confidence * 100:.2f}%",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def pil_to_b64(pil_img):
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")


if __name__ == '__main__':
    app.run(debug=True)
