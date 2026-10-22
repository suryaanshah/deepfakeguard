# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo==0.25.2",
#     "numpy>=2.0",
#     "pyarrow>=15.0",
#     "pillow>=11.0.0",
#     "scikit-learn>=1.6.0",
#     "torch>=2.11",
#     "torchvision>=0.26.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import marimo as mo
    import zipfile

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from torchvision import transforms

    # reproducible run
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)


@app.cell
def _(_cell_0_HEAD_DETECTOR):
    # --- Dataset helpers ---------------------------------------------------------
    # The pipeline needs two archives in the working directory:
    #   FF++.zip    -> real/<video>/000.png and fake/<method>/<video>/000.png
    #   CelebDF.zip -> Celeb_V2/Test/{real,fake}/*.jpg
    # Both are fetched from public HuggingFace datasets the first time this runs.
    import os
    import struct
    import subprocess
    import tempfile
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    FF_ZIP_URL = "https://huggingface.co/datasets/bitmind/FaceForensicsC23/resolve/main/FaceForensics++_C23.zip"
    CELEBDF_PARQUET_URL = "https://huggingface.co/datasets/thenewsupercell/celeb-df-image-dataset/resolve/main/data/test-00000-of-00001.parquet"
    FF_ZIP_PATH = "FF++.zip"
    CELEBDF_ZIP_PATH = "CelebDF.zip"


    def http_get_range(url, start, end, retries=5):
        # Fetch bytes[start:end+1] of a remote file with HTTP range requests.
        last_error = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    return response.read()
            except Exception as error:
                last_error = error
                time.sleep(2 * (attempt + 1))
        raise last_error


    def zip64_values(extra):
        # Pull the 64-bit sizes/offsets out of a zip64 extended information field.
        offset = 0
        while offset + 4 <= len(extra):
            header_id, size = struct.unpack("<HH", extra[offset:offset + 4])
            body = extra[offset + 4:offset + 4 + size]
            if header_id == 0x0001:
                count = len(body) // 8
                return list(struct.unpack("<%dQ" % count, body[:count * 8]))
            offset += 4 + size
        return []


    class RemoteZip:
        # Random access to a remote .zip over HTTP range requests: nothing is
        # downloaded until a member is actually requested.
        def __init__(self, url):
            self.url = url
            head = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(head, timeout=60) as response:
                self.size = int(response.headers.get("Content-Length"))
            tail = http_get_range(url, max(0, self.size - 300000), self.size - 1)
            end_of_central = tail.rfind(b"PK\x05\x06")
            locator = tail.rfind(b"PK\x06\x07")
            if locator >= 0:
                zip64_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
                record = http_get_range(url, zip64_offset, zip64_offset + 56)
                directory_size, directory_offset = struct.unpack("<QQ", record[40:56])
            else:
                directory_size, directory_offset = struct.unpack(
                    "<II", tail[end_of_central + 12:end_of_central + 20]
                )
            directory = http_get_range(
                url, directory_offset, directory_offset + directory_size - 1
            )
            self.entries = {}
            position = 0
            while (
                position + 46 <= len(directory)
                and directory[position:position + 4] == b"PK\x01\x02"
            ):
                method = struct.unpack("<H", directory[position + 10:position + 12])[0]
                compressed = struct.unpack("<I", directory[position + 20:position + 24])[0]
                uncompressed = struct.unpack("<I", directory[position + 24:position + 28])[0]
                name_len, extra_len, comment_len = struct.unpack(
                    "<HHH", directory[position + 28:position + 34]
                )
                local_offset = struct.unpack("<I", directory[position + 42:position + 46])[0]
                name = directory[position + 46:position + 46 + name_len].decode("utf-8", "replace")
                extra = directory[
                    position + 46 + name_len:position + 46 + name_len + extra_len
                ]
                if 0xFFFFFFFF in (uncompressed, compressed, local_offset):
                    values = zip64_values(extra)
                    index = 0
                    if uncompressed == 0xFFFFFFFF:
                        uncompressed = values[index]
                        index += 1
                    if compressed == 0xFFFFFFFF:
                        compressed = values[index]
                        index += 1
                    if local_offset == 0xFFFFFFFF:
                        local_offset = values[index]
                self.entries[name] = (local_offset, compressed, method)
                position += 46 + name_len + extra_len + comment_len

        def read(self, name):
            # Return the uncompressed bytes of a single member.
            local_offset, compressed, method = self.entries[name]
            header = http_get_range(self.url, local_offset, local_offset + 29)
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            start = local_offset + 30 + name_len + extra_len
            payload = http_get_range(self.url, start, start + compressed - 1)
            if method == 0:
                return payload
            import zlib

            return zlib.decompressobj(-15).decompress(payload)


    def decode_first_frame(video_bytes):
        # Decode the first frame of an mp4 into a PIL image.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(video_bytes)
            path = handle.name
        try:
            try:
                import cv2

                capture = cv2.VideoCapture(path)
                ok, frame = capture.read()
                capture.release()
                if ok:
                    return Image.fromarray(frame[:, :, ::-1])
            except Exception:
                pass
            import io

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                capture_output=True,
                check=True,
            )
            return Image.open(io.BytesIO(result.stdout))
        finally:
            os.unlink(path)


    def head_detector():
        # COCO person detector, used to localise the head inside a frame.
        global _HEAD_DETECTOR
        if _HEAD_DETECTOR is None:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _HEAD_DETECTOR = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            ).eval().to(device)
        return _HEAD_DETECTOR


    def head_crop(image):
        # Crop the head region of a frame and resize it to 224x224.
        frame = np.asarray(image)
        height, width = frame.shape[:2]
        scale = 640 / width
        small = image.resize((640, max(1, int(height * scale))))
        tensor = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float() / 255.0
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            detections = head_detector()([tensor])[0]
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        keep = (labels == 1) & (scores > 0.5)
        if keep.any():
            box = boxes[keep][int(np.argmax(scores[keep]))]
            x0, y0, x1, y1 = box
            head_height = (y1 - y0) * 0.42
            center_x = (x0 + x1) / 2
            center_y = y0 + head_height / 2
            side = max(head_height, x1 - x0) * 1.5
            left = max(0, int((center_x - side / 2) / scale))
            top = max(0, int((center_y - side / 2) / scale))
            right = min(width, int((center_x + side / 2) / scale))
            bottom = min(height, int((center_y + side / 2) / scale))
            if right > left and bottom > top:
                return image.crop((left, top, right, bottom)).resize((224, 224))
        side = min(height, width)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side)).resize((224, 224))


    def build_ff_zip(path=FF_ZIP_PATH, train_videos=200, test_videos=100):
        # Build FF++.zip: one head crop per video, laid out like the original
        # archive (real/<video>/000.png and fake/<method>/<video>/000.png).
        import io

        remote = RemoteZip(FF_ZIP_URL)
        prefix = "FaceForensics++_C23/"
        real, fake = {}, {}
        for name in remote.entries:
            if not name.endswith(".mp4"):
                continue
            parts = name[len(prefix):].split("/")
            if parts[0] == "real":
                real[parts[1][:-4]] = name
            elif parts[0] == "fake":
                fake.setdefault(parts[1], {})[parts[2][:-4].split("_")[0]] = name
        real_ids = sorted(real, key=int)
        train_ids = real_ids[:train_videos]
        test_ids = real_ids[train_videos:train_videos + test_videos]
        plan = [("real", video_id, real[video_id]) for video_id in train_ids + test_ids]
        plan += [
            ("fake/Deepfakes", video_id, fake["Deepfakes"][video_id])
            for video_id in train_ids
            if video_id in fake.get("Deepfakes", {})
        ]
        for method in ("Face2Face", "FaceShifter", "FaceSwap"):
            plan += [
                (f"fake/{method}", video_id, fake[method][video_id])
                for video_id in test_ids
                if video_id in fake.get(method, {})
            ]
        print(f"Fetching {len(plan)} FaceForensics++ videos ...")

        def download(item):
            kind, video_id, member = item
            return kind, video_id, remote.read(member)

        with ThreadPoolExecutor(max_workers=10) as pool:
            videos = list(pool.map(download, plan))

        frames = []
        for kind, video_id, payload in videos:
            frames.append((kind, video_id, head_crop(decode_first_frame(payload))))
            if len(frames) % 100 == 0:
                print(f"  {len(frames)}/{len(videos)} frames cropped")

        temporary = path + ".tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for kind, video_id, frame in frames:
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                archive.writestr(f"{kind}/{video_id}/000.png", buffer.getvalue())
        os.replace(temporary, path)
        print(f"Wrote {path} with {len(frames)} frames")
        return path


    def build_celebdf_zip(path=CELEBDF_ZIP_PATH, per_class=100):
        # Build CelebDF.zip: Celeb_V2/Test/real/*.jpg and .../fake/*.jpg.
        import pyarrow.parquet as pq

        parquet_path = "celebdf_test.parquet"
        if not os.path.exists(parquet_path):
            urllib.request.urlretrieve(CELEBDF_PARQUET_URL, parquet_path)
        table = pq.read_table(parquet_path)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        fake = sorted(
            [d for d, label in zip(images, labels) if label == 0], key=lambda d: d["path"]
        )[:per_class]
        real = sorted(
            [d for d, label in zip(images, labels) if label == 1], key=lambda d: d["path"]
        )[:per_class]
        temporary = path + ".tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            for item in real:
                archive.writestr(f"Celeb_V2/Test/real/{item['path']}", item["bytes"])
            for item in fake:
                archive.writestr(f"Celeb_V2/Test/fake/{item['path']}", item["bytes"])
        os.replace(temporary, path)
        print(f"Wrote {path} with {len(real)} real and {len(fake)} fake images")
        return path


    def ensure_datasets(ff_path=FF_ZIP_PATH, celebdf_path=CELEBDF_ZIP_PATH):
        # Download + build the archives the pipeline needs (idempotent).
        if not os.path.exists(ff_path):
            build_ff_zip(ff_path)
        if not os.path.exists(celebdf_path):
            build_celebdf_zip(celebdf_path)
        return {
            "FF++.zip": round(os.path.getsize(ff_path) / 1e6, 1),
            "CelebDF.zip": round(os.path.getsize(celebdf_path) / 1e6, 1),
        }


    return (ensure_datasets,)


@app.cell
def _(ensure_datasets):
    # Fetch the FaceForensics++ and Celeb-DF archives (first run only) and report
    # what is available on disk.
    dataset_report = ensure_datasets()
    dataset_report
    return


@app.class_definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@app.function
def transform_to_tensor_with_resize(images, labels):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_tensors = torch.stack([transform(image) for image in images])

    labels_tensor = torch.tensor(labels)

    print("Image tensor shape:", image_tensors.shape)
    print("Labels shape:", labels_tensor.shape)
    return image_tensors, labels_tensor


@app.function
def model_train(model, image_tensors, labels_tensor):
    loss_function = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    image_tensors = image_tensors.to(DEVICE)
    labels_tensor = labels_tensor.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 80

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(image_tensors)

        # Calculate loss
        loss = loss_function(outputs, labels_tensor)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        print("Epoch:", epoch + 1, "Loss:", loss.item())

    with torch.no_grad():
        outputs = model(image_tensors)
        predictions = torch.argmax(outputs, dim=1)

    print("Actual labels:    ", labels_tensor.tolist())
    print("Predicted labels: ", predictions.tolist())


@app.function
def load_one_frame_per_video(zip_path, real_count, fake_count):

    images = []
    labels = []

    with zipfile.ZipFile(zip_path, "r") as z:

        # ---------- REAL ----------
        real_folders = sorted(set(
            path.split("/")[1]
            for path in z.namelist()
            if path.startswith("real/") and path.endswith(".png")
        ))

        for folder in real_folders[:real_count]:

            # Find first PNG inside this video folder
            files = [
                path for path in z.namelist()
                if path.startswith(f"real/{folder}/")
                and path.endswith(".png")
            ]

            if len(files) == 0:
                continue

            image_path = files[0]

            with z.open(image_path) as f:
                image = Image.open(f).convert("RGB")
                images.append(image.copy())

            labels.append(0)   # 0 = Real


        # ---------- DEEPFAKE ----------
        fake_folders = sorted(set(
            path.split("/")[2]
            for path in z.namelist()
            if path.startswith("fake/Deepfakes/")
            and path.endswith(".png")
        ))

        for folder in fake_folders[:fake_count]:

            files = [
                path for path in z.namelist()
                if path.startswith(f"fake/Deepfakes/{folder}/")
                and path.endswith(".png")
            ]

            if len(files) == 0:
                continue

            image_path = files[0]

            with z.open(image_path) as f:
                image = Image.open(f).convert("RGB")
                images.append(image.copy())

            labels.append(1)   # 1 = Deepfake

    return images, labels


@app.cell
def train():
    images, labels = load_one_frame_per_video("FF++.zip", 200, 200)
    model = SimpleCNN()
    image_tensors, labels_tensor = transform_to_tensor_with_resize(images, labels)
    model_train(model, image_tensors, labels_tensor)
    return (model,)


@app.cell
def _():
    def test_set(folder_name):
        test_images = []
        test_labels = []

        data_folder = zipfile.Path("CelebDF.zip")

        # Loading fake images
        FAKE_DIR = data_folder / f'Celeb_V2/{folder_name}/fake'
        image_files = [x for x in FAKE_DIR.iterdir() if x.name.endswith(".jpg")][:100]
        for image_path in image_files:
            with image_path.open("rb") as f:
                image = Image.open(f).convert("RGB")
                test_images.append(image)
        for i in range(100):
            test_labels.append(1)

        # Loading real images
        REAL_DIR = data_folder / f'Celeb_V2/{folder_name}/real'
        image_files = [x for x in REAL_DIR.iterdir() if x.name.endswith(".jpg")][:100]
        for image_path in image_files:
            with image_path.open("rb") as f:
                image = Image.open(f).convert("RGB")
                test_images.append(image)
        for i in range(100):
            test_labels.append(0)

        print("Images:", len(test_images))
        print("Labels:", len(test_labels))
        return test_images, test_labels


    # The notebook keeps its original helper name, but pytest (which some hosts
    # run over this file) would try to collect it as a test case.
    test_set.__test__ = False

    return (test_set,)


@app.function
def model_test(model, image_tensors, labels_tensor):
    loss_function = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    image_tensors = image_tensors.to(DEVICE)
    labels_tensor = labels_tensor.to(DEVICE)

    # Forward Pass
    with torch.no_grad():

        outputs = model(image_tensors)
        # Calculate loss
        loss = loss_function(outputs, labels_tensor)
        predictions = torch.argmax(outputs, dim=1)

        # Probability/score that an image is fake
        fake_scores = torch.softmax(outputs, dim=1)[:, 1]

    predictions = predictions.cpu()
    fake_scores = fake_scores.cpu()
    labels_tensor = labels_tensor.cpu()

    print("Test Loss:", loss.item())
    print("Actual labels:    ", labels_tensor.tolist())
    print("Predicted labels: ", predictions.tolist())
    return predictions, fake_scores


@app.function
def calculate_metrics(labels, predictions, scores):

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions, zero_division=0)

    recall = recall_score(labels, predictions, zero_division=0)

    f1 = f1_score(labels, predictions, zero_division=0)

    auc = roc_auc_score(labels, scores)

    ap = average_precision_score(labels, scores)

    pauc = roc_auc_score(
        labels,
        scores,
        max_fpr=0.10
    )

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[index] + fnr[index]) / 2

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1       :", f1)
    print("AUC      :", auc)
    print("AP       :", ap)
    print("pAUC     :", pauc)
    print("EER      :", eer)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "AP": ap,
        "pAUC": pauc,
        "EER": eer
    }


@app.cell
def eval_celebdf(model, test_set):
    print("----------------------Testing the CNN Model on unseen Celeb-DF V2 dataset--------------------------\n")
    test_images, test_labels = test_set('Test')
    test_image_tensors, test_labels_tensor = transform_to_tensor_with_resize(test_images, test_labels)
    celebdf_predictions, celebdf_fake_scores = model_test(model, test_image_tensors, test_labels_tensor)
    celebdf_metrics = calculate_metrics(test_labels_tensor.numpy(), celebdf_predictions.numpy(), celebdf_fake_scores.numpy())
    return (celebdf_metrics,)


@app.cell
def _():
    def test_load_one_frame_per_video(zip_path, folder_name, real_count, fake_count):

        images = []
        labels = []

        with zipfile.ZipFile(zip_path, "r") as z:

            # ---------- REAL ----------
            real_folders = sorted(set(
                path.split("/")[1]
                for path in z.namelist()
                if path.startswith("real/") and path.endswith(".png")
            ))

            for folder in real_folders[200:(real_count + 200)]:

                # Find first PNG inside this video folder
                files = sorted([
                    path for path in z.namelist()
                    if path.startswith(f"real/{folder}/")
                    and path.endswith(".png")
                ])

                if len(files) == 0:
                    continue

                image_path = files[0]

                with z.open(image_path) as f:
                    image = Image.open(f).convert("RGB")
                    images.append(image.copy())

                labels.append(0)   # 0 = Real


            # ---------- DEEPFAKE ----------
            fake_folders = sorted(set(
                path.split("/")[2]
                for path in z.namelist()
                if path.startswith(f"fake/{folder_name}/")
                and path.endswith(".png")
            ))

            for folder in fake_folders[:fake_count]:

                files = sorted([
                    path for path in z.namelist()
                    if path.startswith(f"fake/{folder_name}/{folder}/")
                    and path.endswith(".png")
                ])

                if len(files) == 0:
                    continue

                image_path = files[0]

                with z.open(image_path) as f:
                    image = Image.open(f).convert("RGB")
                    images.append(image.copy())

                labels.append(1)   # 1 = manipulated image

        return images, labels

    # The notebook keeps its original helper name, but pytest (which some
    # hosts run over this file) would collect it as a test case.
    test_load_one_frame_per_video.__test__ = False

    return (test_load_one_frame_per_video,)


@app.cell
def eval_face2face(model, test_load_one_frame_per_video):
    print("------------------------Testing on unseen Face2Face dataset from FaceForensics++-----------------------\n")
    images_face2face, labels_face2face = test_load_one_frame_per_video("FF++.zip", "Face2Face", 100, 100)
    image_tensors_face2face, labels_tensor_face2face = transform_to_tensor_with_resize(images_face2face, labels_face2face)
    face2face_predictions, face2face_fake_scores = model_test(model, image_tensors_face2face, labels_tensor_face2face)
    face2face_metrics = calculate_metrics(labels_tensor_face2face.numpy(), face2face_predictions.numpy(), face2face_fake_scores.numpy())
    return (face2face_metrics,)


@app.cell
def eval_faceshifter(model, test_load_one_frame_per_video):
    print("------------------------Testing on unseen FaceShifter dataset from FaceForensics++-----------------------\n")
    images_faceshifter, labels_faceshifter = test_load_one_frame_per_video("FF++.zip", "FaceShifter", 100, 100)
    image_tensors_faceshifter, labels_tensor_faceshifter = transform_to_tensor_with_resize(images_faceshifter, labels_faceshifter)
    faceshifter_predictions, faceshifter_fake_scores = model_test(model, image_tensors_faceshifter, labels_tensor_faceshifter)
    faceshifter_metrics = calculate_metrics(labels_tensor_faceshifter.numpy(), faceshifter_predictions.numpy(), faceshifter_fake_scores.numpy())
    return (faceshifter_metrics,)


@app.cell
def eval_faceswap(model, test_load_one_frame_per_video):
    print("------------------------Testing on unseen FaceSwap dataset from FaceForensics++-----------------------\n")
    images_faceswap, labels_faceswap = test_load_one_frame_per_video("FF++.zip", "FaceSwap", 100, 100)
    image_tensors_faceswap, labels_tensor_faceswap = transform_to_tensor_with_resize(images_faceswap, labels_faceswap)
    faceswap_predictions, faceswap_fake_scores = model_test(model, image_tensors_faceswap, labels_tensor_faceswap)
    faceswap_metrics = calculate_metrics(labels_tensor_faceswap.numpy(), faceswap_predictions.numpy(), faceswap_fake_scores.numpy())
    return (faceswap_metrics,)


@app.cell
def summary(
    celebdf_metrics,
    face2face_metrics,
    faceshifter_metrics,
    faceswap_metrics,
):
    _summary = {
        "Celeb-DF v2 (Test)": celebdf_metrics,
        "FF++ Face2Face": face2face_metrics,
        "FF++ FaceShifter": faceshifter_metrics,
        "FF++ FaceSwap": faceswap_metrics,
    }
    mo.md(
        "| Test set | Accuracy | Precision | Recall | F1 | AUC | AP | pAUC | EER |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {name} | {m['Accuracy']:.3f} | {m['Precision']:.3f} | {m['Recall']:.3f} "
            f"| {m['F1']:.3f} | {m['AUC']:.3f} | {m['AP']:.3f} | {m['pAUC']:.3f} | {m['EER']:.3f} |"
            for name, m in _summary.items()
        )
    )
    return


if __name__ == "__main__":
    app.run()
