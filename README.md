
---

````markdown
# ByteTrack-X3D-ActionRecognition

![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![Status](https://img.shields.io/badge/Status-Working-success)

A lightweight **multi-person action recognition pipeline** based on **ByteTrack** and **X3D**.

This project combines **multi-object tracking** and **video action recognition**, enabling **per-person action prediction** from a single video stream.

---

## 🚀 System Architecture

```text
Video Stream
      │
      ▼
ByteTrack (Multi-object Tracking)
      │
      ▼
Clip Buffer Manager
(Per-track Sliding Window)
      │
      ▼
X3D Action Recognition
(MMAction2)
      │
      ▼
Visualization
(BBox + Track ID + Action Label)
````

---

## ✨ Features

* Multi-person tracking using **ByteTrack**
* Per-person action recognition using **X3D**
* Sliding-window clip buffer
* Modular pipeline design
* Easy to extend to other action recognition models

---

## 📁 Project Structure

```text
ActionRecognition
│
├── core
│   ├── action_recognizer.py
│   ├── tracker_adapter.py
│   ├── clip_buffer.py
│   ├── visualizer.py
│   ├── pipeline_demo.py
│   └── config.py
│
├── ByteTrack
│
├── mmaction2
│
├── video
│
├── output
│
├── tmp
│
└── README.md
```

---

## ⚙️ Environment

Recommended environment:

```text
Python 3.8
CUDA 11+
PyTorch
```

Install dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install mmengine mmcv-lite
```

---

## 📦 Models

### ByteTrack

```text
ByteTrack/pretrained/bytetrack_x_mot17.pth.tar
```

### X3D-S

```text
mmaction2/checkpoints/x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth
```

---

## ▶️ Run Demo

```bash
cd core
python pipeline_demo.py
```

Input video:

```text
video/demo.mp4
```

Output video:

```text
output/output_track_x3d.mp4
```

---

## 📊 Example Output

```text
frame 183 track 1 action = motorcycling score = 0.45
frame 215 track 1 action = welding score = 0.55
frame 247 track 1 action = bending metal score = 0.43
```

---

## ⚠️ Limitations

* Uses **Kinetics-400** pretrained labels
* No temporal smoothing yet
* Predictions may fluctuate between frames
* Temporary video clips are still used for inference

---

## 🛠 TODO

### Pipeline

* [ ] remove temporary video files
* [ ] switch to in-memory inference
* [ ] add temporal smoothing
* [ ] add action confidence filtering

### Performance

* [ ] async inference
* [ ] batch inference for multiple tracks
* [ ] optimize latency

### Models

* [ ] support SlowFast
* [ ] support VideoMAE
* [ ] support pose-based action recognition

### Features

* [ ] webcam support
* [ ] RTSP stream support
* [ ] REST API
* [ ] web visualization dashboard

---

## 🙏 Acknowledgements

This project is built upon the following open-source projects:

* **ByteTrack**
* **MMAction2**
* **PyTorch**

---

## 📜 License

This project follows the licenses of the included open-source projects.



