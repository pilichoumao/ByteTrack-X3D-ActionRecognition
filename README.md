
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
ByteTrack
(Multi-object Tracking)
      │
      ▼
Clip Buffer Manager
(Per-track Sliding Window Buffer)
      │
      ▼
X3D Action Recognition
(In-memory clip inference)
      │
      ▼
Visualization
(BBox + Track ID + Action Label)
````

---

## ✨ Features

* Multi-person tracking using **ByteTrack**
* Per-person action recognition using **X3D (MMAction2)**
* **Track-based sliding window clip buffer**
* **In-memory inference (no temporary video files)**
* Modular pipeline design
* Easy to extend to other action recognition models

---

## 📁 Project Structure

```text
ActionRecognition
│
├── core
│   ├── action_recognizer.py     # X3D inference wrapper
│   ├── tracker_adapter.py       # ByteTrack adapter
│   ├── clip_buffer.py           # per-track clip buffer
│   ├── visualizer.py            # drawing bounding boxes
│   ├── pipeline_demo.py         # main pipeline
│   └── config.py                # configuration
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

### Dependencies

This project relies on the dependencies required by the following two frameworks:

* **ByteTrack**
* **MMAction2**

Please install the dependencies according to the official installation guides of these projects.

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
* Action recognition is performed directly on **in-memory clips**

---

## 🙏 Acknowledgements

This project is built upon the following open-source projects:

* **ByteTrack**
* **MMAction2**
* **PyTorch**

---

## 📜 License

This project follows the licenses of the included open-source projects.

```
```
