# Digit3D (3D MNIST): Geometry, Classification & Generative Modeling

Digit3D transforms the classic 2D MNIST dataset into a rich, lightweight 3D multimodal benchmark. It features lightweight `.obj` meshes, high-resolution sparse Signed Distance Fields (SDF), 3D point clouds with surface normals, and 2D source images for end-to-end 3D vision, representation learning, and generative modeling.

---

## Key Features

- **Multimodal Representations**:
  - **Dense 3D Meshes**: Watertight, Taubin-smoothed `.obj` meshes with controlled triangle counts (~500 faces).
  - **Sparse Voxel SDFs**: Offline-precomputed Signed Distance Fields compressed into sparse narrow-band `.npz` archives via GPU-accelerated BVH queries.
  - **Point Clouds with Normals**: Lightweight $[N, 6]$ point coordinates $(X, Y, Z)$ and surface normal vectors $(N_x, N_y, N_z)$.
  - **Paired 2D Images**: Source $28 \times 28$ PNG images for multimodal image $\to$ 3D tasks.
- **High-Performance Dataset Streaming**: Zero-extraction PyTorch datasets (`Digit3D`, `PointDigit3D`, `SparseDigit3D`) that stream directly from compressed zip archives into CPU/GPU memory without inode exhaustion.
- **Point Cloud Classification Benchmark**: `PointTransformerCls` architecture achieving **~99.0% accuracy** on 10-class 3D digit recognition, complete with automated misclassification export and error analysis.
- **Continuous Normalizing Flows & Generative Modeling**:
  - State-of-the-art flow matching algorithms: **Rectified Flow**, **Mean Flow**, and **SoFlow** (1-step generative flows).
  - Flexible conditioning modes: **Unconditional**, **Class-Conditioned** (`--class_cond`), and **Image-Conditioned** (`--img_cond`).
  - **Inference & Sampling**: Single-image 3D generation (`--img_path`), class filtering (`--class_label`), Classifier-Free Guidance (`--cfg_scale`), and latent class morphing/interpolation (`interpolation.py`).

---

## Repository Structure

```text
digit3d/
├── data/                               # Dataset construction & SDF computation pipeline
│   ├── construct.py                   # 2D MNIST -> 3D .obj mesh & .png construction
│   ├── compute.py                     # Mesh -> Sparse SDF narrow-band computation (GPU BVH)
│   └── run.sh                         # Complete end-to-end dataset generation bash script
│
├── docs/                               # Interactive academic project page & WebGL viewers
│
└── experiments/
    ├── pc/                            # Point Cloud experiments
    │   ├── classification/            # 3D Point Cloud classification benchmark
    │   │   ├── models.py              # PointTransformerCls model definition
    │   │   ├── train.py               # Classification training script
    │   │   ├── eval.py                # Evaluation & misclassified case export (.ply + .png)
    │   │   └── eval_results.json      # Benchmark metrics & per-class breakdown
    │   │
    │   └── generation/                # 3D Generative modeling & continuous flows
    │       ├── transformer.py         # PointTransformer, ClassConditioned, ImgConditionPointTransformer
    │       ├── train.py               # RectifiedFlow, MeanFlow, SoFlow training
    │       ├── generation.py          # Image-conditioned & class-conditioned 3D sampling
    │       ├── interpolation.py       # Latent shape morphing & class interpolation
    │       ├── fid.py                 # Fréchet Distance evaluation for 3D point clouds
    │       └── checkpoint.py          # Memory-efficient gradient checkpointing
    │
    └── sparse_voxel/                  # Sparse Voxel experiments
        ├── classification/            # Sparse Voxel ResNet classification benchmark
        │   ├── models.py              # SparseClassifier architecture
        │   ├── train.py               # Classification training script
        │   ├── eval.py                # Evaluation & misclassified error export (.ply + .png)
        │   └── eval_results.json      # Benchmark metrics & per-class breakdown
        │
        └── reconstruction/            # Sparse Voxel VAE 3D reconstruction
            ├── models.py              # SimpleSparseVAE architecture
            ├── train.py               # VAE reconstruction training script
            ├── generation.py          # 3D Mesh reconstruction & ply_samples export
            ├── eval.py                # Reconstruction evaluation (Chamfer Distance, SDF MSE)
            └── arch.py                # Layer-by-layer stride & coordinate inspection
```

---

## 1. Dataset Generation Pipeline

To generate the complete 70,000-sample dataset from scratch:

```bash
cd data/
bash run.sh
```

### Pipeline Steps:
1. **Mesh Construction (`construct.py`)**:
   Downloads MNIST via `torchvision`, computes distance transforms, applies spherical parabolic thickness along the Z-axis, extracts isosurfaces via Marching Cubes, smooths via Taubin filter, and decimates to ~500 triangles. Saves 70,000 paired `.obj` meshes and `.png` images into `src/`.
2. **Dense Archiving**: Compresses `src/` into `digit3d.zip`.
3. **Offline Sparse Voxelization (`compute.py`)**: Queries GPU-accelerated BVH trees with `conquer3d` to compute Signed Distance Fields, saving sparse narrow-band coordinates (`idx_grids`) and features (`sdf`) into `sdf/`.
4. **Sparse Archiving**: Compresses `sdf/` into `digit3d_sdf.zip`.

---

## 2. Dataset Usage in PyTorch

The dataset classes stream assets directly from zip archives without requiring manual extraction on disk:

```python
from conquer3d.data.dataset.digit3d import Digit3D, PointDigit3D, SparseDigit3D

# 1. Point Cloud Dataset (XYZ + Normals + Optional Paired Image)
point_dataset = PointDigit3D(
    root="~/.conquer3d/",
    train=True,
    download=True,
    num_points=512,
    return_img=True
)
points, features, label, img = point_dataset[0]
# points:   [512, 3] (XYZ)
# features: [512, 6] (XYZ + NxNyNz normals)
# label:    int (0-9)
# img:      [1, 28, 28] (Tensor in [0, 1])

# 2. Dense Mesh Dataset (Vertices & Faces)
mesh_dataset = Digit3D(root="~/.conquer3d/", train=False, download=True)
vertices, faces, label = mesh_dataset[0]

# 3. Sparse SDF Dataset (Voxel Grid Indices & SDF Values)
sdf_dataset = SparseDigit3D(root="~/.conquer3d/", train=False, download=True)
idx_grids, sdf_values, label = sdf_dataset[0]
```

---

## 3. Point Cloud Classification Benchmark

The classification pipeline trains and evaluates a `PointTransformerCls` (depth=4, in_channels=6, dim=128, share_planes=8, patch_size=32) on 10-class point cloud digit recognition.

### Training:
```bash
python experiments/pc/classification/train.py --epochs 100 --batch_size 32
```

### Evaluation & Error Case Export:
```bash
python experiments/pc/classification/eval.py
```
- **Accuracy**: **~98.95%** on the 10,000-sample test set.
- **Misclassification Export**: Automatically identifies any misclassified test samples and saves paired 2D images (`.png`) and 3D point clouds with normals (`.ply`) to `experiments/pc/classification/wrong/` for error analysis.

---

## 4. 3D Generative Modeling & Flow Matching

The generative suite implements continuous normalizing flows on 3D point clouds with surface normals.

### Generative Algorithms (`--mode`):
- `--mode 0`: **Rectified Flow** (Continuous straight-path flow matching)
- `--mode 1`: **Mean Flow** (Fast velocity matching with `torch.func.jvp`)
- `--mode 2`: **SoFlow** (1-step generative flow model)

---

### A. Training Generative Models

#### 1. Image-Conditioned Generation (2D Image $\to$ 3D Point Cloud):
```bash
# Rectified Flow
python experiments/pc/generation/train.py --mode 0 --img_cond --epochs 200 --batch_size 32

# Mean Flow
python experiments/pc/generation/train.py --mode 1 --img_cond --epochs 200 --batch_size 32

# SoFlow (1-step generator)
python experiments/pc/generation/train.py --mode 2 --img_cond --epochs 200 --batch_size 32
```

#### 2. Class-Conditioned Generation (Class ID $\to$ 3D Point Cloud):
```bash
python experiments/pc/generation/train.py --mode 0 --class_cond --epochs 200 --batch_size 32
```

#### 3. Unconditional Generation:
```bash
python experiments/pc/generation/train.py --mode 0 --epochs 200 --batch_size 32
```

---

### B. Inference & Sampling (`generation.py`)

#### 1. Generate 3D Point Cloud from a Custom 2D Image File:
Pass any 2D image file (`.png`, `.jpg`, etc.). The script automatically resizes/normalizes it to $1 \times 28 \times 28$:
```bash
python experiments/pc/generation/generation.py \
    --img_cond \
    --img_path /path/to/my_digit.png \
    --num_samples 5 \
    --cfg_scale 2.0
```

#### 2. Generate Conditioned on a Specific Digit Class from the Dataset:
```bash
python experiments/pc/generation/generation.py \
    --img_cond \
    --class_label 7 \
    --num_samples 10 \
    --cfg_scale 1.5
```

#### 3. Generate Sequential Samples from the Test Dataset:
```bash
python experiments/pc/generation/generation.py \
    --img_cond \
    --num_samples 10 \
    --sample_offset 0
```

All generated samples are saved as:
- Individual readable `.ply` point cloud files (with coordinates and normal vectors) in `runs/<exp_name>/ply_samples/`.
- Paired input `.png` images side-by-side (`sample_000_class_7_input.png` next to `sample_000_class_7.ply`).
- Complete PyTorch trajectory tensor `samples.pt`.

---

### C. Continuous Latent Class Interpolation (`interpolation.py`)

Morph continuously between any two digit classes ($A \to B$) through the unconditional latent flow:

```bash
python experiments/pc/generation/interpolation.py \
    --mode 0 \
    --class_start 0 \
    --class_end 8 \
    --k 10 \
    --steps 64 \
    --noise_strength 1.0
```

- Generates $k$ intermediate shapes smoothly transitioning from `class_start` to `class_end`.
- Exports all trajectory point clouds into `runs/<exp_name>/interpolation/from_0_to_8/`.

---

### D. Generative Quality Evaluation (`fid.py`)

Evaluate generative sample quality using 3D point cloud Fréchet Inception Distance (FID):

```bash
python experiments/pc/generation/fid.py --mode 0 --class_cond --batch_size 500
```

---

## 5. 3D Visualization

All generated `.ply` and `.obj` files can be directly opened in standard 3D viewers such as **MeshLab**, **CloudCompare**, **Blender**, or **macOS Preview**.

---

## References

- **Point-E (OpenAI)**: *Point-E: A System for Generating 3D Point Clouds from Complex Prompts*. Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, Mark Chen. [arXiv:2212.08751](https://arxiv.org/abs/2212.08751) | [GitHub](https://github.com/openai/point-e)
- **Rectified Flow**: *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. Xingchao Liu, Chengyue Gong, Qiang Liu. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) | [rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch)
- **Point Transformer**: *Point Transformer*. Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun. [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)
- **Classifier-Free Guidance**: *Classifier-Free Diffusion Guidance*. Jonathan Ho, Tim Salimans. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- **Conquer3D**: *Differentiable 3D Geometry and Fast GPU Spatial Acceleration Engine*. [GitHub](https://github.com/KhoiDOO/conquer3d)
