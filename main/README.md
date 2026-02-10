# 3DR Stage-1 (6号数据) 快速跑通说明

本目录用于第一阶段“数据摸底与 Baseline 搭建”，目前已基于 `main/data/6-ScanTask-仅4个微粒` 跑通：
- ROI 自动裁剪（把 4608×4096 的 DR 投影裁剪到仅包含样品区域）
- 2D 基线：用 `sin/Sino.sin` 做 fan-beam FBP（快速给出传统重建切片）
- 3D 基线：用裁剪后的 `projections.npy` + ASTRA CUDA 做 cone-beam FDK（输出 3D 体）
- 指标脚本：MSE/SSIM/PSNR/DICE（按 [0,1] 归一化后 slice-wise 平均）
- 微粒统计：阈值分割 + 连通域，输出中心点与等效球直径

## 1) ROI 裁剪并生成压缩投影

建议用 `--downsample 4`（节省空间，适合服务器 60G 限制的阶段）：
```
python3 main/scripts/stage1_prepare_6.py --downsample 4 --out_dir main/processed/6_ds4
```
输出：
- `main/processed/6_ds4/projections.npy`：形状 `(V,H,W)`，uint16
- `main/processed/6_ds4/projections_meta.json`：裁剪框与缩放倍率
- `main/processed/6_ds4/scan_recon_meta.json`：解析出的扫描/重建参数
- `preview_0001_full.png`/`preview_0001_crop.png`：裁剪效果预览

## 2) 2D 传统基线（FBP）
```
python3 main/scripts/run_fdk_slice.py --device cuda --out_dir main/output/fdk_2d
```
输出：
- `fbp_slice.npy` / `fbp_slice.png`
- `gt_slice.png`：来自 `Volume/0位置切片.raw` 的参考切片
- `report.json`：含 MSE/SSIM/PSNR/DICE

## 3) 3D 传统基线（ASTRA FDK CUDA）
```
python3 main/scripts/run_fdk_3d_astra.py --processed_dir main/processed/6_ds4 --out_dir main/output/fdk_3d --nx 160 --ny 160 --nz 160 --voxel_mm 0.25
```
输出：
- `vol.npy`：形状 `(Z,Y,X)`，float32
- `slice_zc.png`/`slice_yc.png`/`slice_xc.png`：三个正交切面预览
- `report.json`

## 4) 微粒统计（点云/连通域）
```
python3 main/scripts/particle_stats.py main/output/fdk_3d/vol.npy --voxel_mm 0.25 --min_vox 20 --max_vox 20000 --topk 20 --out_json main/output/fdk_3d/particles.json --out_csv main/output/fdk_3d/particles.csv
```

## 5) NAF 原型（2D，先出图）
这只是“端到端隐式场”思路的最小原型，先保证能跑通与出图：
```
python3 main/scripts/train_naf2d.py --steps 800 --device cuda --out_dir main/output/naf2d
```

