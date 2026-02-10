from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ScanParams:
    detector_rows: int
    detector_cols: int
    pixel_size_mm: float
    num_projections: int
    start_angle_deg: float
    rotation_ccw: bool
    dso_mm: float  # source to rotation center
    dsd_mm: float  # source to detector
    det_h_center_offset_px: float  # + means shift to the right in pixel index space
    det_v_center_offset_px: float

    @property
    def angles_rad(self) -> list[float]:
        start = math.radians(self.start_angle_deg)
        step = 2 * math.pi / float(self.num_projections)
        # For CCW rotation, we use increasing beta.
        # For CW, we use decreasing beta (equivalent to reversing view order).
        if self.rotation_ccw:
            return [start + i * step for i in range(self.num_projections)]
        return [start - i * step for i in range(self.num_projections)]


@dataclass(frozen=True)
class ReconParams:
    recon_size_full: int
    recon_x0: int
    recon_nx: int
    recon_y0: int
    recon_ny: int
    recon_nz: int
    voxel_size_mm: float
    slice_spacing_mm: float
    slice_start_mm: float
    det_h_spacing_mm: float
    det_v_spacing_mm: float
    det_h_center_offset_px: float
    det_v_center_offset_px: float
    dso_mm: float
    dsd_mm: float


def _xml_get_value(root: ET.Element, tag: str, name: str) -> str | None:
    for parent in root.findall(f".//{tag}"):
        if parent.attrib.get("name") == name:
            return parent.attrib.get("value")
        # nested structure <ScanPara><ScanPara .../></ScanPara>
        for child in parent.findall(f".//{tag}"):
            if child.attrib.get("name") == name:
                return child.attrib.get("value")
    return None


def load_scan_params(scan_xml: str | Path, recon_txt: str | Path | None = None) -> ScanParams:
    scan_xml = Path(scan_xml)
    root = ET.fromstring(scan_xml.read_text(encoding="utf-8", errors="ignore"))

    num_projections = int(float(_xml_get_value(root, "ScanPara", "投影数量") or "0"))
    detector_rows = int(float(_xml_get_value(root, "DetPara", "探测器行数") or "0"))
    detector_cols = int(float(_xml_get_value(root, "DetPara", "探测器列数") or "0"))
    pixel_size_mm = float(_xml_get_value(root, "DetPara", "像素尺寸mm") or "0.0")

    start_angle_deg = float(_xml_get_value(root, "ScanPara", "投影起始角度") or "0.0")
    rotation_dir = _xml_get_value(root, "ScanPara", "旋转方向") or "逆时针"
    rotation_ccw = "逆时针" in rotation_dir

    det_h_center_offset_px = float(_xml_get_value(root, "RebuildPara", "水平探测器中心方向偏移") or "0.0")
    det_v_center_offset_px = float(_xml_get_value(root, "RebuildPara", "垂直探测器中心方向偏移") or "0.0")

    # Prefer values from ReconPara.txt if provided (more complete).
    dso_mm = 0.0
    dsd_mm = 0.0
    if recon_txt is not None:
        rp = load_recon_params(recon_txt)
        dso_mm = rp.dso_mm
        dsd_mm = rp.dsd_mm
        if rp.det_h_center_offset_px != 0:
            det_h_center_offset_px = rp.det_h_center_offset_px
        if rp.det_v_center_offset_px != 0:
            det_v_center_offset_px = rp.det_v_center_offset_px

    if dso_mm == 0.0 or dsd_mm == 0.0:
        # Fallback: derive approximate from XML terms (may be device-specific).
        fod = _xml_get_value(root, "ScanPara", "FOD位置")
        fdd = _xml_get_value(root, "ScanPara", "FDD位置")
        if fod and fdd:
            dso_mm = float(fod)
            dsd_mm = float(fod) + float(fdd)

    return ScanParams(
        detector_rows=detector_rows,
        detector_cols=detector_cols,
        pixel_size_mm=pixel_size_mm,
        num_projections=num_projections,
        start_angle_deg=start_angle_deg,
        rotation_ccw=rotation_ccw,
        dso_mm=dso_mm,
        dsd_mm=dsd_mm,
        det_h_center_offset_px=det_h_center_offset_px,
        det_v_center_offset_px=det_v_center_offset_px,
    )


_KV_RE = re.compile(r"^\s*([^:：]+)\s*[:：]\s*(.*?)\s*$")


def load_recon_params(recon_txt: str | Path) -> ReconParams:
    recon_txt = Path(recon_txt)
    kv: dict[str, str] = {}
    for line in recon_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _KV_RE.match(line)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        kv[k] = v

    def get_int(key: str, default: int = 0) -> int:
        s = kv.get(key, str(default))
        return int(float(s.split()[0]))

    def get_float(key: str, default: float = 0.0) -> float:
        s = kv.get(key, str(default))
        return float(s.split()[0])

    return ReconParams(
        recon_size_full=get_int("重建图像规模"),
        recon_x0=get_int("重建切片X方向起始位置"),
        recon_nx=get_int("重建切片X方向像素数"),
        recon_y0=get_int("重建切片Y方向起始位置"),
        recon_ny=get_int("重建切片Y方向像素数"),
        recon_nz=get_int("重建图像层数"),
        voxel_size_mm=get_float("重建像素尺寸"),
        slice_spacing_mm=get_float("重建层间隔"),
        slice_start_mm=get_float("重建断层起始位置"),
        det_h_spacing_mm=get_float("相邻探测器水平间距"),
        det_v_spacing_mm=get_float("相邻探测器垂直间距"),
        det_h_center_offset_px=get_float("中心探测器水平方向偏移"),
        det_v_center_offset_px=get_float("中心探测器垂直方向偏移"),
        dso_mm=get_float("光源到旋转中心的距离"),
        dsd_mm=get_float("光源到探测器的距离"),
    )


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

