# ck_gfx906

VALU (non-WMMA) conv3d kernels for AMD gfx906 (MI50/MI60), for comfy_kitchen.

gfx906 has no WMMA/MFMA so comfy_kitchen's HIP fp16 conv3d cannot run: ComfyUI's
VAE decode falls back to eager NCDHW convs, which is the dominant cost of a
SeedVR2 batch. `conv3d_gfx906.hip` is an implicit-GEMM fp16 NDHWC conv3d for the
VALU pipe: fp32 accumulate via v_dot2_f32_f16, ds_read_b128 LDS staging,
interleaved row ownership (bank-conflict-free), fp16 bias + optional residual
epilogue - same I/O contract as upstream conv3d_fp16.hip.

`upstream/` = vendored comfy_kitchen HIP sources for reference/diff.

Sibling project: github.com/dc740/gfx906 (attention kernels, same silicon rules).

Status: in development, not yet GPU-validated.
