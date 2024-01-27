import os
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional

@dataclass
class Options:
    # path to input mesh
    mesh: Optional[str] = None
    # input text prompt
    prompt: Optional[str] = None
    # additional positive prompt
    positive_prompt: str = "best quality, extremely detailed, masterpiece, high resolution, high quality"
    # additional negative prompt
    negative_prompt: str = "blur, lowres, cropped, low quality, worst quality, ugly, dark, shadow, oversaturated"
    # whether to append directional text prompt
    text_dir: bool = False
    # set mesh front-facing direction (camera front=+z, right=+x, up=+y, clock-wise rotation 90=1, 180=2, 270=3, e.g., +z, -y1)
    front_dir: str = "+z"

    # training iterations
    iters: int = 500
    # training resolution
    render_resolution: int = 512
    # training camera radius
    radius: float = 2.5
    # training camera fovy in degree
    fovy: float = 49.1
    # whether to allow geom training
    fix_geo: bool = False
    # whether to mix normal with rgb for geometry training
    mix_normal: bool = True
    # whether to pretrain texture first
    fit_tex: bool = True
    # pretrain texture iterations
    fit_tex_iters: int = 512

    # output folder
    outdir: str = '.'
    # output filename, default to {name}_fine.{ext}
    save: Optional[str] = None

    # guidance mode
    mode: Literal['SD', 'IF', 'IF2', 'SDCN', 'SD_NFSD', 'IF2_NFSD', 'SD_ISM', 'IF2_ISM'] = 'IF2'
    # renderer geometry mode
    geom_mode: Literal['mesh', 'diffmc', 'pbr_mesh', 'pbr_diffmc'] = 'diffmc'
    # renderer texture mode
    tex_mode: Literal['hashgrid', 'mlp', 'triplane'] = 'hashgrid'
    
    # training batch size per iter
    batch_size: int = 1
    # environmental texture
    env_texture: Optional[str] = None
    # environmental light scale
    env_scale: float = 2
    
    # DiffMC grid size
    mc_grid_size: int = 128
    # Mesh remeshing interval
    remesh_interval: int = 200
    # mesh decimation target face number
    decimate_target: int = 5e4
    # remesh target edge length (smaller value lead to finer mesh)
    remesh_size: float = 0.015
    # texture resolution
    texture_resolution: int = 1024
    # learning rate for hashgrid
    hashgrid_lr: float = 0.01
    # learning rate for feature MLP
    mlp_lr: float = 0.001
    # learning rate for SDF
    sdf_lr: float = 0.0001
    # learning rate for deformation
    deform_lr: float = 0.0001
    # learning rate for mesh geometry
    geom_lr: float = 0.0001

    # guidance loss weights
    lambda_sd: float = 1
    # mesh laplacian regularization weight
    lambda_lap: float = 0
    # mesh normal consistency weight (should be large enough)
    lambda_normal: float = 10000
    # mesh vertices offset penalty weight
    lambda_offsets: float = 100

    # whether to open a GUI
    gui: bool = False
    # GUI height
    H: int = 800
    # GUI width
    W: int = 800
    # whether to use CUDA rasterizer (in case OpenGL fails)
    force_cuda_rast: bool = False
    # whether to use GPU memory-optimized mode (slower, but uses less GPU memory)
    vram_O: bool = False


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['sd'] = 'coarse-level generation with stable-diffusion 2.'
config_defaults['sd'] = Options(
    mode='SD',
    iters=800,
)

config_doc['if'] = 'coarse-level generation with deepfloyd-if I.'
config_defaults['if'] = Options(
    mode='IF',
    iters=400,
)

config_doc['if2'] = 'fine-level refinement with deepfloyd-if II.'
config_defaults['if2'] = Options(
    mode='IF2',
    iters=400,
)

config_doc['sd_fixgeo'] = 'coarse-level generation with stable-diffusion 2, fixed goemetry.'
config_defaults['sd_fixgeo'] = Options(
    mode='SD',
    iters=800,
    fix_geo=True,
    geom_mode='mesh',
)

config_doc['if_fixgeo'] = 'coarse-level generation with deepfloyd-if I, fixed goemetry.'
config_defaults['if_fixgeo'] = Options(
    mode='IF',
    iters=400,
    fix_geo=True,
    geom_mode='mesh',
)

config_doc['if2_fixgeo'] = 'fine-level refinement with deepfloyd-if II, fixed goemetry.'
config_defaults['if2_fixgeo'] = Options(
    mode='IF2',
    iters=400,
    fix_geo=True,
    geom_mode='mesh',
)

def check_options(opt: Options):
    assert opt.mesh is not None, 'mesh path must be specified!'
    assert opt.prompt is not None, 'prompt must be specified!'

    if opt.save is None:
        input_name, input_ext = os.path.splitext(os.path.basename(opt.mesh))
        opt.save = input_name + '_fine' + '.glb'
        print(f'[INFO] save to default output path: {os.path.join(opt.outdir, opt.save)}.')

    return opt