import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr

import kiui
from kiui.mesh import Mesh
from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.op import safe_normalize, scale_img_hwc, make_divisible, uv_padding
from kiui.cam import orbit_camera, get_perspective

from threefiner.nn import MLP, HashGridEncoder, FrequencyEncoder, TriplaneEncoder
from threefiner.renderer.mesh_renderer import render_mesh

from diso import DiffMC, DiffDMC

class Renderer(nn.Module):
    def __init__(self, opt, device):
        
        super().__init__()

        self.opt = opt
        self.device = device

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        # diffmc
        self.verts = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, self.opt.mc_grid_size, device=device),
                torch.linspace(-1, 1, self.opt.mc_grid_size, device=device),
                torch.linspace(-1, 1, self.opt.mc_grid_size, device=device),
                indexing="ij",
            ), dim=-1,
        ) # [N, N, N, 3]
        self.grid_scale = 1
        self.diffmc = DiffMC(dtype=torch.float32).to(device)
        
        # vert sdf and deform
        self.sdf = nn.Parameter(torch.zeros_like(self.verts[..., 0]))
        self.deform = nn.Parameter(torch.zeros_like(self.verts))
        
        # init diffmc from mesh
        self.mesh = Mesh.load(self.opt.mesh, bound=0.9, front_dir=self.opt.front_dir)

        vertices = self.mesh.v.detach().cpu().numpy()
        triangles = self.mesh.f.detach().cpu().numpy()
        vertices, triangles = clean_mesh(vertices, triangles, min_f=32, min_d=10, remesh=False)
        self.mesh.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        self.mesh.f = torch.from_numpy(triangles).contiguous().int().to(self.device)

        self.grid_scale = self.mesh.v.abs().max() + 1e-1
        self.verts = self.verts * self.grid_scale
        
        try:
            import cubvh
            BVH = cubvh.cuBVH(self.mesh.v, self.mesh.f)
            sdf, _, _ = BVH.signed_distance(self.verts.reshape(-1, 3), return_uvw=False, mode='raystab') # some mesh may not be watertight...
        except:
            from pysdf import SDF
            sdf_func = SDF(self.mesh.v.detach().cpu().numpy(), self.mesh.f.detach().cpu().numpy())
            sdf = sdf_func(self.verts.detach().cpu().numpy().reshape(-1, 3))
            sdf = torch.from_numpy(sdf).to(self.device)
            sdf *= -1

        # OUTER is POSITIVE
        self.sdf.data += sdf.reshape(*self.sdf.data.shape).to(self.sdf.data.dtype)

        # texture
        if self.opt.tex_mode == 'hashgrid':
            self.encoder = HashGridEncoder().to(self.device)
        elif self.opt.tex_mode == 'mlp':
            self.encoder = FrequencyEncoder().to(self.device)
        elif self.opt.tex_mode == 'triplane':
            self.encoder = TriplaneEncoder().to(self.device)
        else:
            raise NotImplementedError(f"unsupported texture mode: {self.opt.tex_mode} for {self.opt.geom_mode}")
        
        self.mlp = MLP(self.encoder.output_dim, 3, 32, 2, bias=True).to(self.device)

        self.v, self.f = None, None # placeholder

        # init hashgrid texture from mesh
        if self.opt.fit_tex:
            self.fit_texture_from_mesh(self.opt.fit_tex_iters)
    
    def render_mesh(self, pose, proj, h, w, ssaa=1, bg_color=1):
        return render_mesh(
            self.glctx, 
            self.mesh.v, self.mesh.f, self.mesh.vt, 
            self.mesh.ft, self.mesh.albedo, 
            self.mesh.vc, self.mesh.vn, self.mesh.fn, 
            pose, proj, h, w, 
            ssaa=ssaa, bg_color=bg_color,
        )

    def fit_texture_from_mesh(self, iters=512):
        # a small training loop...

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.opt.hashgrid_lr},
            {'params': self.mlp.parameters(), 'lr': self.opt.mlp_lr},
        ])

        resolution = 512

        print(f"[INFO] fitting texture...")
        pbar = tqdm.trange(iters)
        for i in pbar:

            ver = np.random.randint(-45, 45)
            hor = np.random.randint(-180, 180)
            
            pose = orbit_camera(ver, hor, self.opt.radius)
            proj = get_perspective(self.opt.fovy)

            image_mesh = self.render_mesh(pose, proj, resolution, resolution)['image']
            image_pred = self.render(pose, proj, resolution, resolution)['image']

            loss = loss_fn(image_pred, image_mesh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"MSE = {loss.item():.6f}")
        
        print(f"[INFO] finished fitting texture!")

    def get_params(self):

        params = [
            {'params': self.encoder.parameters(), 'lr': self.opt.hashgrid_lr},
            {'params': self.mlp.parameters(), 'lr': self.opt.mlp_lr},
        ]

        if not self.opt.fix_geo:
            params.append({'params': self.sdf, 'lr': self.opt.sdf_lr})
            params.append({'params': self.deform, 'lr': self.opt.deform_lr})

        return params

    @torch.no_grad()
    def export_mesh(self, save_path, texture_resolution=2048, padding=16):

        # get v
        sdf = self.sdf
        deform = torch.tanh(self.deform) / 2 # [-0.5, 0.5]

        v, f = self.diffmc(sdf, deform)
        v = (2 * v - 1) * self.grid_scale
        f = f.int()
        self.v, self.f = v, f
        
        vertices = v.detach().cpu().numpy()
        triangles = f.detach().cpu().numpy()

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=self.opt.remesh_size)
        
        # decimation
        if self.opt.decimate_target > 0 and triangles.shape[0] > self.opt.decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, self.opt.decimate_target)
        
        v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.device)
        
        mesh = Mesh(v=v, f=f, albedo=None, device=self.device)
        print(f"[INFO] uv unwrapping...")
        mesh.auto_normal()
        mesh.auto_uv()

        # render uv maps
        h = w = texture_resolution
        uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

        # masked query 
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        
        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)

        if mask.any():
            print(f"[INFO] querying texture...")

            xyzs = xyzs[mask] # [M, 3]

            # batched inference to avoid OOM
            batch = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                batch.append(torch.sigmoid(self.mlp(self.encoder(xyzs[head:tail]))).float())
                head += 640000

            albedo[mask] = torch.cat(batch, dim=0)
        
        albedo = albedo.view(h, w, -1)
        mask = mask.view(h, w)

        print(f"[INFO] uv padding...")
        albedo = uv_padding(albedo, mask, padding)

        mesh.albedo = albedo
        mesh.write(save_path)

    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1):
        
        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        sdf = self.sdf
        deform = torch.tanh(self.deform) / 2 # [-0.5, 0.5]

        v, f = self.diffmc(sdf, deform)
        v = (2 * v - 1) * self.grid_scale
        f = f.int()
        self.v, self.f = v, f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [V, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(0) # important to enable gradients!
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, H, W, 3]
        xyzs = xyzs.view(-1, 3)
        mask = (alpha > 0).view(-1)
        color = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = torch.sigmoid(self.mlp(self.encoder(xyzs[mask], bound=1)))
            color[mask] = masked_albedo.float()
        color = color.view(1, h, w, 3)

        # antialias
        color = dr.antialias(color, rast, v_clip, f).squeeze(0) # [H, W, 3]
        color = alpha * color + (1 - alpha) * bg_color

        # get vn and render normal
        i0, i1, i2 = f[:, 0].long(), f[:, 1].long(), f[:, 2].long()
        v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals)
        
        vn = torch.zeros_like(v)
        vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, f)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        # ssaa
        if ssaa != 1:
            color = scale_img_hwc(color, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = color.clamp(0, 1)
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos

        return results