import os
import tqdm
import random
import imageio
import numpy as np

import torch
import torch.nn.functional as F

GUI_AVAILABLE = True
try:
    import dearpygui.dearpygui as dpg
except Exception as e:
    GUI_AVAILABLE = False

import kiui
from kiui.cam import orbit_camera, OrbitCamera
from kiui.mesh_utils import laplacian_smooth_loss, normal_consistency

from threefiner.opt import Options

class GUI:
    def __init__(self, opt: Options):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        if not GUI_AVAILABLE and opt.gui:
            print(f'[WARN] cannot import dearpygui, assume without --gui')
        self.gui = opt.gui and GUI_AVAILABLE # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        self.save_path = os.path.join(self.opt.outdir, self.opt.save)
        os.makedirs(self.opt.outdir, exist_ok=True)

        # models
        self.device = torch.device("cuda")

        self.guidance = None

        # renderer
        if self.opt.geom_mode == 'mesh':
            from threefiner.renderer.mesh_renderer import Renderer
        elif self.opt.geom_mode == 'diffmc':
            from threefiner.renderer.diffmc_renderer import Renderer
        elif self.opt.geom_mode == 'pbr_mesh':
            from threefiner.renderer.pbr_mesh_renderer import Renderer
        elif self.opt.geom_mode == 'pbr_diffmc':
            from threefiner.renderer.pbr_diffmc_renderer import Renderer
        else:
            raise NotImplementedError(f"unknown geometry mode: {self.opt.geom_mode}")

        self.renderer_class = Renderer
        
        if self.opt.mesh is None:
            self.renderer = None
        else:
            self.renderer = Renderer(opt, self.device).to(self.device)

        # input prompt
        self.prompt = self.opt.prompt
        self.negative_prompt = ""

        if self.opt.positive_prompt is not None:
            self.prompt = self.opt.positive_prompt + ', ' + self.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        
        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.last_seed = seed

    def prepare_train(self):

        assert self.renderer is not None, 'no mesh loaded!'

        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # lazy load guidance model
        if self.guidance is None:
            print(f"[INFO] loading guidance...")
            if self.opt.mode == 'SD':
                from threefiner.guidance.sd_utils import StableDiffusion
                self.guidance = StableDiffusion(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'SD_NFSD':
                from threefiner.guidance.sd_nfsd_utils import StableDiffusion
                self.guidance = StableDiffusion(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'SDCN':
                from threefiner.guidance.sdcn_utils import StableDiffusionControlNet
                self.guidance = StableDiffusionControlNet(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'IF':
                from threefiner.guidance.if_utils import IF
                self.guidance = IF(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'IF2':
                from threefiner.guidance.if2_utils import IF2
                self.guidance = IF2(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'IF2_NFSD':
                from threefiner.guidance.if2_nfsd_utils import IF2
                self.guidance = IF2(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'SD_ISM':
                from threefiner.guidance.sd_ism_utils import StableDiffusion
                self.guidance = StableDiffusion(self.device, vram_O=self.opt.vram_O)
            elif self.opt.mode == 'IF2_ISM':
                from threefiner.guidance.if2_ism_utils import IF2
                self.guidance = IF2(self.device, vram_O=self.opt.vram_O)
            else:
                raise NotImplementedError(f"unknown guidance mode {self.opt.mode}!")
            print(f"[INFO] loaded guidance!")

        # prepare embeddings
        with torch.no_grad():
            self.guidance.get_text_embeds([self.prompt], [self.negative_prompt])

           
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.renderer.train()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            loss = 0

            ### novel view (manual batch)
            images = []
            poses = []
            normals = []
            ori_images = []
            vers, hors, radii = [], [], []
            
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(-60, 30)
                hor = np.random.randint(-180, 180)
                radius = np.random.uniform() - 0.5 # [-0.5, 0.5]
                pose = orbit_camera(ver, hor, self.opt.radius + radius)

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)
                poses.append(pose)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.cam.perspective, self.opt.render_resolution, self.opt.render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # mix_normal
                if not self.opt.fix_geo and self.opt.mix_normal:
                    normal = out['normal']
                    normal = normal.permute(2,0,1).contiguous().unsqueeze(0)
                    normals.append(normal)

                # IF SR model requires the original rendering
                if self.opt.mode in ['IF2', 'IF2_NFSD', 'IF2_ISM', 'SDCN']:
                    out_mesh = self.renderer.render_mesh(pose, self.cam.perspective, self.opt.render_resolution, self.opt.render_resolution, ssaa=1)
                    ori_image = out_mesh["image"] # [H, W, 3] in [0, 1]
                    ori_image = ori_image.permute(2,0,1).contiguous().unsqueeze(0)
                    ori_images.append(ori_image)
                    # ori_images.append(image.clone())

            # guidance loss
            guidance_input = {'pred_rgb': torch.cat(images, dim=0)}

            if not self.opt.fix_geo and self.opt.mix_normal:
                if random.random() > 0.5:
                    ratio = random.random()
                    guidance_input['pred_rgb'] = guidance_input['pred_rgb'] * ratio + torch.cat(normals, dim=0) * (1 - ratio)
            
            # guidance_input['step_ratio'] = step_ratio
            if self.opt.mode in ['IF2', 'IF2_NFSD', 'IF2_ISM']:
                guidance_input['ori_rgb'] = torch.cat(ori_images, dim=0)
            if self.opt.mode == 'SDCN':
                guidance_input['control_images'] = {'tile': torch.cat(ori_images, dim=0)}
            if self.opt.text_dir:
                guidance_input['vers'] = vers
                guidance_input['hors'] = hors
            
            loss = loss + self.opt.lambda_sd * self.guidance.train_step(**guidance_input)
            
            # geom regularizations
            if self.opt.geom_mode in ['diffmc', 'pbr_diffmc', 'mesh', 'pbr_mesh'] and not self.opt.fix_geo:
                if self.opt.lambda_lap > 0:
                    lap_loss = laplacian_smooth_loss(self.renderer.v, self.renderer.f)
                    loss = loss + self.opt.lambda_lap * lap_loss
                if self.opt.lambda_normal > 0:
                    normal_loss = normal_consistency(self.renderer.v, self.renderer.f)
                    loss = loss + self.opt.lambda_normal * normal_loss
                if self.opt.geom_mode in ['mesh', 'pbr_mesh'] and self.opt.lambda_offsets > 0:
                    offset_loss = (self.renderer.v_offsets ** 2).sum(-1).mean()
                    loss = loss + self.opt.lambda_offsets * offset_loss
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # for mesh geom_mode: peoriodically remesh
            if self.opt.geom_mode in ['mesh', 'pbr_mesh'] and not self.opt.fix_geo:
                if self.step > 0 and self.step % self.opt.remesh_interval == 0:
                    self.renderer.remesh()
                    # reset optimizer
                    self.optimizer = torch.optim.Adam(self.renderer.get_params())

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image
            self.renderer.eval()

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    def save_model(self, save_path=None):

        if save_path is None:
            save_path = self.save_path

        # export video
        if save_path.endswith(".mp4"):
            images = []
            elevation = 0
            azimuth = np.arange(0, 360, 3, dtype=np.int32) # front-->back-->front
            for azi in tqdm.tqdm(azimuth):
                pose = orbit_camera(elevation, azi, self.opt.radius)
                out = self.renderer.render(pose, self.cam.perspective, self.opt.render_resolution, self.opt.render_resolution, ssaa=1)    
                image = (out["image"].detach().cpu().numpy() * 255).astype(np.uint8)
                images.append(image)
            images = np.stack(images, axis=0)
            # ~4 seconds, 120 frames at 30 fps
            imageio.mimwrite(save_path, images, fps=30, quality=8, macro_block_size=1)
        # export mesh
        else:
            self.renderer.export_mesh(save_path, texture_resolution=self.opt.texture_resolution)

        print(f"[INFO] save model to {save_path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Threefiner",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        # save
        self.save_model()