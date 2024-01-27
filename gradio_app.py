import os
import tyro
import tqdm
import torch
import gradio as gr

import kiui

from threefiner.opt import config_defaults, config_doc, check_options
from threefiner.gui import GUI

GRADIO_SAVE_PATH_MESH = 'gradio_output.glb'
GRADIO_SAVE_PATH_VIDEO = 'gradio_output.mp4'

opt = tyro.cli(tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc))

# hacks for not loading mesh at initialization
opt.save = GRADIO_SAVE_PATH_MESH
opt.prompt = ''
opt.text_dir = True
opt.front_dir = '+z'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gui = GUI(opt)

# process function
def process(input_model, input_text, input_dir, iters):

    # set front facing direction (map from gradio model3D's mysterious coordinate system to OpenGL...)
    opt.text_dir = True
    if input_dir == 'front':
        opt.front_dir = '-z'
    elif input_dir == 'back':
        opt.front_dir = '+z'
    elif input_dir == 'left':
        opt.front_dir = '+x'
    elif input_dir == 'right':
        opt.front_dir = '-x'
    elif input_dir == 'up':
        opt.front_dir = '+y'
    elif input_dir == 'down':
        opt.front_dir = '-y'
    else:
        # turn off text_dir
        opt.text_dir = False
        opt.front_dir = '+z'
    
    # set mesh path
    opt.mesh = input_model

    # load mesh!
    gui.renderer = gui.renderer_class(opt, device).to(device)

    # set prompt
    gui.prompt = opt.positive_prompt + ', ' + input_text

    # train
    gui.prepare_train() # update optimizer and prompt embeddings
    for i in tqdm.trange(iters):
        gui.train_step()

    # save mesh & video
    gui.save_model(GRADIO_SAVE_PATH_MESH)
    gui.save_model(GRADIO_SAVE_PATH_VIDEO)
    
    # return 3d model & video
    return GRADIO_SAVE_PATH_MESH, GRADIO_SAVE_PATH_VIDEO

# gradio UI
block = gr.Blocks().queue()
with block:
    gr.Markdown("""
    ## Threefiner: Text-guided mesh refinement.
    """)

    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            input_model = gr.Model3D(label="input mesh")
            input_text = gr.Text(label="prompt")
            input_dir = gr.Radio(['front', 'back', 'left', 'right', 'up', 'down'], label="front-facing direction")
            iters = gr.Slider(minimum=100, maximum=1000, step=100, value=400, label="training iterations")
            button_gen = gr.Button("Refine!")
        
        with gr.Column(scale=1):
            output_model = gr.Model3D(label="output mesh")
            output_video = gr.Video(label="output video")

        button_gen.click(process, inputs=[input_model, input_text, input_dir, iters], outputs=[output_model, output_video])
    
block.launch(server_name="0.0.0.0", share=True)