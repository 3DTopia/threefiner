export CUDA_VISIBLE_DEVICES=0

# the mesh is already with good initial texture, just refine it using IF2
threefiner if2 --mesh data/car.glb --prompt 'a red car' --outdir logs --save car_fine.glb --text_dir --front_dir='+x'

# the mesh is coarse, using SD for diverse texture generation and IF2 for refinement
threefiner sd --mesh data/chair.ply --prompt 'a swivel chair' --outdir logs --save chair_coarse.glb --text_dir --front_dir='-y'
threefiner if2 --mesh logs/chair_coarse.glb --prompt 'a swivel chair' --outdir logs --save chair_fine.glb --text_dir --front_dir='+z'
