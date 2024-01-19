export CUDA_VISIBLE_DEVICES=1

# geom_mode
threefiner if2 --geom_mode diffmc --save car_diffmc.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'
threefiner if2 --geom_mode mesh --save car_mesh.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'
threefiner if2 --geom_mode pbr_diffmc --save car_pbr_diffmc.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'
threefiner if2 --geom_mode pbr_mesh --save car_pbr_mesh.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'

# tex_mode
threefiner if2 --tex_mode mlp --save car_mlp.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'
threefiner if2 --tex_mode triplane --save car_triplane.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'

# guidance mode
threefiner sd --save car_SD.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'
threefiner if --save car_IF.glb --mesh data/car.glb --prompt 'a red car' --outdir logs_test --text_dir --front_dir='+x'