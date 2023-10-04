import os
import glob
import argparse


def run(args):
    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'outputs_mesh_t3/{args.method}_{args.group}', exist_ok=True)
    os.chdir('third_party/threestudio')
    
    for prompt in lines:
        prompt = prompt.strip()

        if args.method == 'dreamfusion':
            d = glob.glob(os.path.join(f'../../outputs_t3/dreamfusion_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue
            print(f'python launch.py --config configs/dreamfusion-sd.yaml --export \
                  --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/dreamfusion-sd.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')

            d = sorted(glob.glob(os.path.join('outputs_mesh/dreamfusion-sd', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        elif args.method == 'sjc':
            d = glob.glob(os.path.join(f'../../outputs_t3/sjc_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue
            print(f'python launch.py --config configs/sjc.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_threshold=5. \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/sjc.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_threshold=5. \
                    exp_root_dir=outputs_mesh')

            d = sorted(glob.glob(os.path.join('outputs_mesh/sjc', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        elif args.method == 'magic3d':
            d = glob.glob(os.path.join(f'../../outputs_t3/magic3d_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue
            print(f'python launch.py --config configs/magic3d-refine-sd.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry_convert_from={d} \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/magic3d-refine-sd.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry_convert_from={d} \
                    exp_root_dir=outputs_mesh')
            
            d = sorted(glob.glob(os.path.join('outputs_mesh/magic3d-refine-sd', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        elif args.method == 'latentnerf':
            d = glob.glob(os.path.join(f'../../outputs_t3/latentnerf_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue

            print(f'python launch.py --config configs/latentnerf-refine.yaml --export \
                    --gpu {args.gpu} resume={d} system.weights={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/latentnerf-refine.yaml --export \
                    --gpu {args.gpu} resume={d} system.weights={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')
            
            d = sorted(glob.glob(os.path.join('outputs_mesh/latentnerf-refine', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        elif args.method == 'fantasia3d':
            d = glob.glob(os.path.join(f'../../outputs_t3/magic3d_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue
            print(f'python launch.py --config configs/magic3d-refine-sd.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry_convert_from={d} \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/magic3d-refine-sd.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry_convert_from={d} \
                    exp_root_dir=outputs_mesh')
            
            d = sorted(glob.glob(os.path.join('outputs_mesh/magic3d-refine-sd', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        elif args.method == 'prolificdreamer':
            d = glob.glob(os.path.join(f'../../outputs_t3/prolificdreamer_{args.group}', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt'))
            try:
                d = d[-1].replace('\'', '\\\'')
            except:
                continue
            print(f'python launch.py --config configs/prolificdreamer.yaml --export \
                  --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')
            os.system(f'python launch.py --config configs/prolificdreamer.yaml --export \
                    --gpu {args.gpu} resume={d} system.prompt_processor.prompt="{prompt}" \
                    system.exporter_type=mesh-exporter system.exporter.context_type=cuda \
                    system.geometry.isosurface_method=mc-cpu \
                    system.geometry.isosurface_resolution=256 \
                    exp_root_dir=outputs_mesh')

            d = sorted(glob.glob(os.path.join('outputs_mesh/prolificdreamer', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_mesh_t3/{args.method}_{args.group}')

        else:
            raise NotImplementedError
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='single', choices=['single', 'surr', 'multi'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['latentnerf', 'magic3d', 'fantasia3d', 'dreamfusion', 'sjc', 'prolificdreamer'])
    args = parser.parse_args()
    
    run(args)
