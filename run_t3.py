import os
import glob
import argparse


def run(args):
    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'outputs_t3/{args.method}_{args.group}', exist_ok=True)
    os.chdir('third_party/threestudio')
    
    for prompt in lines:
        prompt = prompt.strip()

        if args.method == 'dreamfusion':
            print(f'python launch.py --config configs/dreamfusion-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/dreamfusion-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/dreamfusion-sd', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        elif args.method == 'sjc':
            print(f'python launch.py --config configs/sjc.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/sjc.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/sjc', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        elif args.method == 'magic3d':
            print(f'python launch.py --config configs/magic3d-coarse-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/magic3d-coarse-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/magic3d-coarse-sd', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt')))[-1].replace('\'', '\\\'')
            print(f'python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}" system.geometry_convert_from={d} system.geometry_convert_override.isosurface_threshold=5.')
            os.system(f'python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}" system.geometry_convert_from={d} system.geometry_convert_override.isosurface_threshold=5.')

            d = sorted(glob.glob(os.path.join('outputs/magic3d-refine-sd', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        elif args.method == 'latentnerf':
            print(f'python launch.py --config configs/latentnerf.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/latentnerf.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/latentnerf', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt')))[-1].replace('\'', '\\\'')
            print(f'python launch.py --config configs/latentnerf-refine.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}" system.weights={d}')
            os.system(f'python launch.py --config configs/latentnerf-refine.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}" system.weights={d}')

            d = sorted(glob.glob(os.path.join('outputs/latentnerf-refine', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        elif args.method == 'fantasia3d':
            print(f'python launch.py --config configs/fantasia3d.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/fantasia3d.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/fantasia3d', prompt.replace(' ', '_') + '*', 'ckpts', 'last.ckpt')))[-1].replace('\'', '\\\'')
            print(f'python launch.py --config configs/fantasia3d-texture.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"  system.geometry_convert_from={d} system.renderer.context_type=cuda')
            os.system(f'python launch.py --config configs/fantasia3d-texture.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"  system.geometry_convert_from={d} system.renderer.context_type=cuda')

            d = sorted(glob.glob(os.path.join('outputs/fantasia3d-texture', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        elif args.method == 'prolificdreamer':
            print(f'python launch.py --config configs/prolificdreamer.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')
            os.system(f'python launch.py --config configs/prolificdreamer.yaml --train --gpu {args.gpu} system.prompt_processor.prompt="{prompt}"')

            d = sorted(glob.glob(os.path.join('outputs/prolificdreamer', prompt.replace(' ', '_') + '*')))[-1].replace('\'', '\\\'')
            os.system(f'mv {d} ../../outputs_t3/{args.method}_{args.group}')

        else:
            raise NotImplementedError
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='single', choices=['single', 'surr', 'multi'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['latentnerf', 'magic3d', 'fantasia3d', 'dreamfusion', 'sjc', 'prolificdreamer'])
    args = parser.parse_args()
    
    run(args)
