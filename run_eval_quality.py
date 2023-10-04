import os
import glob
import argparse

import ImageReward as RM
import trimesh
from PIL import Image


def run(args, temp_path):
    Radius = 2.2
    model = RM.load("ImageReward-v1.0")
    icosphere = trimesh.creation.icosphere(subdivisions=2, radius=Radius)

    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'result/quality', exist_ok=True)
    mean_score = 0
    
    for prompt in lines:
        prompt = prompt.strip()
        try:
            obj_path = glob.glob(f'outputs_mesh_t3/{args.method}_{args.group}/{prompt.replace(" ", "_")}*/save/it*-export/model.obj')[-1].replace("\'", "\\\'")
        except IndexError:
            obj_path = 'outputs_mesh_t3/FALSE_PATH'
        
        os.system(f'python render/meshrender.py --path {obj_path} --name {temp_path}')

        scores = { i: -114514 for i in range(len(icosphere.vertices)) }

        for idx in range(len(icosphere.vertices)):
            for j in range(5):
                img_path = f'{idx:03d}_{j}.png'
                # convert color to PIL image
                color = Image.open(os.path.join(temp_path, img_path))
                reward = model.score(prompt, color)
                scores[idx] = max(scores[idx], reward)

        # convolute scores on the icosphere for 3 times
        for _ in range(3):
            new_scores = {}
            for idx, v in enumerate(icosphere.vertices):
                new_scores[idx] = scores[idx]
                for n in icosphere.vertex_neighbors[idx]:
                    new_scores[idx] += scores[n]
                new_scores[idx] /= (len(icosphere.vertex_neighbors[idx]) + 1)
            scores = new_scores

        for idx in sorted(scores, key=lambda x: scores[x], reverse=True)[:1]:
            now_score = scores[idx] * 20 + 50
            mean_score += now_score / len(lines)
            print(now_score)

            with open(f'result/quality/{args.method}_{args.group}.txt', 'a+') as f:
                f.write(f'{now_score:.1f}\t\t{prompt}\n')

    print("Quality score:", mean_score)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='single', choices=['single', 'surr', 'multi'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['latentnerf', 'magic3d', 'fantasia3d', 'dreamfusion', 'sjc', 'prolificdreamer'])
    args = parser.parse_args()

    i = 0
    while True:
        try: 
            temp_path = f'temp/temp_{i}'
            os.makedirs(temp_path)
            break
        except:
            i += 1
    
    run(args, temp_path)
    os.system(f'rm -r {temp_path}')
