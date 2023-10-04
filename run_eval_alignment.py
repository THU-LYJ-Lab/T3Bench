from PIL import Image
import argparse
import os
import numpy as np
import glob
from copy import deepcopy

import openai
import backoff


openai.api_key = "PLEASE USE YOUR OWN API KEY"
MODEL = "gpt-4"

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
def predict(prompt, temperature):
    message = openai.ChatCompletion.create(
        model = MODEL,
        temperature = temperature,
        messages = [
                # {"role": "system", "content": prompt}
                {"role": "user", "content": prompt}
            ]
    )
    # print(message)
    return message["choices"][0]["message"]["content"]


def process_text(text):
    if text[-1] == '.':
        text = text[:-1]
    if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if text[0] == '\'' and text[-1] == '\'':
        text = text[1:-1]
    if text[-1] == '.':
        text = text[:-1]
    return text


def run(args, temp_path):
    mean_score = 0
    output = ''

    prompt_to_gpt4_ori = '''You are an assessment expert responsible for prompt-prediction pairs. Your task is to score the prediction according to the following requirements:

    1. Evaluate the recall, or how well the prediction covers the information in the prompt. If the prediction contains information that does not appear in the prompt, it should not be considered as bad.
    2. If the prediction contains correct information about color or features in the prompt, you should also consider raising your score.
    3. Assign a score between 1 and 5, with 5 being the highest. Do not provide a complete answer; give the score in the format: 3

    '''

    with open(f'outputs_caption/{args.method}_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'result/alignment', exist_ok=True)
    
    for line in lines:
        line = line.strip()
        prompt, text = line.split(':')
        text = process_text(text)

        prompt_to_gpt4 = prompt_to_gpt4_ori
        prompt_to_gpt4 += 'Prompt: ' + prompt + '\n'
        prompt_to_gpt4 += 'Prediction: ' + text
        print(prompt_to_gpt4)
        res = predict(prompt_to_gpt4, 0)
    
        print(res)
        mean_score += np.round(float(res)) / len(lines)
        output += f'{np.round(float(res)):.0f}\t\t{prompt}\n'

    print("Alignment Score:", mean_score)

    with open(f'result/alignment/{args.method}_{args.group}.txt', 'a+') as f:
        f.write(output)
        
        

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
