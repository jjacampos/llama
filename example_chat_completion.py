# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import pandas as pd
import fire
import re
from llama import Llama
from tqdm import tqdm

def load_dialogs(path: str):
    data = pd.read_json(path, lines=True)
    dialogs = []
    prompts = []
    for elem in data.iterrows():
        cur_dial = []
        prompt = elem[1].prompt
        if "User:" not in prompt:
            cur_dial.append({"role": "user", "content": prompt})
        else:
            user_turns = re.findall("User:(.*?)\nChatbot:", prompt, re.DOTALL)
            chatbot_turns = re.findall("Chatbot:(.*?)\nUser:", prompt, re.DOTALL)
            for user_turn, chatbot_turn in zip(user_turns, chatbot_turns):
                cur_dial.append({"role": "user", "content": user_turn.strip()})
                cur_dial.append({"role": "assistant", "content": chatbot_turn.strip()})
            cur_dial.append({"role": "user", "content": user_turns[-1]})
            
        prompts.append(prompt)
        dialogs.append(cur_dial)
    return dialogs, prompts
        

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompts_path: str,        
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    subset: int = 1000000,
    max_gen_len: Optional[int] = None,
):


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    
    '''
    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    '''
    
    dialogs, prompts = load_dialogs(prompts_path)
    dialogs, prompts = dialogs[:subset], prompts[:subset]
    results = []
    for i in tqdm(range(0, len(dialogs), max_batch_size)):
        results += generator.chat_completion(
            dialogs[i: i + max_batch_size],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    new_data = []
    for prompt, result in zip(prompts, results):
        new_data.append({"prompt": prompt, "completion": result['generation']['content'].strip()})
        
    df = pd.DataFrame(new_data)
    with open(prompts_path + f"_{ckpt_dir.split('/')[-1]}" +  '_out', "w") as f:
        f.write(df.to_json(orient='records', lines=True, force_ascii=False))

if __name__ == "__main__":
    fire.Fire(main)
