from torch.utils.data import Dataset
import random
import json

system_prompt = '''You are a helpful, warm, and empathetic AI assistant.
You will be provided with the user's basic profile (Persona) and their current context (Scenario). Your task is to generate a response to the user.'''

user_prompt = '''
**User Persona:** {persona}
**Current scenario:** {scenario}

**User Query:** {query}

**Output Requirement:**
Respond in exactly this output format:

# Analysis
<Analyze the current scenario and user's emotion>

# Response
<Write your response to the user here>
'''

def load_jsonl(data_path: str):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

class EmpathyDataset(Dataset):
    def __init__(self, data_path: str, num_samples: int = None):
        anchor_data = load_jsonl(data_path)
        self.data = []
        for item in anchor_data:
            self.data.append({
                "persona": item['persona']['identity'],
                "scenario": item['scenario'],
                "query": item['dialogue'][0],
            })
        if num_samples and num_samples < len(self.data):
            self.data = random.sample(self.data, num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        prompt = [{"role": "system", "content": system_prompt}]
        prompt.append({"role": "user", "content": user_prompt.format(
            persona=item["persona"],
            scenario=item["scenario"],
            query=item["query"],
        )})
        item['prompt'] = prompt
        return item

    def shuffle(self):
        random.shuffle(self.data)