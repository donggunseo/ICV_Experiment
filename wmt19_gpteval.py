import openai
import json
from tqdm import tqdm

openai.api_key = "YOUR_API"

def build_prompt(source, reference, prediction):
    return f"""
You are a professional translation evaluator. Your task is to evaluate how well a predicted German translation matches a reference translation, given the English source.

Use the following evaluation criteria:

1. Accuracy: Does the translation faithfully convey all information from the source text without omissions or distortions?
2. Fluency: Is the German translation grammatically correct and natural-sounding?
3. Context Awareness: Does it appropriately reflect tone, register, and word choice based on the context?
4. Consistency: Is terminology used consistently, especially for technical or repeated terms?
5. No Hallucinations: Are there any additions not present in the source?
6. Formatting: Are lists, punctuation, and special formatting preserved correctly?

Rate the translation on a scale from 1 to 5:
1 = Very poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Respond only in this JSON format:
{{
  "score": <integer from 1 to 5>,
  "justification": "<brief explanation>"
}}

Source (English):
"{source}"

Reference Translation (German):
"{reference}"

Predicted Translation (German):
"{prediction}"
""".strip()


def evaluate_translation_batch(pairs, model="gpt-4o-mini"):
    results = []
    for example in tqdm(pairs):
        source = example["test_query"]
        reference = example["gt"]
        prediction = example["cleaned_prediction"]

        prompt = build_prompt(source, reference, prediction)

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            output = response.choices[0].message.content.strip()
            parsed = json.loads(output)
            results.append({
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "score": parsed["score"],
                "justification": parsed["justification"]
            })

        except Exception as e:
            results.append({
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "score": None,
                "justification": f"Error: {str(e)}"
            })

    return results


if __name__ == "__main__":
    for seed in ['42','41','40']:
        with open(f"./results/wmt19/{seed}/zs_result.json", "r") as f:
            zs_result = json.load(f)
        zs_evaluation = evaluate_translation_batch(zs_result['result'])

        with open(f"./results/wmt19/{seed}/zs_gpteval.json", "w") as f:
            json.dump(zs_evaluation, f)
        
        for config in ['fs', 'task_vector', 'diff_icv_baseline', 'stacked_diff_icv']:
            path = f"./results/wmt19/{seed}/50shots/{config}_result.json"
            with open(path, 'r') as f:
                res = json.load(f)
            eval = evaluate_translation_batch(res['result'])
            with open(f"./results/wmt19/{seed}/50shots/{config}_gpteval.json", "w") as f:
                json.dump(eval, f)
