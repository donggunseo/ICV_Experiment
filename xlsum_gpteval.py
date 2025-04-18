import openai
import json
from tqdm import tqdm

openai.api_key = "YOURAPI"

def build_prompt(article, reference, prediction):
    return f"""
You are an evaluation assistant. Evaluate how well the predicted summary represents the original article, using the given reference summary as a guide. Assess the predicted summary on the following criteria, and score each from 1 (very poor) to 5 (excellent), followed by a brief justification for each.

### Evaluation Criteria:
1. **Conciseness**: Is the predicted summary significantly shorter than the article, and is its length appropriate compared to the reference summary? The prediction should ideally match the level of brevity shown in the reference summary.
2. **Factual Accuracy**: Does the predicted summary accurately reflect the key facts from the article without omitting critical points or adding incorrect information?
3. **Coherence and Readability**: Is the summary well-structured, grammatically correct, and easy to follow?
4. **Neutral and Objective Tone**: Is the summary written in a neutral, factual tone without personal opinions, exaggerations, or biases?
5. **Language Matching**: Does the summary use the same language as the article (e.g., English)?

---

### Inputs:

**Original Article**:
{article}

**Reference Summary**:
{reference}

**Predicted Summary**:
{prediction}

---

Rate the Summary on a scale from 1 to 5:
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
""".strip()


def evaluate_summarization_batch(pairs, model="gpt-4o-mini"):
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
        with open(f"./results/xlsum/{seed}/zs_result.json", "r") as f:
            zs_result = json.load(f)
        zs_evaluation = evaluate_summarization_batch(zs_result['result'])

        with open(f"./results/xlsum/{seed}/zs_gpteval.json", "w") as f:
            json.dump(zs_evaluation, f)
        
        for config in ['fs', 'stacked_diff_icv']:
            path = f"./results/xlsum/{seed}/10shots/{config}_result.json"
            with open(path, 'r') as f:
                res = json.load(f)
            eval = evaluate_summarization_batch(res['result'])
            with open(f"./results/xlsum/{seed}/10shots/{config}_gpteval.json", "w") as f:
                json.dump(eval, f)
