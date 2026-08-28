import string
import re
import json
from collections import Counter

def variable_ngram_repetition(text, min_n=5, max_n=5, threshold=10):
    tokens = text.lower().split()
    for n in range(min_n, max_n + 1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        for ngram, count in counts.items():
            if count >= threshold:
                return True, ngram, count
    return False, None, 0

def contains_chinese(text):
    """
    Returns True if the string contains any Chinese characters.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def normalize_text(text: str) -> str:
    """
    Normalize the text by removing punctuation and converting to lowercase.
    """
    #text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('**', '')
    ### newly appended to address beaverTails' category with comma
    text = text.replace(', ', ',')
    return text.lower().strip().strip('\'"')

def extract_safety_content(text: str) -> str:
    """
    Extracts answers in \\safety{}.
    """
    
    depth = 0
    start_pos = text.rfind(r"\safety{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\safety{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"


def extract_category_content(text: str) -> str:
    """
    Extracts answers in \\category{}.
    """
    
    depth = 0
    start_pos = text.rfind(r"\category{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\category{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def safety_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\safety\{.*\}.*\\category\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    is_repeating, ngram, count = variable_ngram_repetition(predict_str)
    if is_repeating:
        print(f"Detected repetition: {' '.join(ngram)} (repeated {count} times)")
        return 0.0
    if contains_chinese(predict_str):
        print(f'Detected chinese char: \n{predict_str}')
        return 0.0
    return 1.0 if format_match else 0.0


def safety_acc_reward(predict_str: str, ground_truth: str) -> float:
    #answer str should be in the json format


    answer = predict_str


    if answer.lower() == "unsafe" and ground_truth.lower() == "unsafe":
        return 1.0

    if answer.lower() == "safe" and ground_truth.lower() == "safe":
        return 1.0
    
    return 0.0

def category_acc_reward(predict_str: str, ground_truth: str) -> float:
    #answer str should be in the json format
    answer = predict_str.strip()
    ground_truth = ground_truth.strip()
    if answer.lower() == ground_truth.lower():
        return 1.0
    return 0.0
    
def checklist_acc_reward(thinking_str: str, category_label: str, ground_truth: str, category_dict: dict) -> float:
    key_len = len(category_dict.keys())
    reward = 0.0
    if thinking_str is None or thinking_str == "":
        return 0.0
    thinking_str = normalize_text(thinking_str)
    thinking_str = thinking_str.replace('_', ' ')
    if category_label == "not applicable" and ground_truth.lower() == "safe":
        return 1.0
    for key in category_dict.keys():
        key_str = key.replace('_', ' ')
        if key_str in thinking_str:
            reward += 1
    category_label = category_label.replace('_', ' ')
    if category_label in thinking_str:
        reward += key_len
    reward = reward / (2 * key_len)
    return reward

def get_thinking_str(solution_str: str) -> str:
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = re.search(pattern, solution_str)
    if match:
        return match.group(1).strip()
    else:
        return ""

def compute_score(solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compare the extracted answer with ground truth.
    Returns 1.0 for correct answer, 0.0 for incorrect.
    
    Args:
        solution_str: The complete solution/response string
        ground_truth: The expected answer ('safe' or 'unsafe')
        extra_info: Additional information (not used in this implementation)
    """

    format = safety_format_reward(solution_str)
    # extract the box content
    safety_accuracy = 0.0
    category_accuracy = 0.0
    checklist_accuracy = 0.0
    category_label = extra_info['category']
    category_dict_str = extra_info['category_dict']
    category_dict = json.loads(category_dict_str)
    '''
    predict_str inside the boxed
    {
    "safety": "safe" or "unsafe",
    "category": category listed in the categories above if unsafe, otherwise "not applicable"
    }
    '''
    thinking_str = ""
    scale = 1.0
    safety = extract_safety_content(solution_str)
    category = extract_category_content(solution_str)
    if safety == "None" or category == "None":
        print(f'Error extracting safety or category from solution_str: {solution_str}')
        print(f'safety: {safety}, category: {category}')
    safety = normalize_text(safety)
    category = normalize_text(category)
    try:
        thinking_str = get_thinking_str(solution_str)
        ### overlong response penalty
        if thinking_str != '':
            if len(solution_str.split())  >= 2.5 * len(thinking_str.split()):
                print('overlong response')
                print(f"solution_str: {solution_str}")
                scale = scale * 0.5
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"solution_str: {solution_str}")


    safety_accuracy = safety_acc_reward(safety, ground_truth)
    category_accuracy = category_acc_reward(category, category_label)
    #checklist_accuracy = checklist_acc_reward(thinking_str, category_label, ground_truth, category_dict)
    score = 0.55 * format * safety_accuracy + 0.45 * format * category_accuracy
    score = scale * score
    ### length penalty for long response
    #if len()
    return {
        "score": score,
        "format": format,
        "safety_accuracy": safety_accuracy,
        "category_accuracy": category_accuracy,
        "checklist_accuracy": checklist_accuracy, 
        "pred": safety_accuracy
    }
    