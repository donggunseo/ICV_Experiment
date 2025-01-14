from datasets import load_dataset, ClassLabel
import argparse
import os
import re
import json

def split_dataset(dataset, max_size):
    if len(dataset)>max_size:
        dataset = dataset.train_test_split(test_size=max_size, stratify_by_column="label")['test']
    return dataset

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string

def preprocess_data(dataset, id2label):
    text = dataset['text']
    label = dataset['label']
    text = [general_detokenize(t).strip() for t in text]
    label = [id2label[l] for l in label]
    return text, label

def convert_to_dict(text,label):
    d = []
    for t, l in zip(text, label):
        d.append({"input":t, "output":l})
    return d
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--train_max_limit', help="Maximum train dataset size ", type=int, required=False, default=10000)
    parser.add_argument('--val_max_limit', help="Maximum valiation dataset size ", type=int, required=False, default=1000)
    parser.add_argument('--test_max_limit', help="Maximum test dataset size ", type=int, required=False, default=500)
    parser.add_argument('--data_path_root', help='File save path of dataset', type=str, required=False, default='./dataset_files')
   

    args = parser.parse_args()

    dataset_name = args.dataset_name
    train_max_limit = args.train_max_limit
    val_max_limit = args.val_max_limit
    test_max_limit = args.test_max_limit
    data_path_root = args.data_path_root


    print(f"Preprocess {dataset_name} dataset")
    
    if not os.path.exists(os.path.join(data_path_root, dataset_name)):
        os.makedirs(os.path.join(data_path_root, dataset_name), exist_ok=True)
    
    if dataset_name == "sst2": ##train 67349 validation 872 test 1821(X)
        raw_data = load_dataset('glue', 'sst2')
        raw_data = raw_data.rename_column('sentence', 'text')
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\nvalidation:{len(raw_data['validation'])}")
        d = raw_data['train'].train_test_split(test_size=1000, stratify_by_column="label")
        raw_data['train'] = d['train']
        raw_data['test'] = raw_data['validation']
        raw_data['validation'] = d['test']
        id2label = {0: 'negative', 1: 'positive'}
    elif dataset_name == "cr": ## 전자제품 리뷰 감정 분류 데이터셋 train 3394 test 376
        raw_data = load_dataset('SetFit/CR')
        class_label = ClassLabel(names = [0,1])
        raw_data = raw_data.cast_column('label', class_label)
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\ntest:{len(raw_data['test'])}")
        d = raw_data['train'].train_test_split(test_size=500, stratify_by_column="label")
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        id2label = {0: 'negative', 1: 'positive'}
    elif dataset_name == "trec": ##train 5452 test 500
        raw_data = load_dataset('CogComp/trec', trust_remote_code=True)
        raw_data = raw_data.rename_column('coarse_label', 'label')
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\ntest:{len(raw_data['test'])}")
        d = raw_data['train'].train_test_split(test_size=1000, stratify_by_column="label")
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        id2label = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
    elif dataset_name == "agnews": ##train 120000 test 7600
        raw_data = load_dataset("fancyzhx/ag_news")
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\ntest:{len(raw_data['test'])}")
        d = raw_data['train'].train_test_split(test_size=1000, stratify_by_column="label")
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        id2label = {0: "world", 1: "sports", 2: "buisness", 3: "sci/tech"}
    elif dataset_name == "mr": ## test 10662 
        raw_data = load_dataset('mattymchen/mr')
        class_label = ClassLabel(names = [0,1])
        raw_data = raw_data.cast_column('label', class_label)
        print(f"Original dataset statistics\ntest:{len(raw_data['test'])}")
        d = raw_data['test'].train_test_split(test_size=500, stratify_by_column="label")
        d2 = d['train'].train_test_split(test_size=1000, stratify_by_column="label")
        raw_data['train'] = d2['train']
        raw_data['validation'] = d2['test']
        raw_data['test'] = d['test']
        id2label = {0: 'negative', 1: 'positive'}

    
    train = split_dataset(raw_data['train'], train_max_limit)
    val = split_dataset(raw_data['validation'], val_max_limit)
    test = split_dataset(raw_data['test'], test_max_limit)

    print(f"preprocessed dataset statistics\ntrain:{len(train)}\nvalidation:{len(val)}\ntest:{len(test)}")

    train_text, train_label = preprocess_data(train, id2label)
    val_text, val_label = preprocess_data(val, id2label)
    test_text, test_label = preprocess_data(test, id2label)
    
    train_dataset = convert_to_dict(train_text, train_label)
    val_dataset = convert_to_dict(val_text, val_label)
    test_dataset = convert_to_dict(test_text, test_label)

    with open(os.path.join(data_path_root, dataset_name, "train.json"), "w") as f:
        json.dump(train_dataset,f)
    with open(os.path.join(data_path_root, dataset_name, "val.json"), "w") as f:
        json.dump(val_dataset,f)
    with open(os.path.join(data_path_root, dataset_name, "test.json"), "w") as f:
        json.dump(test_dataset,f)







    
    