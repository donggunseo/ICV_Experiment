from datasets import load_dataset, ClassLabel, DatasetDict
import argparse
import os
import re
import json
import random
from tqdm import tqdm

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string

def preprocess_data(dataset, id2label=None):
    text = dataset['text']
    text = [general_detokenize(t).strip() for t in text]
    if id2label is not None:
        label = dataset['label']
        label = [id2label[l] for l in label]
    else:
        label = dataset['label_text']
    return text, label

def convert_to_dict(text,label):
    d = []
    for t, l in zip(text, label):
        d.append({"input":t, "output":l})
    return d
    
def preprocess_data_wmt19(dataset):
    code_list = {'en':[], 'de':[]}
    item = list(dataset['translation'])

    for c in code_list:
        text = [i[c] for i in item]
        code_list[c].extend(text)
    return code_list['en'], code_list['de']

def preprocess_data_gsm8k(dataset):
    text = dataset['question']
    text = [general_detokenize(t).strip() for t in text]
    answer = dataset['answer']
    answer = [general_detokenize(t).strip() for t in answer]
    return text, answer

def preprocess_data_math(dataset):
    text = dataset['problem']
    text = [general_detokenize(t).strip() for t in text]
    answer = dataset['solution']
    answer = [general_detokenize(t).strip() for t in answer]
    return text, answer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--train_max_limit', help="Maximum train dataset size ", type=int, required=False, default=10000)
    parser.add_argument('--val_max_limit', help="Maximum valiation dataset size ", type=int, required=False, default=100)
    parser.add_argument('--test_max_limit', help="Maximum test dataset size ", type=int, required=False, default=100)
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
    
    if dataset_name == "trec": ##train 5452 test 500
        raw_data = load_dataset('CogComp/trec', trust_remote_code=True)
        raw_data = raw_data.rename_column('fine_label', 'label')
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\ntest:{len(raw_data['test'])}")
        d = raw_data['train'].train_test_split(test_size=500, stratify_by_column="label")
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        id2label = {
            0: "abbreviation:abbreviation",
            1: "abbreviation:expression",
            2: "entity:animal",
            3: "entity:body",
            4: "entity:color",
            5: "entity:creative work",
            6: "entity:currency",
            7: "entity:disease and medicine",
            8: "entity:event",
            9: "entity:food",
            10: "entity:instrument",
            11: "entity:language",
            12: "entity:letter",
            13: "entity:other",
            14: "entity:plant",
            15: "entity:product",
            16: "entity:religion",
            17: "entity:sport",
            18: "entity:substance",
            19: "entity:symbol",
            20: "entity:technique and method",
            21: "entity:equivalent term",
            22: "entity:vehicle",
            23: "entity:special word",
            24: "description:definition",
            25: "description:description",
            26: "description:manner",
            27: "description:reason",
            28: "human:group",
            29: "human:individual",
            30: "human:title",
            31: "human:description",
            32: "location:city",
            33: "location:country",
            34: "location:mountain",
            35: "location:other",
            36: "location:state",
            37: "numeric:code",
            38: "numeric:count",
            39: "numeric:date",
            40: "numeric:distance",
            41: "numeric:money",
            42: "numeric:order",
            43: "numeric:other",
            44: "numeric:period",
            45: "numeric:percentage",
            46: "numeric:speed",
            47: "numeric:temperature",
            48: "numeric:size and volume",
            49: "numeric:weight"
        }
    elif dataset_name == "banking77": ## train 10003 test 3080
        raw_data = load_dataset('mteb/banking77')
        class_label = ClassLabel(names = list(range(77)))
        raw_data = raw_data.cast_column('label', class_label)
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\ntest:{len(raw_data['test'])}")
        d = raw_data['train'].train_test_split(test_size=1000, stratify_by_column="label")
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        id2label=None
    elif dataset_name == "clinc150": ## train 15000 val 3000 test 4500 oos_test 1200
        raw_data = load_dataset('contemmcm/clinc150')
        raw_data = raw_data.rename_column('intent', 'label')
        train = raw_data.filter(lambda x: x['split']=="train")['complete']
        val = raw_data.filter(lambda x: x['split']=="val")['complete']
        test = raw_data.filter(lambda x: x['split']=="test")['complete']
        raw_data = DatasetDict({
            'train': train,
            'validation': val,
            'test': test
        })
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\nval:{len(raw_data['validation'])}\ntest:{len(raw_data['test'])}")
        id2label={
            1: "banking:freeze_account",
            2: "banking:routing",
            3: "banking:pin_change",
            4: "banking:bill_due",
            5: "banking:pay_bill",
            6: "banking:account_blocked",
            7: "banking:interest_rate",
            8: "banking:min_payment",
            9: "banking:bill_balance",
            10: "banking:transfer",
            11: "banking:order_checks",
            12: "banking:balance",
            13: "banking:spending_history",
            14: "banking:transactions",
            15: "banking:report_fraud",
            16: "credit_cards:replacement_card_duration",
            17: "credit_cards:expiration_date",
            18: "credit_cards:damaged_card",
            19: "credit_cards:improve_credit_score",
            20: "credit_cards:report_lost_card",
            21: "credit_cards:card_declined",
            22: "credit_cards:credit_limit_change",
            23: "credit_cards:apr",
            24: "credit_cards:redeem_rewards",
            25: "credit_cards:credit_limit",
            26: "credit_cards:rewards_balance",
            27: "credit_cards:application_status",
            28: "credit_cards:credit_score",
            29: "credit_cards:new_card",
            30: "credit_cards:international_fees",
            31: "kitchen_and_dining:food_last",
            32: "kitchen_and_dining:confirm_reservation",
            33: "kitchen_and_dining:how_busy",
            34: "kitchen_and_dining:ingredients_list",
            35: "kitchen_and_dining:calories",
            36: "kitchen_and_dining:nutrition_info",
            37: "kitchen_and_dining:recipe",
            38: "kitchen_and_dining:restaurant_reviews",
            39: "kitchen_and_dining:restaurant_reservation",
            40: "kitchen_and_dining:meal_suggestion",
            41: "kitchen_and_dining:restaurant_suggestion",
            42: "kitchen_and_dining:cancel_reservation",
            43: "kitchen_and_dining:ingredient_substitution",
            44: "kitchen_and_dining:cook_time",
            45: "kitchen_and_dining:accept_reservations",
            46: "home:what_song",
            47: "home:play_music",
            48: "home:todo_list_update",
            49: "home:reminder",
            50: "home:reminder_update",
            51: "home:calendar_update",
            52: "home:order_status",
            53: "home:update_playlist",
            54: "home:shopping_list",
            55: "home:calendar",
            56: "home:next_song",
            57: "home:order",
            58: "home:todo_list",
            59: "home:shopping_list_update",
            60: "home:smart_home",
            61: "auto_and_commute:current_location",
            62: "auto_and_commute:oil_change_when",
            63: "auto_and_commute:oil_change_how",
            64: "auto_and_commute:uber",
            65: "auto_and_commute:traffic",
            66: "auto_and_commute:tire_pressure",
            67: "auto_and_commute:schedule_maintenance",
            68: "auto_and_commute:gas",
            69: "auto_and_commute:mpg",
            70: "auto_and_commute:distance",
            71: "auto_and_commute:directions",
            72: "auto_and_commute:last_maintenance",
            73: "auto_and_commute:gas_type",
            74: "auto_and_commute:tire_change",
            75: "auto_and_commute:jump_start",
            76: "travel:plug_type",
            77: "travel:travel_notification",
            78: "travel:translate",
            79: "travel:flight_status",
            80: "travel:international_visa",
            81: "travel:timezone",
            82: "travel:exchange_rate",
            83: "travel:travel_suggestion",
            84: "travel:travel_alert",
            85: "travel:vaccines",
            86: "travel:lost_luggage",
            87: "travel:book_flight",
            88: "travel:book_hotel",
            89: "travel:carry_on",
            90: "travel:car_rental",
            91: "utility:weather",
            92: "utility:alarm",
            93: "utility:date",
            94: "utility:find_phone",
            95: "utility:share_location",
            96: "utility:timer",
            97: "utility:make_call",
            98: "utility:calculator",
            99: "utility:definition",
            100: "utility:measurement_conversion",
            101: "utility:flip_coin",
            102: "utility:spelling",
            103: "utility:time",
            104: "utility:roll_dice",
            105: "utility:text",
            106: "work:pto_request_status",
            107: "work:next_holiday",
            108: "work:insurance_change",
            109: "work:insurance",
            110: "work:meeting_schedule",
            111: "work:payday",
            112: "work:taxes",
            113: "work:income",
            114: "work:rollover_401k",
            115: "work:pto_balance",
            116: "work:pto_request",
            117: "work:w2",
            118: "work:schedule_meeting",
            119: "work:direct_deposit",
            120: "work:pto_used",
            121: "small_talk:who_made_you",
            122: "small_talk:meaning_of_life",
            123: "small_talk:who_do_you_work_for",
            124: "small_talk:do_you_have_pets",
            125: "small_talk:what_are_your_hobbies",
            126: "small_talk:fun_fact",
            127: "small_talk:what_is_your_name",
            128: "small_talk:where_are_you_from",
            129: "small_talk:goodbye",
            130: "small_talk:thank_you",
            131: "small_talk:greeting",
            132: "small_talk:tell_joke",
            133: "small_talk:are_you_a_bot",
            134: "small_talk:how_old_are_you",
            135: "small_talk:what_can_i_ask_you",
            136: "meta:change_speed",
            137: "meta:user_name",
            138: "meta:whisper_mode",
            139: "meta:yes",
            140: "meta:change_volume",
            141: "meta:no",
            142: "meta:change_language",
            143: "meta:repeat",
            144: "meta:change_accent",
            145: "meta:cancel",
            146: "meta:sync_device",
            147: "meta:change_user_name",
            148: "meta:change_ai_name",
            149: "meta:reset_settings",
            150: "meta:maybe"
        }
    elif dataset_name == "xlsum":
        raw_data = load_dataset('GEM/xlsum', 'english', trust_remote_code=True)
        raw_data = raw_data.rename_column('target', 'label')
        id2label=None
    elif dataset_name == "wmt19": ## train 34782245 val 2998
        raw_data = load_dataset('wmt/wmt19', 'de-en', trust_remote_code=True)
        print(f"Original dataset statistics\ntrain:{len(raw_data['train'])}\nvalidation:{len(raw_data['validation'])}")
        raw_data['test'] = raw_data['validation']
        raw_data['train'] = raw_data['train'].shuffle()
        raw_data['validation'] = raw_data['train'][0:100]
        raw_data['train'] = raw_data['train'][100:10100]
    elif dataset_name == 'gsm8k': ##train 7473 test 1319
        raw_data = load_dataset('openai/gsm8k', 'main')
        d = raw_data['train'].train_test_split(test_size=500)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_algebra':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'algebra')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_counting_and_probability':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'counting_and_probability')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_geometry':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'geometry')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_intermediate_algebra':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'intermediate_algebra')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_number_theory':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'number_theory')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_prealgebra':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'prealgebra')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
    elif dataset_name == 'math_precalculus':
        raw_data = load_dataset('EleutherAI/hendrycks_math', 'precalculus')
        d = raw_data['train'].train_test_split(test_size=100)
        raw_data['validation'] = d['test']
        raw_data['train'] = d['train']
        
    train = raw_data['train']
    val = raw_data['validation']
    test = raw_data['test']

    if dataset_name == "wmt19":
        train_text, train_label = preprocess_data_wmt19(train)
        val_text, val_label = preprocess_data_wmt19(val)
        test_text, test_label = preprocess_data_wmt19(test)
    elif dataset_name == "gsm8k":
        train_text, train_label = preprocess_data_gsm8k(train)
        val_text, val_label = preprocess_data_gsm8k(val)
        test_text, test_label = preprocess_data_gsm8k(test)
    elif 'math' in dataset_name:
        train_text, train_label = preprocess_data_math(train)
        val_text, val_label = preprocess_data_math(val)
        test_text, test_label = preprocess_data_math(test)
    else:
        train_text, train_label = preprocess_data(train, id2label)
        val_text, val_label = preprocess_data(val, id2label)
        test_text, test_label = preprocess_data(test, id2label)
    
    train_dataset = convert_to_dict(train_text, train_label)
    val_dataset = convert_to_dict(val_text, val_label)
    test_dataset = convert_to_dict(test_text, test_label)

    if len(train_dataset)>train_max_limit:
        train_dataset = random.sample(train_dataset, train_max_limit)
    if len(val_dataset)>val_max_limit:
        val_dataset = random.sample(val_dataset, val_max_limit)
    if len(test_dataset)>test_max_limit:
        test_dataset = random.sample(test_dataset, test_max_limit)

    print(f"preprocessed dataset statistics\ntrain:{len(train_dataset)}\nvalidation:{len(val_dataset)}\ntest:{len(test_dataset)}")


    with open(os.path.join(data_path_root, dataset_name, "train.json"), "w") as f:
        json.dump(train_dataset,f)
    with open(os.path.join(data_path_root, dataset_name, "val.json"), "w") as f:
        json.dump(val_dataset,f)
    with open(os.path.join(data_path_root, dataset_name, "test.json"), "w") as f:
        json.dump(test_dataset,f)







    
    