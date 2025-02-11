import os, json
import torch, numpy as np
import argparse
from utils.model_utils import *
from utils.data_utils import *
from utils.inference_utils import *
from utils.extract_utils import *
from tqdm import tqdm

INS_DICT={
    "banking77" : """You are an intelligent assistant trained to classify customer queries into predefined categories. Your task is to read a customer's query and assign it to the most appropriate category from the following list of categories:
1.activate_my_card
2.age_limit
3.apple_pay_or_google_pay
4.atm_support
5.automatic_top_up
6.balance_not_updated_after_bank_transfer
7.balance_not_updated_after_cheque_or_cash_deposit
8.beneficiary_not_allowed
9.cancel_transfer
10.card_about_to_expire
11.card_acceptance
12.card_arrival
13.card_delivery_estimate
14.card_linking
15.card_not_working
16.card_payment_fee_charged
17.card_payment_not_recognised
18.card_payment_wrong_exchange_rate
19.card_swallowed
20.cash_withdrawal_charge
21.cash_withdrawal_not_recognised
22.change_pin
23.compromised_card
24.contactless_not_working
25.country_support
26.declined_card_payment
27.declined_cash_withdrawal
28.declined_transfer
29.direct_debit_payment_not_recognised
30.disposable_card_limits
31.edit_personal_details
32.exchange_charge
33.exchange_rate
34.exchange_via_app
35.extra_charge_on_statement
36.failed_transfer
37.fiat_currency_support
38.get_disposable_virtual_card
39.get_physical_card
40.getting_spare_card
41.getting_virtual_card
42.lost_or_stolen_card
43.lost_or_stolen_phone
44.order_physical_card
45.passcode_forgotten
46.pending_card_payment
47.pending_cash_withdrawal
48.pending_top_up
49.pending_transfer
50.pin_blocked
51.receiving_money
52.refund_not_showing_up
53.request_refund
54.reverted_card_payment
55.supported_cards_and_currencies
56.terminate_account
57.top_up_by_bank_transfer_charge
58.top_up_by_card_charge
59.top_up_by_cash_or_cheque
60.top_up_failed
61.top_up_limits
62.top_up_reverted
63.topping_up_by_card
64.transaction_charged_twice
65.transfer_fee_charged
66.transfer_into_account
67.transfer_not_received_by_recipient
68.transfer_timing
69.unable_to_verify_identity
70.verify_my_identity
71.verify_source_of_funds
72.verify_top_up
73.virtual_card_not_working
74.visa_or_mastercard
75.why_verify_identity
76.wrong_amount_of_cash_received
77.wrong_exchange_rate_for_cash_withdrawal
Please analyze the query and return the most suitable category from the list above. If the query does not fit perfectly into one category, select the category that best matches the main topic of the query. Only return intent category text and don't return any other words or numbers.""",
    "trec" : """You are an intelligent assistant trained to classify questions into predefined fine categories. Your task is to read a question and assign it to the most appropriate category from the following list of categories:
0.abbreviation:abbreviation
1.abbreviation:expression
2.entity:animal
3.entity:body
4.entity:color
5.entity:creative work
6.entity:currency
7.entity:disease and medicine
8.entity:event
9.entity:food
10.entity:instrument
11.entity:language
12.entity:letter
13.entity:other
14.entity:plant
15.entity:product
16.entity:religion
17.entity:sport
18.entity:substance
19.entity:symbol
20.entity:technique and method
21.entity:equivalent term
22.entity:vehicle
23.entity:special word
24.description:definition
25.description:description
26.description:manner
27.description:reason
28.human:group
29.human:individual
30.human:title
31.human:description
32.location:city
33.location:country
34.location:mountain
35.location:other
36.location:state
37.numeric:code
38.numeric:count
39.numeric:date
40.numeric:distance
41.numeric:money
42.numeric:order
43.numeric:other
44.numeric:period
45.numeric:percentage
46.numeric:speed
47.numeric:temperature
48.numeric:size and volume
49.numeric:weight
Please analyze the question and return the most suitable coarse category from the list above. If the question does not fit perfectly into one category, select the category that best matches the main topic of the question. Only return intent category text and don't return any other words or numbers.""",
    'clinc150':"""You are an intelligent assistant trained to classify customer queries into predefined categories. Your task is to read a customer's query and assign it to the most appropriate category from the following list of categories:
1.banking:freeze_account
2.banking:routing
3.banking:pin_change
4.banking:bill_due
5.banking:pay_bill
6.banking:account_blocked
7.banking:interest_rate
8.banking:min_payment
9.banking:bill_balance
10.banking:transfer
11.banking:order_checks
12.banking:balance
13.banking:spending_history
14.banking:transactions
15.banking:report_fraud
16.credit_cards:replacement_card_duration
17.credit_cards:expiration_date
18.credit_cards:damaged_card
19.credit_cards:improve_credit_score
20.credit_cards:report_lost_card
21.credit_cards:card_declined
22.credit_cards:credit_limit_change
23.credit_cards:apr
24.credit_cards:redeem_rewards
25.credit_cards:credit_limit
26.credit_cards:rewards_balance
27.credit_cards:application_status
28.credit_cards:credit_score
29.credit_cards:new_card
30.credit_cards:international_fees
31.kitchen_and_dining:food_last
32.kitchen_and_dining:confirm_reservation
33.kitchen_and_dining:how_busy
34.kitchen_and_dining:ingredients_list
35.kitchen_and_dining:calories
36.kitchen_and_dining:nutrition_info
37.kitchen_and_dining:recipe
38.kitchen_and_dining:restaurant_reviews
39.kitchen_and_dining:restaurant_reservation
40.kitchen_and_dining:meal_suggestion
41.kitchen_and_dining:restaurant_suggestion
42.kitchen_and_dining:cancel_reservation
43.kitchen_and_dining:ingredient_substitution
44.kitchen_and_dining:cook_time
45.kitchen_and_dining:accept_reservations
46.home:what_song
47.home:play_music
48.home:todo_list_update
49.home:reminder
50.home:reminder_update
51.home:calendar_update
52.home:order_status
53.home:update_playlist
54.home:shopping_list
55.home:calendar
56.home:next_song
57.home:order
58.home:todo_list
59.home:shopping_list_update
60.home:smart_home
61.auto_and_commute:current_location
62.auto_and_commute:oil_change_when
63.auto_and_commute:oil_change_how
64.auto_and_commute:uber
65.auto_and_commute:traffic
66.auto_and_commute:tire_pressure
67.auto_and_commute:schedule_maintenance
68.auto_and_commute:gas
69.auto_and_commute:mpg
70.auto_and_commute:distance
71.auto_and_commute:directions
72.auto_and_commute:last_maintenance
73.auto_and_commute:gas_type
74.auto_and_commute:tire_change
75.auto_and_commute:jump_start
76.travel:plug_type
77.travel:travel_notification
78.travel:translate
79.travel:flight_status
80.travel:international_visa
81.travel:timezone
82.travel:exchange_rate
83.travel:travel_suggestion
84.travel:travel_alert
85.travel:vaccines
86.travel:lost_luggage
87.travel:book_flight
88.travel:book_hotel
89.travel:carry_on
90.travel:car_rental
91.utility:weather
92.utility:alarm
93.utility:date
94.utility:find_phone
95.utility:share_location
96.utility:timer
97.utility:make_call
98.utility:calculator
99.utility:definition
100.utility:measurement_conversion
101.utility:flip_coin
102.utility:spelling
103.utility:time
104.utility:roll_dice
105.utility:text
106.work:pto_request_status
107.work:next_holiday
108.work:insurance_change
109.work:insurance
110.work:meeting_schedule
111.work:payday
112.work:taxes
113.work:income
114.work:rollover_401k
115.work:pto_balance
116.work:pto_request
117.work:w2
118.work:schedule_meeting
119.work:direct_deposit
120.work:pto_used
121.small_talk:who_made_you
122.small_talk:meaning_of_life
123.small_talk:who_do_you_work_for
124.small_talk:do_you_have_pets
125.small_talk:what_are_your_hobbies
126.small_talk:fun_fact
127.small_talk:what_is_your_name
128.small_talk:where_are_you_from
129.small_talk:goodbye
130.small_talk:thank_you
131.small_talk:greeting
132.small_talk:tell_joke
133.small_talk:are_you_a_bot
134.small_talk:how_old_are_you
135.small_talk:what_can_i_ask_you
136.meta:change_speed
137.meta:user_name
138.meta:whisper_mode
139.meta:yes
140.meta:change_volume
141.meta:no
142.meta:change_language
143.meta:repeat
144.meta:change_accent
145.meta:cancel
146.meta:sync_device
147.meta:change_user_name
148.meta:change_ai_name
149.meta:reset_settings
150.meta:maybe
Please analyze the query and return the most suitable category from the list above. If the query does not fit perfectly into one category, select the category that best matches the main topic of the query. Only return intent category text and don't return any other words or numbers.""",
}
PREFIX_DICT={
    "banking77" : {"input": "Query:", "output" : "Intent:", "instructions" : INS_DICT["banking77"]},
    "trec" : {"input": "Question:", "output" : "Type:", "instructions" : INS_DICT["trec"]},
    "clinc150": {"input": "Query:", "output" : "Intent:", "instructions" : INS_DICT["clinc150"]},
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=100)
    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset_name
    model_name = args.model_name
    n_shots = args.n_shots
    save_path_root = f"{args.save_path_root}/{dataset_name}/{n_shots}shots/{seed}/"
    device = args.device

    prefixes = PREFIX_DICT[dataset_name]
    separators = {"input":"\n", "output":"\n\n", "instructions":"\n"}


    print(args)
    set_seed(seed)
    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, device=device)
    model.eval()

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root, exist_ok=True)

    print("Loading Dataset")
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name)


    # print("load KV-cache of Instruction prompt")
    # kv_cache = instruction_kv_caching(model, tokenizer, prefixes, separators)

    # print(f"Vanilla {n_shots}ICL")
    # fs_result, fs_score= icl_without_intervention(train_dataset=train_dataset, test_dataset=test_dataset, n_shots=n_shots, model=model, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    # print(f"Vanilla ICL result: {fs_score}")
    # fs_result = {'score': fs_score, 'result':fs_result}
    # with open(save_path_root+"fs_result.json", 'w') as f:
    #     json.dump(fs_result, f)
    
    # print("Zero-shot")
    # zs_result, zs_score= icl_without_intervention(train_dataset=train_dataset, test_dataset=test_dataset, n_shots=0, model=model, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    # print(f"Zero-shot ICL result: {zs_score}")
    # zs_result = {'score': zs_score, 'result':zs_result}
    # with open(save_path_root+"zs_result.json", 'w') as f:
    #     json.dump(zs_result, f)

    # EDIT_LAYER = [11,12,13,14,15,16,17,18]
    
    # print("Baseline ICV")
    # if os.path.exists(os.path.join(save_path_root, f"baseline_icv_{n_shots}shots.pt")):
    #     icv = torch.load(os.path.join(save_path_root, f"baseline_icv_{n_shots}shots.pt"))
    # else:
    #     icv = get_mean_hidden_states(train_dataset=train_dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=len(test_dataset), prefixes=prefixes, separators=separators)
    #     torch.save(icv, os.path.join(save_path_root, f"baseline_icv_{n_shots}shots.pt"))
    # val_f1_per_layer = {l:0 for l in EDIT_LAYER}
    # for l in tqdm(EDIT_LAYER):
    #     edit_layer = [l]
    #     _, baseline_icv_val_f1 = icl_with_intervention(test_dataset=val_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer)
    #     val_f1_per_layer[l] = baseline_icv_val_f1
    # edit_layer = [max(val_f1_per_layer, key=val_f1_per_layer.get)]
    # baseline_icv_res, baseline_icv_f1 = icl_with_intervention(test_dataset=test_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer)
    # print(f"Baseline-ICV result: {baseline_icv_f1} with edit layer {edit_layer[0]}")
    # baseline_icv_val_f1_per_layer = val_f1_per_layer
    # baseline_icv_res = {'score': baseline_icv_f1, 'result':baseline_icv_res, 'val_f1_per_layer': baseline_icv_val_f1_per_layer}
    # with open(save_path_root+"baseline_icv_result.json", "w") as f:
    #     json.dump(baseline_icv_res, f)
    
    # print("Diff-ICV Baseline")
    # if os.path.exists(os.path.join(save_path_root, f"baseline_diff_icv_{n_shots}shots.pt")):
    #     icv = torch.load(os.path.join(save_path_root, f"baseline_diff_icv_{n_shots}shots.pt"))
    # else:
    #     icv = get_diff_mean_hidden_states(train_dataset=train_dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=len(test_dataset), prefixes=prefixes, separators=separators)
    #     torch.save(icv, os.path.join(save_path_root, f"baseline_diff_icv_{n_shots}shots.pt"))
    # val_f1_per_layer = {l:0 for l in EDIT_LAYER}
    # for l in tqdm(EDIT_LAYER):
    #     edit_layer = [l]
    #     _, baseline_diff_icv_val_f1 = icl_with_intervention(test_dataset=val_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer)
    #     val_f1_per_layer[l] = baseline_diff_icv_val_f1
    # edit_layer = [max(val_f1_per_layer, key=val_f1_per_layer.get)]
    # baseline_diff_icv_res, baseline_diff_icv_f1 = icl_with_intervention(test_dataset=test_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer)
    # print(f"Diff-ICV Baseline result: {baseline_diff_icv_f1} with edit layer {edit_layer[0]}")
    # baseline_diff_icv_val_f1_per_layer = val_f1_per_layer
    # baseline_diff_icv_res = {'score': baseline_diff_icv_f1, 'result':baseline_diff_icv_res, 'val_f1_per_layer': baseline_diff_icv_val_f1_per_layer}
    # with open(save_path_root+"baseline_diff_icv_result.json", "w") as f:
    #     json.dump(baseline_diff_icv_res, f)


    print("Diff-ICV Stacked")
    icv = get_diff_stacked_hidden_states(train_dataset=train_dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=len(test_dataset), prefixes=prefixes, separators=separators)
    edit_layer = list(range(model_config['n_layers']))
    stacked_diff_icv_res, stacked_diff_icv_score = icl_with_intervention(test_dataset=test_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer)
    print(f"Diff-ICV Stacked result : {stacked_diff_icv_score}")
    stacked_diff_icv_res = {'score': stacked_diff_icv_score, 'result': stacked_diff_icv_res}
    with open(save_path_root+"stacked_diff_icv_result.json", "w") as f:
        json.dump(stacked_diff_icv_res, f)













    
