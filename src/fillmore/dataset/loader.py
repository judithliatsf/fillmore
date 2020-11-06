import json
from fillmore.dataset.utils import tprint

import numpy as np

def _get_clinc150_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'translate': 0,
        'transfer': 1,
        'timer': 2,
        'definition': 3,
        'meaning_of_life': 4,
        'insurance_change': 5,
        'find_phone': 6,
        'travel_alert': 7,
        'pto_request': 8,
        'improve_credit_score': 9,
        'fun_fact': 10,
        'change_language': 11,
        'payday': 12,
        'replacement_card_duration': 13,
        'time': 14,
        'application_status': 15,
        'flight_status': 16,
        'flip_coin': 17,
        'change_user_name': 18,
        'where_are_you_from': 19,
        'shopping_list_update': 20,
        'what_can_i_ask_you': 21,
        'maybe': 22,
        'oil_change_how': 23,
        'restaurant_reservation': 24,
        'balance': 25,
        'confirm_reservation': 26,
        'freeze_account': 27,
        'rollover_401k': 28,
        'who_made_you': 29,
        'distance': 30,
        'user_name': 31,
        'timezone': 32,
        'next_song': 33,
        'transactions': 34,
        'restaurant_suggestion': 35,
        'rewards_balance': 36,
        'pay_bill': 37,
        'spending_history': 38,
        'pto_request_status': 39,
        'credit_score': 40,
        'new_card': 41,
        'lost_luggage': 42,
        'repeat': 43,
        'mpg': 44,
        'oil_change_when': 45,
        'yes': 46,
        'travel_suggestion': 47,
        'insurance': 48,
        'todo_list_update': 49,
        'reminder': 50,
        'change_speed': 51,
        'tire_pressure': 52,
        'no': 53,
        'apr': 54,
        'nutrition_info': 55,
        'calendar': 56,
        'uber': 57,
        'calculator': 58,
        'date': 59,
        'carry_on': 60,
        'pto_used': 61,
        'schedule_maintenance': 62,
        'travel_notification': 63,
        'sync_device': 64,
        'thank_you': 65,
        'roll_dice': 66,
        'food_last': 67,
        'cook_time': 68,
        'reminder_update': 69,
        'report_lost_card': 70,
        'ingredient_substitution': 71,
        'make_call': 72,
        'alarm': 73,
        'todo_list': 74,
        'change_accent': 75,
        'w2': 76,
        'bill_due': 77,
        'calories': 78,
        'damaged_card': 79,
        'restaurant_reviews': 80,
        'routing': 81,
        'do_you_have_pets': 82,
        'schedule_meeting': 83,
        'gas_type': 84,
        'plug_type': 85,
        'tire_change': 86,
        'exchange_rate': 87,
        'next_holiday': 88,
        'change_volume': 89,
        'who_do_you_work_for': 90,
        'credit_limit': 91,
        'how_busy': 92,
        'accept_reservations': 93,
        'order_status': 94,
        'pin_change': 95,
        'goodbye': 96,
        'account_blocked': 97,
        'what_song': 98,
        'international_fees': 99,
        'last_maintenance': 100,
        'meeting_schedule': 101,
        'ingredients_list': 102,
        'report_fraud': 103,
        'measurement_conversion': 104,
        'smart_home': 105,
        'book_hotel': 106,
        'current_location': 107,
        'weather': 108,
        'taxes': 109,
        'min_payment': 110,
        'whisper_mode': 111,
        'cancel': 112,
        'international_visa': 113,
        'vaccines': 114,
        'pto_balance': 115,
        'directions': 116,
        'spelling': 117,
        'greeting': 118,
        'reset_settings': 119,
        'what_is_your_name': 120,
        'direct_deposit': 121,
        'interest_rate': 122,
        'credit_limit_change': 123,
        'what_are_your_hobbies': 124,
        'book_flight': 125,
        'shopping_list': 126,
        'text': 127,
        'bill_balance': 128,
        'share_location': 129,
        'redeem_rewards': 130,
        'play_music': 131,
        'calendar_update': 132,
        'are_you_a_bot': 133,
        'gas': 134,
        'expiration_date': 135,
        'update_playlist': 136,
        'cancel_reservation': 137,
        'tell_joke': 138,
        'change_ai_name': 139,
        'how_old_are_you': 140,
        'car_rental': 141,
        'jump_start': 142,
        'meal_suggestion': 143,
        'recipe': 144,
        'income': 145,
        'order': 146,
        'traffic': 147,
        'order_checks': 148,
        'card_declined': 149}

    train_classes = list(range(135))
    val_classes = list(range(136, 140))
    test_classes = list(range(141, 150))

    return train_classes, val_classes, test_classes


def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
            'talk.politics.mideast': 0,
            'sci.space': 1,
            'misc.forsale': 2,
            'talk.politics.misc': 3,
            'comp.graphics': 4,
            'sci.crypt': 5,
            'comp.windows.x': 6,
            'comp.os.ms-windows.misc': 7,
            'talk.politics.guns': 8,
            'talk.religion.misc': 9,
            'rec.autos': 10,
            'sci.med': 11,
            'comp.sys.mac.hardware': 12,
            'sci.electronics': 13,
            'rec.sport.hockey': 14,
            'alt.atheism': 15,
            'rec.motorcycles': 16,
            'comp.sys.ibm.pc.hardware': 17,
            'rec.sport.baseball': 18,
            'soc.religion.christian': 19,
        }

    train_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['sci', 'rec']:
            train_classes.append(label_dict[key])

    val_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['comp']:
            val_classes.append(label_dict[key])

    test_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
            test_classes.append(label_dict[key])

    return train_classes, val_classes, test_classes


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }

    train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
    val_classes = [1, 22, 23, 6, 9]
    test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

    return train_classes, val_classes, test_classes


def _get_rcv1_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = [1, 2, 12, 15, 18, 20, 22, 25, 27, 32, 33, 34, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                     54, 55, 56, 57, 58, 59, 60, 61, 66]
    val_classes = [5, 24, 26, 28, 29, 31, 35, 23, 67, 36]
    test_classes = [0, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 30, 37,
                    62, 63, 64, 65, 68, 69, 70]

    return train_classes, val_classes, test_classes


def _get_fewrel_classes(args):
    '''
        @return list of classes associated with each split
    '''
    # head=WORK_OF_ART validation/test split
    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    return train_classes, val_classes, test_classes


def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(20))
    val_classes = list(range(20,25))
    test_classes = list(range(25,41))

    return train_classes, val_classes, test_classes


def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(15))
    val_classes = list(range(15,20))
    test_classes = list(range(20,31))

    return train_classes, val_classes, test_classes


def _load_json(path, dataset):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        data_by_class = {}
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item_label = int(row['label'])
            
            if dataset == "clinc150":
                item = {
                    'label': item_label,
                    'text': row['raw']
                }
            else:
                item = {
                    'label': item_label,
                    'text': row['text'][:500]  # truncate the text to 500 tokens
                }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)
            
            # add items under the same class label
            if item_label in data_by_class.keys():
                data_by_class[item_label].append(item)
            else:
                data_by_class[item_label] = [item]

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data, data_by_class


# def _read_words(data):
#     '''
#         Count the occurrences of all words
#         @param data: list of examples
#         @return words: list of words (with duplicates)
#     '''
#     words = []
#     for example in data:
#         words += example['text']
#     return words


# def _meta_split(all_data, train_classes, val_classes, test_classes):
#     '''
#         Split the dataset according to the specified train_classes, val_classes
#         and test_classes

#         @param all_data: list of examples (dictionaries)
#         @param train_classes: list of int
#         @param val_classes: list of int
#         @param test_classes: list of int

#         @return train_data: list of examples
#         @return val_data: list of examples
#         @return test_data: list of examples
#     '''
#     train_data, val_data, test_data = [], [], []

#     for example in all_data:
#         if example['label'] in train_classes:
#             train_data.append(example)
#         if example['label'] in val_classes:
#             val_data.append(example)
#         if example['label'] in test_classes:
#             test_data.append(example)

#     return train_data, val_data, test_data

def load_dataset(args):
    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes = _get_20newsgroup_classes(args)
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes = _get_amazon_classes(args)
    elif args.dataset == 'fewrel':
        train_classes, val_classes, test_classes = _get_fewrel_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes(args)
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes = _get_reuters_classes(args)
    elif args.dataset == 'rcv1':
        train_classes, val_classes, test_classes = _get_rcv1_classes(args)
    elif args.dataset == 'clinc150':
        train_classes, val_classes, test_classes = _get_clinc150_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1, clinc150]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    if args.mode == 'finetune':
        # in finetune, we combine train and val for training the base classifier
        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class
        args.n_val_class = args.n_train_class

    tprint('Loading data')
    all_data, data_by_class = _load_json(args.data_path, args.dataset)

    # # Split into meta-train, meta-val, meta-test data
    # train_data, val_data, test_data = _meta_split(
    #         all_data, train_classes, val_classes, test_classes)
    # tprint('#train {}, #val {}, #test {}'.format(
    #     len(train_data), len(val_data), len(test_data)))

    return train_classes, val_classes, test_classes, data_by_class

if __name__ == "__main__":
    all_data, data_by_class = _load_json("data/clinc150.json", "clinc150")
    print(len(all_data))
