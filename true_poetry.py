
import os
import string as str
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from random import randint, seed
import math
import pickle
import re
import sys



punctuation = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 58, 59, 60, 61, 62, 63,
               90, 91, 92, 93, 220, 352, 357, 362, 366, 405, 438, 486, 492, 513, 526,
               532, 553, 580, 604, 642, 650, 657, 678, 685, 705, 718, 720, 737, 764,
               767, 796, 807, 828, 830, 834, 837, 838, 855, 860, 930, 939, 940, 982,
               986, 1003, 1058, 1065, 1105, 1106, 1120, 1129, 1157, 1160, 1174, 1220,
               1222, 1238, 1248, 1264, 1267, 1270, 1279, 1298, 1303, 1314, 1315, 1343,
               1367, 1377, 1378, 1391, 1415, 1421, 1427, 1433, 1467, 1478, 1485, 1495,
               1507, 1511, 1539, 1542, 1558, 1584, 1594, 1596, 1600, 1635, 1679, 1701,
               1731, 1776, 1782, 1783, 1795, 1802, 1821, 1828, 1853, 1875, 1899, 1911,
               1946, 1954, 1959, 1983, 1987, 2014, 2026, 2075, 2078, 2079, 2091, 2109,
               2154, 2162, 2167, 2177, 2211, 2231, 2235, 2242, 2310, 2319, 2321, 2327,
               2361, 2388, 2404, 2414, 2425, 2430, 2466, 2474, 2481, 2488, 2534, 2548,
               2559, 2579, 2598, 2599, 2602, 2608, 2623, 2624, 2625, 2637, 2644, 2670,
               2681, 2682, 2713, 2718, 2757, 2780, 2791, 2808, 2813, 2816, 2857, 2864,
               2919, 2920, 2931, 2996, 2998, 2999, 3023, 3050, 3064, 3070, 3104, 3126,
               3132, 3134, 3228, 3256, 3261, 3270, 3312, 3324, 3365, 3373, 3388, 3419,
               3439, 3459, 3467, 3510, 3548, 3553, 3556, 3559, 3571, 3648, 3682, 3693,
               3695, 3712, 3717, 3720, 3784, 3829, 3865, 3880, 3901, 3933, 3980, 4008,
               4019, 4032, 4051, 4059, 4064, 4083, 4089, 4101, 4153, 4181, 4211, 4242,
               4275, 4304, 4309, 4310, 4317, 4343, 4349, 4353, 4357, 4407, 4458, 4521,
               4524, 4531, 4557, 4570, 4600, 4613, 4626, 4747, 4751, 4761, 4764, 4770,
               4790, 4793, 4808, 4841, 4846, 4869, 4880, 4895, 4907, 4943, 4967, 4974,
               5014, 5066, 5075, 5125, 5145, 5214, 5218, 5237, 5299, 5304, 5320, 5323,
               5332, 5333, 5433, 5441, 5472, 5512, 5534, 5539, 5598, 5607, 5619, 5633,
               5705, 5769, 5774, 5816, 5824, 5846, 5855, 5867, 5878, 5892, 5946, 5974,
               5996, 5999, 6052, 6073, 6135, 6200, 6244, 6298, 6303, 6329, 6337, 6390,
               6420, 6469, 6624, 6640, 6659, 6739, 6740, 6852, 6885, 6927, 6957, 6999,
               7029, 7061, 7131, 7169, 7175, 7192, 7198, 7203, 7225, 7265, 7337, 7358,
               7359, 7388, 7410, 7441, 7479, 7499, 7559, 7600, 7618, 7632, 7643, 7665,
               7724, 7769, 7795, 7804, 7816, 7863, 7874, 7879, 7904, 7908, 7930, 7982,
               8054, 8069, 8093, 8133, 8162, 8172, 8183, 8190, 8235, 8257, 8269, 8275,
               8298, 8309, 8348, 8351, 8412, 8454, 8487, 8541, 8576, 8614, 8628, 8644,
               8646, 8684, 8699, 8702, 8728, 8735, 8753, 8762, 8784, 8854, 8864, 8870,
               8915, 8949, 8964, 8973, 9031, 9063, 9130, 9162, 9166, 9193, 9225, 9313,
               9415, 9466, 9507, 9508, 9609, 9656, 9661, 9698, 9705, 9768, 9773, 9783,
               9796, 9804, 9805, 9816, 9832, 9849, 9879, 9907, 9919, 9959, 10048, 10052,
               10053, 10083, 10091, 10097, 10111, 10148, 10163, 10185, 10190, 10221,
               10232, 10249, 10261, 10333, 10354, 10460, 10495, 10531, 10535, 10541,
               10563, 10612, 10779, 10786, 11024, 11037, 11074, 11097, 11104, 11139,
               11207, 11208, 11245, 11323, 11405, 11420, 11442, 11445, 11470, 11485,
               11496, 11502, 11509, 11528, 11537, 11546, 11592, 11593, 11623, 11639,
               11645, 11709, 11757, 11785, 11900, 11907, 11919, 12095, 12113, 12122,
               12131, 12179, 12195, 12240, 12248, 12279, 12340, 12359, 12404, 12429,
               12713, 12726, 12762, 12813, 12825, 12844, 12863, 12865, 12877, 12878,
               12923, 12952, 12962, 13018, 13037, 13108, 13130, 13151, 13163, 13219,
               13330, 13343, 13348, 13374, 13381, 13412, 13426, 13454, 13464, 13498,
               13521, 13531, 13538, 13539, 13540, 13557, 13679, 13702, 13803, 13841,
               13896, 13979, 13984, 14004, 14030, 14062, 14079, 14198, 14223, 14280,
               14315, 14373, 14436, 14454, 14468, 14489, 14512, 14585, 14610, 14631,
               14656, 14686, 14692, 14726, 14745, 14804, 14808, 14877, 14956, 14980,
               15089, 15090, 15116, 15136, 15143, 15179, 15187, 15197, 15211, 15231,
               15259, 15277, 15306, 15341, 15349, 15363, 15377, 15408, 15426, 15437,
               15473, 15495, 15506, 15524, 15533, 15589, 15629, 15674, 15696, 15711,
               15724, 15729, 15761, 15801, 15853, 15885, 15886, 15897, 15904, 15913,
               15920, 15931, 15963, 15982, 16003, 16078, 16088, 16101, 16102, 16150,
               16151, 16193, 16208, 16226, 16236, 16243, 16315, 16317, 16327, 16382,
               16410, 16450, 16471, 16489, 16529, 16562, 16589, 16616, 16626, 16677,
               16679, 16725, 16763, 16791, 16792, 16799, 16817, 16822, 16942, 16945,
               16994, 17020, 17031, 17032, 17059, 17174, 17202, 17241, 17279, 17318,
               17342, 17405, 17414, 17426, 17430, 17464, 17477, 17501, 17544, 17553,
               17569, 17572, 17575, 17635, 17643, 17657, 17672, 17729, 17759, 17816,
               17817, 17827, 17885, 17912, 17971, 18005, 18083, 18109, 18112, 18125,
               18161, 18182, 18189, 18237, 18294, 18298, 18376, 18395, 18444, 18458,
               18477, 18500, 18523, 18604, 18638, 18693, 18741, 18742, 18781, 18823,
               18897, 18938, 18946, 19004, 19035, 19038, 19039, 19048, 19060, 19104,
               19153, 19203, 19214, 19244, 19322, 19342, 19351, 19355, 19409, 19415,
               19420, 19424, 19427, 19442, 19504, 19510, 19529, 19570, 19571, 19588,
               19622, 19629, 19683, 19707, 19708, 19710, 19755, 19779, 19782, 19841,
               19880, 19884, 19891, 19924, 19953, 19969, 19990, 20004, 20007, 20033,
               20064, 20107, 20167, 20198, 20219, 20224, 20233, 20248, 20262, 20299,
               20343, 20356, 20368, 20370, 20379, 20416, 20479, 20483, 20510, 20520,
               20548, 20598, 20662, 20666, 20679, 20708, 20740, 20789, 20809, 20924,
               20943, 20959, 20964, 20986, 20995, 21017, 21033, 21056, 21113, 21139,
               21148, 21215, 21261, 21268, 21273, 21288, 21315, 21355, 21387, 21395,
               21409, 21431, 21489, 21495, 21498, 21503, 21526, 21536, 21577, 21599,
               21601, 21626, 21643, 21652, 21709, 21719, 21734, 21737, 21738, 21747,
               21761, 21777, 21794, 21807, 21844, 21895, 21908, 21912, 21940, 22039,
               22042, 22047, 22131, 22133, 22135, 22136, 22148, 22169, 22172, 22186,
               22219, 22241, 22243, 22288, 22291, 22305, 22318, 22330, 22337, 22345,
               22352, 22369, 22370, 22413, 22416, 22446, 22458, 22492, 22515, 22538,
               22544, 22567, 22572, 22579, 22613, 22626, 22666, 22675, 22709, 22717,
               22725, 22730, 22745, 22759, 22784, 22799, 22800, 22831, 22842, 22855,
               22857, 22883, 22909, 22913, 22914, 22935, 22951, 22955, 22980, 22985,
               22986, 22995, 22996, 23029, 23031, 23045, 23055, 23120, 23134, 23148,
               23188, 23193, 23195, 23234, 23237, 23313, 23330, 23336, 23344, 23349,
               23362, 23378, 23451, 23460, 23487, 23516, 23539, 23601, 23628, 23664,
               23666, 23679, 23721, 23726, 23734, 23753, 23756, 23785, 23815, 23846,
               23847, 23859, 23871, 23884, 23906, 23924, 23926, 23984, 24018, 24022,
               24036, 24038, 24041, 24045, 24063, 24096, 24136, 24137, 24168, 24200,
               24214, 24217, 24235, 24293, 24294, 24305, 24309, 24339, 24356, 24369,
               24403, 24409, 24414, 24457, 24465, 24529, 24555, 24591, 24598, 24618,
               24620, 24648, 24652, 24669, 24693, 24718, 24760, 24793, 24839, 24840,
               24844, 24848, 24894, 24909, 24938, 24940, 24943, 24970, 24977, 24991,
               25022, 25054, 25061, 25090, 25096, 25106, 25113, 25128, 25150, 25177,
               25181, 25190, 25191, 25226, 25240, 25248, 25257, 25264, 25270, 25272,
               25295, 25307, 25325, 25326, 25399, 25429, 25475, 25500, 25508, 25540,
               25597, 25600, 25609, 25643, 25644, 25645, 25658, 25666, 25667, 25674,
               25698, 25707, 25710, 25719, 25764, 25780, 25787, 25816, 25829, 25836,
               25838, 25859, 25870, 25887, 25915, 25947, 25948, 25964, 25970, 25998,
               26007, 26033, 26050, 26063, 26073, 26076, 26115, 26118, 26143, 26171,
               26200, 26214, 26250, 26259, 26276, 26279, 26290, 26352, 26358, 26398,
               26422, 26427, 26429, 26481, 26488, 26492, 26513, 26514, 26525, 26539,
               26561, 26582, 26598}
acceptable_punctuation = {13, 0, 11, 30, 25, 26, 6, 1, 7, 8, 438, 12, 705, 357, 1267, 366, 1377, 220, 764}
end_punctuation = {0, 13, 11, 26, 30}

class Struct:
    pass

params = Struct()
params.rhyme_set_size = 20
params.probability_threshold = .00005
params.line_probability_threshold = 0
params.ultimate_expansion = 1000
params.penultimate_expansion = 10
params.other_expansion = 10
params.random_seed = 28
params.line_end_punctuation_constraint = True
params.punctuation_probability_threshold = .001
# recommended for Mac CPU: use small gpt2; change to "gpt2-xl" if you want the large model
params.model_name = "gpt2-xl"
params.stuck_counter_limit = 1000
params.one_syllable_suppression = 20
debug = False

def xprint(*args, **kwargs):
    global debug
    if debug == True:
        try:
            print(*args, **kwargs)
        except:
            print("error in printing")

def text_to_meter(text, stress_dictionary):
    if len(text) == 0:
        return ''
    s = text.upper()
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    s2 = ''.join(filter(whitelist.__contains__, s))
    split_list = re.split('[\s\']', s2)
    line_stress = ""
    for word in split_list:
        if len(word) > 0:
            if word in stress_dictionary:
                line_stress = line_stress + stress_dictionary[word]
            else:
                line_stress = line_stress + "*"
    return line_stress

def rhyme_check(text1, target_rhyme_list, rhyme_dictionary, reverse_rhyme_dictionary, params):
    global acceptable_punctuation
    global bad_rhymes
    xprint("target_rhyme_list =")
    xprint(target_rhyme_list)
    if len(target_rhyme_list) > 0:
        target_rhyme_line = target_rhyme_list[0]
    else:
        target_rhyme_line = ""
    xprint("rhyme_check target_rhyme_line =")
    xprint(target_rhyme_line)
    text1 = text1.strip().lower()
    target_rhyme_line = target_rhyme_line.strip().lower()
    if target_rhyme_line == "!":
        return True
    if target_rhyme_line == "":
        if text1 == "":
            return True
        else:
            text1_words = text1.split(" ")
            last_word1 = text1_words[-1]
            if last_word1 in rhyme_dictionary:
                if rhyme_dictionary[last_word1] in reverse_rhyme_dictionary:
                    enough_rhymes = len(reverse_rhyme_dictionary[rhyme_dictionary[last_word1]]) > params.rhyme_set_size
                    if enough_rhymes and (not last_word1 in bad_rhymes):
                        return True
                    else:
                        xprint("! not enough rhymes or last word 1 in bad_rhymes")
                        return False
                else:
                    xprint("! not in reverse dictionary ")
                    return False
            else:
                xprint("! last word 1 not in rhyme dictionary")
                return False
    else:
        text1_words = text1.split(" ")
        last_word1 = text1_words[-1]
        regex = re.compile('[^a-zA-Z]')
        last_word1 = regex.sub('', last_word1)
        target_rhyme_line_words = target_rhyme_line.split(" ")
        last_word2 = target_rhyme_line_words[-1]
        regex = re.compile('[^a-zA-Z]')
        last_word2 = regex.sub('', last_word2)
        for line in target_rhyme_list:
            target_rhyme_line = line.strip().lower()
            target_rhyme_line_words = target_rhyme_line.split(" ")
            last_word2 = target_rhyme_line_words[-1]
            regex = re.compile('[^a-zA-Z]')
            last_word2 = regex.sub('', last_word2)
            xprint("checking rhymes against:", last_word2)
            if last_word1 == last_word2:
                xprint("! a word is rhyming with itself")
                return False
        if (last_word1 in rhyme_dictionary) and (last_word2 in rhyme_dictionary):
            rhyme1 = rhyme_dictionary[last_word1]
            rhyme2 = rhyme_dictionary[last_word2]
            rhyme1 = rhyme1.replace("0", "1")
            rhyme2 = rhyme2.replace("0", "1")
            if (rhyme1 == rhyme2):
                return True
            else:
                xprint("! last word1 does not rhyme with last word 2")
                return False
        else:
            xprint("! last word 1 or last word 2 not in rhyme dictionary")
            xprint(last_word1)
            xprint(last_word2)
            return False

def compare_meters(test_meter, target_meter):
    matchflag = False
    if len(test_meter) > 0 and test_meter[-1] == "*":
        test_meter = test_meter[:-1]
    if "*" in test_meter[:-1]:
        return False
    if len(test_meter) <= len(target_meter):
        matchflag = True
        for character1, character2 in zip(test_meter, target_meter):
            if (character1 == "`" and character2 == "`") or (character1 == "~" and character2 == "~") or character1 == "?":
                pass
            else:
                matchflag = False
    if len(test_meter) == 0:
        matchflag = True
    return matchflag

def rhyme_and_meter_filter(this_text_sentence, target_rhyme_list, target_meter, probs, params):
    global stress_tokens
    global acceptable_punctuation
    global rhyming_tokens
    global syllable_tokens
    offset = randint(0, 2)
    this_meter = text_to_meter(this_text_sentence, stress_dictionary)
    xprint("target_rhyme_list =")
    xprint(target_rhyme_list)
    if len(target_rhyme_list) > 0:
        target_rhyme = target_rhyme_list[0]
    else:
        target_rhyme = target_rhyme_list
    xprint("target_rhyme_list =")
    xprint(target_rhyme_list)

    next_stresses = target_meter[len(this_meter):min(len(this_meter) + 3, len(target_meter) + 1)]
    if len(next_stresses) == 0:
        return []
    all_tokens = set(range(0, 50257))
    stress_okay = set(stress_tokens[next_stresses])
    for token in all_tokens.difference(stress_okay.union(acceptable_punctuation)):
        probs[token] = 0

    too_common_tokens = syllable_tokens[1].union(acceptable_punctuation)
    for t in range(0, 50257):
        if t in too_common_tokens:
            probs[t] = probs[t] / params.one_syllable_suppression
    if len(target_rhyme) > 0 and target_rhyme != "!":
        target_rhyme_words = target_rhyme.split(" ")
        last_target_rhyme_word = target_rhyme_words[-1].strip().lower()
        if last_target_rhyme_word and last_target_rhyme_word[-1] in {"!", ".", ",", ";", ":", "?", "-"}:
            last_target_rhyme_word = last_target_rhyme_word[:-1]
        xprint("target rhyme =", last_target_rhyme_word)
        these_rhyming_tokens = rhyming_tokens.get(last_target_rhyme_word, set())
        if len(this_meter) == len(target_meter) - 1:
            for t in range(0, 50257):
                if t in these_rhyming_tokens:
                    pass
                else:
                    probs[t] = 0
        elif len(this_meter) == len(target_meter) - 2:
            safeset = syllable_tokens[1].union(these_rhyming_tokens)
            for t in range(0, 50257):
                if t in safeset:
                    pass
                else:
                    probs[t] = 0
        sorted_probability_list = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        short_probability_list = sorted_probability_list[0 + offset:params.ultimate_expansion + offset]
        xprint("PART 1")
    elif len(this_meter) > len(target_meter) - 3:
        sorted_probability_list = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        short_probability_list = sorted_probability_list[0 + offset:params.penultimate_expansion + offset]
        xprint("PART 2")
    elif len(this_meter) < 1:
        sorted_probability_list = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        short_probability_list = sorted_probability_list
        xprint("PART 3")
    else:
        sorted_probability_list = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        short_probability_list = sorted_probability_list[0 + offset:params.other_expansion + offset]
        xprint("PART 4")
    short_probability_list = [i for i in short_probability_list if i[1] != 0]
    xprint("short prob list len = ", end=" ")
    xprint(len(short_probability_list))

    return short_probability_list

def grow_branches(these_tokens, probs, input_probability, past, params, prompt_length, target_rhyme_list, target_meter):
    xprint("___________________________________________")
    global model
    global tokenizer
    global stress_dictionary
    global rhyme_dictionary
    global reverse_rhyme_dictionary
    global stuck_counter
    global past_backup
    stuck_counter = stuck_counter + 1
    if stuck_counter > params.stuck_counter_limit:
        params.probability_threshold = params.probability_threshold / 2
        stuck_counter = 0
        past = past_backup
        these_tokens = these_tokens[:prompt_length]
    found = None
    this_text_sentence = tokenizer.decode(these_tokens[prompt_length:])
    if len(these_tokens[prompt_length:]) < 2:
        probability_threshold = 0
    else:
        probability_threshold = params.probability_threshold
    short_probability_list = rhyme_and_meter_filter(this_text_sentence, target_rhyme_list, target_meter, probs, params)
    if len(short_probability_list) == 0:
        xprint("! len(short_probability_list)==0")
        return False
    else:
        count = 0
        for (this_token, this_probability) in short_probability_list:
            xprint("------------------------------")
            tokens_are_probable_enough_to_continue = this_probability > probability_threshold
            xprint("tokens are probable: ", end="")
            xprint(tokens_are_probable_enough_to_continue)
            if not tokens_are_probable_enough_to_continue:
                xprint("! tokens_are_not probable_enough_to_continue")
                return False
            else:
                count = count + 1
                next_probability = this_probability * input_probability
                next_tokens = these_tokens.copy()
                next_tokens.append(this_token)
                next_text_sentence = tokenizer.decode(next_tokens[prompt_length:])
                next_meter = text_to_meter(next_text_sentence, stress_dictionary)
                xprint(next_meter)
                if "*" in next_meter[:-1]:
                    xprint("! * in next meter")
                    return False
                meter_check = compare_meters(next_meter, target_meter)
                print(next_text_sentence)
                if len(next_meter) > len(target_meter):
                    xprint("! len(next_meter)>len(target_meter)")
                    continue
                elif len(next_meter) == len(target_meter):
                    if not meter_check:
                        xprint("! not meter check")
                        continue
                    else:
                        rhyme_checks_out = rhyme_check(next_text_sentence, target_rhyme_list, rhyme_dictionary, reverse_rhyme_dictionary, params)
                        if not rhyme_checks_out:
                            xprint("! rhyme doesn't check out")
                            continue
                        else:
                            (word_completion_list, next_past) = expand_node(next_tokens, past)
                            sorted_word_completion_list = sorted(enumerate(word_completion_list), key=lambda x: x[1], reverse=True)
                            potential_word_completion = tokenizer.decode(sorted_word_completion_list[0][0])
                            all_tokens = set(range(0, 50257))
                            if potential_word_completion and potential_word_completion[0] in str.ascii_lowercase:
                                print("! potential_word_completion[0] in str.ascii_lowercase")
                                continue
                            elif params.line_end_punctuation_constraint == True:
                                (end_punctuation_list, next_past) = (word_completion_list, next_past)
                                for token in all_tokens.difference(end_punctuation):
                                    end_punctuation_list[token] = 0
                                sorted_end_punctuation_list = sorted(enumerate(end_punctuation_list), key=lambda x: x[1], reverse=True)
                                punctuation_probability = sorted_end_punctuation_list[0][1]
                                if punctuation_probability > params.punctuation_probability_threshold:
                                    end_punctuation_choice = sorted_end_punctuation_list[0][0]
                                    next_tokens.append(end_punctuation_choice)
                                    next_text_sentence = tokenizer.decode(next_tokens[prompt_length:])
                                    print("*** " + next_text_sentence + "\t" + next_meter)
                                    return next_tokens[prompt_length:]
                                else:
                                    print("! end punctuation too rare")
                                    continue
                            else:
                                print("*** " + next_text_sentence + "\t" + next_meter)
                                return next_tokens[prompt_length:]
                punctuation_repeats = (len(these_tokens) > 1 and these_tokens[-2] in punctuation and these_tokens[-1] in punctuation) or (len(these_tokens) > 0 and these_tokens[-1] in punctuation and this_token in punctuation)
                line_is_way_too_long = (len(these_tokens[prompt_length + 1:]) > 20)
                if next_probability < params.line_probability_threshold or punctuation_repeats or line_is_way_too_long:
                    if len(these_tokens[prompt_length + 1:]) > 1:
                        xprint("! len(these_tokens[prompt_length+1:])>1")
                        return False
                    else:
                        xprint("! len(these_tokens[prompt_length+1:])<=1")
                        continue
                else:
                    found = False
                    if meter_check and len(next_meter) < len(target_meter):
                        (next_probability_list, next_past) = expand_node(next_tokens, past)
                        found = grow_branches(next_tokens, next_probability_list, next_probability, next_past, params, prompt_length, target_rhyme_list, target_meter)
                    if found != False:
                        xprint("found =", end="")
                        xprint(found)
                        return found
    xprint("! end of function")
    return False

def expand_node(sentence, past):
    global model
    if past == None:
        input_ids = torch.tensor(sentence).unsqueeze(0)
    else:
        input_ids = torch.tensor([sentence[-1]]).unsqueeze(0)
    inputs = {'input_ids': input_ids}
    with torch.no_grad():
        logits, past = model(**inputs, past_key_values=past, return_dict=False)
        logits[0][0][50256] = -math.inf
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).tolist()[0]
        return (probs, past)

def create_stress_dictionary():
    pronounce_file = open("pronounce.txt", "r")
    stress_dictionary = {}
    for line in pronounce_file:
        line = line.strip("\n")
        parts = line.split(" ")
        syllable_list = parts[2:]
        word = parts[0]
        stresses = ""
        if word in ["A", "AN", "THE", "AND", "BUT", "OR"]:
            stresses = "~"
        elif word in ["I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "MY", "HIS", "HER", "ITS", "OUR", "YOUR", "THEIR", "OURS", "YOURS", "THEIRS", "AM", "IS", "ARE", "WAS", "WERE", "BEEN", "BE", "HAVE", "HAS", "HAD", "DO", "DOES", "DID", "WILL", "WOULD", "SHALL", "SHOULD", "MAY", "MIGHT", "MUST", "CAN", "COULD", "OF", "WITH", "AT", "FROM", "TO", "IN", "FOR", "ON", "BY", "LIKE", "SINCE", "UP", "OFF", "NEAR", "WHICH", "AS", "EACH", "SO", "THAT", "THATS"]:
            stresses = "?"
        else:
            for syllable in syllable_list:
                if syllable.endswith("1"):
                    stresses = stresses + "`"
                elif syllable.endswith("0"):
                    stresses = stresses + "~"
                elif syllable.endswith("2"):
                    stresses = stresses + "?"
        if word in {"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"}:
            pass
        else:
            stress_dictionary[word] = stresses
    return stress_dictionary

def create_rhyme_dictionary(tokenizer):
    pronounce_file = open("pronounce.txt", "r")
    rhyme_dictionary = {}
    reverse_rhyme_dictionary = {}
    syllable_count_dictionary = {}
    for line in pronounce_file:
        line = line.strip()
        if line.startswith(';'): continue
        # some lines have a double space separator
        if "  " in line:
            word, phones = line.split("  ", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phones = " ".join(parts[1:])
        syllables = phones.split(" ")
        syllable_count_dictionary[word] = phones.count("0") + phones.count("1") + phones.count("2")
        join_flag = 0
        outstring = ''
        for syllable in syllables:
            if join_flag == 0:
                if "1" in syllable:
                    join_flag = 1
                    outstring = syllable
            else:
                outstring = outstring + " " + syllable
        if outstring == "":
            for syllable in syllables:
                if join_flag == 0:
                    if "0" in syllable:
                        join_flag = 1
                        outstring = syllable
                else:
                    outstring = outstring + " " + syllable
        rhyme_dictionary[word.lower()] = outstring
        if outstring in reverse_rhyme_dictionary:
            reverse_rhyme_dictionary[outstring].append(word.lower())
        else:
            reverse_rhyme_dictionary[outstring] = [word.lower()]

    rhyming_tokens = pickle.load(open("rhyming_tokens.p", "rb"))
    syllable_tokens = pickle.load(open("syllable_tokens.p", "rb"))

    bad_rhymes = ["a", "an", "it", "is", "as", "at", "was", "of", "at", "that",
                  "has", "your", "my", "his", "their", "on", "for", "its", "to",
                  "from", "if", "ur", "re", "our", "un", "dis", "diss", "mis",
                  "wat", "com", "comm", "psych", "lol", "vis", "al", "los", "pol",
                  "bis", "up", " la", "sa", "ha", "mah", " wal", "lat", "ot", "sol",
                  "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    return rhyme_dictionary, reverse_rhyme_dictionary, bad_rhymes, syllable_count_dictionary, rhyming_tokens, syllable_tokens

def poem_scheme(kind):
    # Normalize the input and allow some synonyms
    global poem_line
    if poem_line is None:
        poem_line = [""] * 1000
    k = kind.strip().lower()

    # map some common typos/synonyms
    if k in {"limerick", "limericks"}:
        k = "limerick"
    elif k in {"sonnet", "sonnets"}:
        k = "sonnet"
    elif k in {"ballad", "ballads"}:
        k = "ballad"
    elif k in {"couplets", "couplet"}:
        k = "couplets"
    elif k in {"mini-couplets", "mini couplets", "mini-couplet"}:
        k = "mini-couplets"
    elif k in {"blank verse", "blank-verse", "blankverse"}:
        k = "blank verse"

    # Default fallback values (in case none match)
    number_of_lines = 10
    meter_scheme = [""] * number_of_lines
    rhyme_scheme = [""] * number_of_lines

    if k == "limerick":
        number_of_lines = 5
        meter_scheme = [""] * number_of_lines
        for line in {0, 1, 4}:
            meter_scheme[line] = "~`~~`~~`"
        for line in {2, 3}:
            meter_scheme[line] = "~`~~`"
        rhyme_scheme = ["", [poem_line[0]], "", [poem_line[2]], [poem_line[0], poem_line[1]]]

    elif k == "sonnet":
        number_of_lines = 10
        meter_scheme = [""] * number_of_lines
        for i in range(number_of_lines):
            meter_scheme[i] = "~`~`~`~`~`"
        rhyme_scheme = ["", "", [poem_line[0]], [poem_line[1]], "", "", [poem_line[4]], [poem_line[5]], "", [poem_line[8]]]

    elif k == "blank verse":
        number_of_lines = 10
        meter_scheme = [""] * number_of_lines
        for i in range(number_of_lines):
            meter_scheme[i] = "~`~`~`~`~`"
        rhyme_scheme = [[0]] * number_of_lines

    elif k == "couplets":
        number_of_lines = 10
        meter_scheme = [""] * number_of_lines
        for i in range(number_of_lines):
            meter_scheme[i] = "`~`~`~"
        rhyme_scheme = ["", [poem_line[0]], "", [poem_line[2]], "", [poem_line[4]], "", [poem_line[6]], "", [poem_line[8]]]

    elif k == "mini-couplets":
        number_of_lines = 20
        meter_scheme = [""] * number_of_lines
        for i in range(number_of_lines):
            meter_scheme[i] = "~`~`"
        rhyme_scheme = [""] * number_of_lines
        # Populate rhyme pairs in A A B B ... style for simplicity
        for i in range(0, number_of_lines, 2):
            rhyme_scheme[i] = ""
            rhyme_scheme[i + 1] = [poem_line[i]]
        params.penultimate_expansion = 10000

    elif k == "ballad":
        number_of_lines = 16
        meter_scheme = [""] * number_of_lines
        for line in {0, 2, 4, 6, 8, 10, 12, 14}:
            meter_scheme[line] = "~`~`~`~`"
        for line in {1, 3, 5, 7, 9, 11, 13, 15}:
            meter_scheme[line] = "~`~`~`"
        rhyme_scheme = [[0], "", [0], [poem_line[1]], [0], "", [0], [poem_line[5]], [0], "", [0], [poem_line[9]], [0], "", [0], [poem_line[13]]]

    # else: we keep default number_of_lines, etc.

    # Ensure returned lists match number_of_lines
    if len(meter_scheme) != number_of_lines:
        meter_scheme = ([""] * number_of_lines)[:number_of_lines]
    if len(rhyme_scheme) != number_of_lines:
        rhyme_scheme = ([""] * number_of_lines)[:number_of_lines]

    return number_of_lines, rhyme_scheme, meter_scheme



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


rhyme_dictionary, reverse_rhyme_dictionary, bad_rhymes, syllable_count_dictionary, rhyming_tokens, syllable_tokens = create_rhyme_dictionary(tokenizer)
stress_dictionary = create_stress_dictionary()
stress_tokens = pickle.load(open("stress_tokens.p", "rb"))
xprint("rhymes loaded")


try:
    print(f"Loading model '{params.model_name}' (this may take a moment)...")
    model = GPT2LMHeadModel.from_pretrained(params.model_name)
    xprint("model loaded")
except Exception as e:
    print("Error loading the model:", e)
    print("If you are on a Mac CPU-only machine and used a large model (gpt2-xl or gwern 'poetry'), the process may have run out of memory.")
    print("Recommended: set params.model_name = 'gpt2' and remove any cached large models in ~/.cache/huggingface/transformers")
    raise

xprint("model loaded")

seed(params.random_seed)
with torch.no_grad():
    try:
        raw_prompt = input("starting prompt: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo prompt given. Exiting.")
        sys.exit(0)
    prompt = tokenizer.encode(raw_prompt)
    original_length = len(prompt)
    past = None
    (probs, past) = expand_node(prompt, None)

    # ask for scheme robustly (re-prompt until valid)
    valid_choices = {"ballad", "limerick", "couplets", "sonnet", "mini-couplets"}
    scheme = ""
    while True:
        try:
            scheme = input("ballad, limerick, couplets, mini-couplets or sonnet? ").strip().lower()
            if scheme not in {"ballad", "limerick", "couplets", "mini-couplets", "sonnet", "blank verse"}:
                print("Unknown scheme; defaulting to 'sonnet'.")
                scheme = "sonnet"
        except (EOFError, KeyboardInterrupt):
            print("\nNo scheme given. Exiting.")
            sys.exit(0)
        if scheme in valid_choices:
            break
        else:
            print("Please type exactly one: ballad, limerick, couplets, mini-couplets or sonnet (lowercase). Try again.")

    # prepare poem and generate
    poem_line = [""] * 100000
    number_of_lines, rhyme_scheme, meter_scheme = poem_scheme(scheme)
    poem_line = [""] * number_of_lines
    line = 0
    backup_prompts = [""] * 100
    backup_pasts = [""] * 100
    while line < number_of_lines:
        stuck_counter = 0
        backup_prompts[line] = prompt
        backup_pasts[line] = past
        number_of_lines, rhyme_scheme, meter_scheme = poem_scheme(scheme)
        target_rhyme_list = []
        for target_rhyme_line in rhyme_scheme[line]:
            # if the scheme uses numbers, convert them to tokens; but the original code expects token lists
            # in the original repo poem_line entries were token lists. We'll keep compatibility by converting ints to empty strings here.
            if isinstance(target_rhyme_line, int):
                target_rhyme_list.append("")
            else:
                target_rhyme_list.append(tokenizer.decode(target_rhyme_line) if isinstance(target_rhyme_line, list) else str(target_rhyme_line))
        if target_rhyme_list == []:
            target_rhyme_list = ""
        xprint(target_rhyme_list)
        target_meter = meter_scheme[line]
        this_line = grow_branches(prompt, probs, 1, past, params, len(prompt), target_rhyme_list, target_meter)
        if this_line == False:
            print("something went wrong, quitting")
            break
        poem_line[line] = this_line
        line = line + 1
        prompt = prompt + this_line
        past_backup = past
        (probs, past) = expand_node(prompt, None)

    if poem_line and poem_line[-1] and poem_line[-1][-1] in end_punctuation:
        poem_line[-1][-1] = tokenizer.encode('.')[0]
    print()
    print(tokenizer.decode(prompt[original_length:]))
    print()
    if number_of_lines > 0 and poem_line[0]:
        print(tokenizer.decode(poem_line[0]))
    for ln in range(1, number_of_lines):
        if poem_line[ln] and len(poem_line[ln]) > 0 and poem_line[ln][0] in acceptable_punctuation:
            poem_line[ln - 1].append(poem_line[ln][0])
            poem_line[ln] = poem_line[ln][1:]
    for ln in range(1, number_of_lines):
        if poem_line[ln]:
            print(tokenizer.decode(poem_line[ln]))
