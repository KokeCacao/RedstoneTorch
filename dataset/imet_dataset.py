import collections
import os
import re

import matplotlib as mpl
import torch
import cv2
from tqdm import tqdm

import config
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch._six import string_classes, int_classes
from torch.utils import data
from torch.utils.data import SubsetRandomSampler, Sampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import numpy_type_map, default_collate
from torchvision.transforms import transforms

from albumentations import (
    HorizontalFlip, CLAHE, ShiftScaleRotate, Blur, GaussNoise, RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose, JpegCompression,
    CenterCrop, PadIfNeeded, RandomCrop, RandomGamma, Resize)
# don't import Normalize from albumentations

import tensorboardwriter
from utils.augmentation import AdaptivePadIfNeeded, DoNothing

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class IMetDataset(data.Dataset):

    def __init__(self, train_csv_dir, test_csv_dir, load_strategy="train", writer=None, id_col = 'id', target_col='target'):
        self.writer = writer
        self.load_strategy = load_strategy
        print("     Reading Data with [test={}]".format(self.load_strategy))

        """Make sure the labels of your dataset is correct"""
        self.train_dataframe = pd.read_csv(train_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)
        self.test_dataframe = pd.read_csv(test_csv_dir, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col).sample(frac=1)

        # self.freq_dict = pd.Series(np.hstack(self.train_dataframe.attribute_ids.str.split(" ").values)).value_counts().to_dict()
        # self.freq_dict = {'813': 19970, '1092': 14281, '147': 13522, '189': 10375, '13': 9151, '671': 8419, '51': 7615, '194': 7394, '1059': 6564, '121': 6542, '896': 5955, '1046': 5591, '79': 5382, '780': 5259, '156': 5163, '369': 4416, '744': 3890, '477': 3692, '738': 3665, '1034': 3570, '188': 3500, '835': 3005, '903': 2552, '420': 2548, '1099': 2327, '552': 2180, '485': 2097, '776': 2075, '161': 2050, '489': 2045, '1039': 2001, '733': 1895, '304': 1881, '612': 1789, '111': 1762, '962': 1744, '487': 1685, '501': 1667, '1062': 1540, '961': 1526, '541': 1492, '734': 1480, '483': 1472, '405': 1457, '737': 1446, '597': 1428, '480': 1414, '335': 1403, '718': 1397, '554': 1390, '99': 1327, '259': 1302, '663': 1286, '584': 1283, '616': 1278, '418': 1260, '434': 1213, '768': 1200, '833': 1138, '724': 1083, '993': 1073, '949': 1053, '626': 1050, '784': 1016, '1020': 977, '698': 965, '1084': 961, '492': 959, '830': 957, '872': 956, '754': 921, '800': 902, '538': 901, '433': 888, '542': 866, '464': 859, '212': 838, '70': 831, '1072': 816, '283': 810, '615': 810, '131': 787, '1035': 775, '650': 773, '498': 766, '639': 750, '579': 745, '1061': 744, '955': 682, '25': 676, '404': 676, '559': 672, '796': 637, '430': 636, '655': 630, '975': 627, '1093': 622, '519': 622, '624': 613, '670': 600, '231': 595, '858': 593, '180': 593, '716': 583, '758': 582, '436': 582, '182': 580, '668': 570, '950': 568, '764': 566, '415': 565, '494': 564, '746': 560, '580': 555, '634': 551, '184': 545, '886': 543, '532': 540, '29': 530, '378': 523, '923': 518, '742': 515, '45': 515, '844': 504, '127': 499, '125': 494, '922': 493, '1098': 487, '543': 481, '598': 480, '931': 477, '313': 467, '704': 462, '753': 461, '1023': 454, '747': 453, '586': 446, '603': 440, '720': 433, '583': 427, '952': 422, '774': 422, '953': 413, '14': 408, '227': 403, '682': 399, '821': 393, '339': 388, '582': 387, '974': 382, '428': 373, '1022': 371, '703': 369, '954': 369, '308': 368, '239': 362, '684': 358, '465': 357, '809': 355, '520': 354, '462': 338, '573': 332, '581': 332, '940': 329, '558': 329, '348': 328, '592': 327, '545': 327, '438': 323, '725': 322, '322': 317, '925': 312, '1019': 303, '633': 303, '497': 302, '493': 298, '665': 297, '595': 296, '506': 295, '331': 293, '33': 292, '916': 290, '587': 289, '745': 286, '503': 284, '236': 281, '1032': 273, '515': 273, '568': 270, '803': 269, '202': 269, '647': 266, '766': 265, '517': 263, '991': 262, '783': 262, '978': 258, '707': 257, '691': 257, '553': 256, '726': 256, '490': 254, '432': 254, '133': 254, '994': 253, '488': 253, '1089': 251, '1030': 251, '23': 244, '534': 244, '524': 243, '607': 240, '58': 240, '840': 239, '1064': 236, '637': 236, '786': 234, '282': 233, '819': 233, '690': 230, '728': 229, '1085': 228, '884': 227, '450': 226, '834': 225, '822': 224, '908': 223, '482': 221, '713': 221, '1045': 220, '332': 219, '445': 219, '695': 219, '756': 218, '971': 216, '563': 216, '864': 215, '354': 210, '721': 208, '252': 207, '1012': 207, '890': 206, '570': 205, '739': 205, '518': 205, '889': 204, '701': 204, '672': 203, '705': 202, '929': 200, '678': 196, '1056': 195, '832': 194, '1009': 193, '983': 188, '495': 186, '628': 186, '548': 185, '927': 184, '811': 182, '963': 181, '555': 181, '1038': 180, '910': 179, '870': 179, '26': 177, '631': 177, '675': 176, '988': 174, '992': 173, '924': 170, '620': 170, '426': 169, '688': 167, '871': 167, '621': 165, '1040': 162, '731': 160, '1001': 160, '638': 160, '973': 159, '1097': 158, '306': 157, '989': 156, '692': 156, '849': 155, '826': 153, '522': 153, '280': 152, '551': 150, '676': 150, '627': 150, '996': 148, '972': 148, '383': 148, '791': 147, '912': 147, '932': 147, '591': 146, '78': 146, '863': 144, '1081': 143, '1069': 143, '425': 142, '41': 141, '888': 141, '448': 140, '887': 139, '1096': 137, '985': 137, '508': 137, '1033': 137, '576': 135, '76': 133, '680': 133, '496': 131, '825': 130, '829': 130, '915': 129, '741': 128, '116': 127, '511': 127, '1058': 126, '732': 124, '629': 124, '600': 124, '1037': 124, '507': 123, '666': 123, '18': 121, '535': 121, '669': 120, '1068': 120, '708': 120, '999': 120, '749': 119, '948': 119, '35': 119, '622': 119, '359': 118, '393': 118, '179': 118, '619': 118, '510': 117, '204': 116, '413': 115, '654': 115, '566': 114, '709': 114, '217': 114, '1083': 114, '454': 114, '1071': 113, '947': 113, '794': 113, '642': 112, '750': 112, '347': 112, '681': 111, '273': 111, '468': 111, '808': 111, '907': 110, '1065': 110, '516': 110, '649': 108, '192': 108, '882': 107, '55': 106, '479': 106, '422': 106, '861': 106, '459': 104, '1018': 104, '685': 104, '755': 103, '771': 103, '662': 103, '502': 102, '1066': 100, '1': 100, '546': 99, '674': 99, '540': 98, '1024': 98, '529': 97, '966': 97, '967': 97, '234': 97, '960': 97, '823': 97, '995': 96, '939': 96, '959': 96, '287': 96, '400': 95, '505': 95, '185': 95, '1025': 95, '1051': 95, '866': 94, '921': 94, '860': 94, '135': 94, '693': 93, '645': 93, '253': 93, '138': 93, '285': 93, '536': 92, '898': 92, '1080': 92, '585': 92, '763': 91, '409': 91, '608': 90, '664': 90, '316': 90, '406': 89, '175': 88, '850': 88, '679': 88, '761': 88, '661': 87, '653': 87, '941': 87, '838': 86, '449': 86, '537': 86, '1055': 85, '567': 85, '1101': 85, '714': 85, '556': 84, '1015': 83, '258': 82, '43': 82, '334': 81, '513': 81, '1010': 81, '307': 81, '936': 81, '457': 81, '874': 80, '885': 80, '105': 80, '512': 80, '478': 79, '1036': 79, '942': 79, '836': 79, '804': 79, '867': 79, '466': 79, '846': 78, '596': 78, '1077': 78, '368': 77, '875': 77, '453': 77, '920': 77, '815': 76, '461': 76, '226': 76, '458': 75, '588': 75, '824': 74, '60': 74, '702': 73, '1091': 72, '32': 72, '486': 72, '65': 72, '102': 72, '410': 71, '613': 71, '275': 71, '232': 71, '61': 71, '528': 70, '521': 70, '816': 70, '155': 69, '191': 69, '842': 69, '735': 69, '618': 69, '1013': 68, '1007': 68, '463': 68, '700': 68, '390': 68, '879': 67, '965': 67, '565': 67, '795': 67, '648': 67, '1003': 67, '906': 66, '1053': 66, '710': 66, '905': 66, '1041': 66, '881': 66, '437': 66, '444': 66, '373': 66, '964': 65, '968': 65, '895': 64, '914': 64, '1027': 64, '769': 64, '345': 64, '154': 64, '210': 63, '439': 63, '338': 63, '696': 63, '248': 63, '611': 63, '827': 62, '564': 62, '1088': 62, '218': 62, '1095': 62, '557': 61, '294': 61, '96': 61, '956': 61, '610': 61, '986': 60, '351': 60, '767': 60, '113': 60, '289': 60, '384': 59, '699': 59, '862': 59, '91': 59, '1002': 59, '891': 59, '1047': 59, '593': 58, '412': 58, '901': 58, '606': 58, '447': 58, '657': 58, '114': 58, '899': 58, '1050': 57, '1044': 57, '667': 57, '723': 57, '793': 57, '577': 57, '807': 56, '590': 56, '276': 56, '817': 56, '269': 56, '549': 56, '913': 56, '162': 55, '1054': 55, '706': 55, '1005': 55, '937': 54, '578': 54, '677': 53, '1073': 53, '423': 53, '806': 53, '163': 53, '646': 52, '145': 52, '173': 52, '1004': 52, '323': 52, '911': 52, '604': 52, '1078': 52, '514': 52, '660': 51, '828': 51, '934': 51, '878': 51, '605': 51, '976': 51, '623': 51, '977': 51, '641': 50, '748': 50, '349': 50, '42': 50, '1006': 50, '569': 50, '868': 49, '1057': 49, '772': 49, '712': 49, '673': 48, '399': 48, '40': 48, '533': 48, '1063': 48, '424': 48, '1043': 48, '797': 48, '256': 48, '951': 48, '1086': 48, '711': 47, '760': 47, '574': 47, '446': 47, '467': 47, '820': 47, '318': 46, '782': 46, '876': 46, '778': 46, '571': 46, '998': 45, '831': 45, '865': 45, '110': 45, '562': 45, '321': 45, '1049': 45, '683': 44, '757': 44, '86': 44, '196': 44, '375': 44, '263': 43, '770': 43, '274': 43, '69': 43, '473': 43, '95': 43, '1026': 42, '902': 42, '401': 41, '938': 41, '451': 41, '427': 41, '969': 41, '24': 41, '470': 41, '107': 41, '360': 40, '309': 40, '124': 40, '818': 40, '484': 40, '773': 40, '1074': 40, '272': 39, '572': 39, '324': 39, '560': 39, '945': 39, '636': 39, '178': 38, '279': 38, '1000': 38, '237': 38, '946': 38, '144': 38, '686': 38, '775': 38, '315': 37, '740': 37, '848': 37, '526': 37, '149': 37, '926': 37, '729': 36, '195': 36, '411': 36, '980': 36, '421': 36, '792': 36, '158': 36, '984': 36, '935': 36, '614': 36, '979': 36, '530': 35, '441': 35, '251': 35, '656': 35, '429': 35, '382': 35, '44': 35, '722': 35, '759': 34, '48': 34, '1075': 34, '442': 34, '1070': 34, '715': 34, '15': 34, '777': 34, '659': 34, '475': 34, '1082': 33, '292': 33, '57': 33, '249': 33, '789': 32, '385': 32, '257': 32, '342': 32, '788': 32, '847': 32, '134': 32, '122': 32, '893': 31, '408': 31, '651': 31, '1102': 31, '539': 31, '90': 30, '299': 30, '531': 30, '265': 30, '1100': 30, '730': 30, '205': 30, '990': 30, '509': 30, '958': 30, '894': 30, '141': 29, '997': 29, '358': 29, '397': 29, '900': 29, '74': 29, '1014': 29, '500': 29, '50': 29, '630': 28, '270': 28, '617': 28, '419': 28, '83': 28, '1042': 28, '499': 28, '957': 28, '9': 28, '877': 27, '837': 27, '717': 27, '609': 27, '839': 27, '320': 27, '547': 27, '169': 27, '440': 26, '151': 26, '933': 26, '109': 26, '1016': 26, '326': 26, '471': 26, '472': 26, '165': 26, '640': 26, '751': 26, '658': 26, '762': 25, '387': 25, '1079': 25, '469': 25, '106': 25, '504': 25, '859': 25, '801': 24, '550': 24, '602': 24, '625': 24, '810': 24, '455': 24, '869': 24, '75': 24, '802': 24, '689': 23, '120': 23, '1008': 23, '814': 23, '325': 23, '171': 23, '150': 22, '67': 22, '398': 22, '402': 22, '59': 22, '743': 22, '785': 22, '909': 22, '341': 22, '841': 22, '880': 22, '245': 22, '93': 22, '22': 22, '222': 22, '694': 22, '1067': 22, '374': 22, '981': 22, '857': 21, '1029': 21, '170': 21, '247': 21, '601': 21, '417': 21, '229': 21, '632': 21, '216': 21, '350': 21, '1011': 21, '456': 21, '380': 21, '781': 20, '1028': 20, '379': 20, '943': 20, '736': 20, '851': 20, '928': 20, '765': 20, '17': 20, '403': 19, '362': 19, '852': 19, '414': 19, '255': 19, '136': 19, '98': 19, '589': 19, '377': 19, '435': 19, '416': 19, '89': 19, '407': 18, '223': 18, '153': 18, '0': 18, '944': 18, '970': 18, '371': 18, '687': 18, '337': 18, '481': 18, '1090': 17, '266': 17, '52': 17, '1021': 17, '525': 17, '261': 17, '336': 17, '117': 17, '790': 17, '228': 17, '5': 17, '344': 17, '982': 17, '644': 17, '317': 16, '357': 16, '1087': 16, '353': 16, '575': 16, '491': 16, '856': 16, '301': 16, '392': 16, '172': 16, '85': 16, '799': 16, '361': 15, '300': 15, '208': 15, '930': 15, '319': 15, '443': 14, '139': 14, '1076': 14, '152': 14, '2': 14, '594': 14, '853': 14, '157': 14, '8': 14, '77': 14, '12': 14, '267': 14, '238': 14, '63': 13, '64': 13, '27': 13, '1052': 13, '1048': 13, '330': 13, '97': 13, '213': 13, '352': 13, '4': 13, '49': 12, '244': 12, '72': 12, '148': 12, '1094': 12, '697': 12, '918': 12, '28': 12, '16': 12, '356': 12, '246': 12, '779': 12, '295': 11, '386': 11, '62': 11, '1031': 11, '206': 11, '897': 11, '474': 11, '66': 11, '46': 11, '243': 10, '225': 10, '719': 10, '209': 10, '343': 10, '54': 10, '87': 10, '220': 10, '10': 10, '56': 10, '310': 10, '164': 10, '752': 10, '302': 10, '38': 10, '277': 10, '919': 10, '143': 10, '296': 9, '854': 9, '452': 9, '303': 9, '523': 9, '643': 9, '476': 9, '193': 9, '200': 9, '47': 9, '305': 9, '183': 9, '798': 9, '288': 9, '395': 8, '599': 8, '174': 8, '812': 8, '168': 8, '84': 8, '652': 8, '381': 8, '460': 8, '254': 8, '394': 8, '561': 8, '215': 8, '68': 8, '364': 7, '37': 7, '346': 7, '101': 7, '333': 7, '224': 7, '370': 7, '845': 7, '355': 7, '635': 7, '917': 7, '544': 7, '892': 7, '119': 7, '340': 7, '159': 7, '73': 7, '128': 7, '235': 7, '241': 7, '19': 7, '242': 7, '197': 7, '181': 6, '264': 6, '219': 6, '118': 6, '311': 6, '388': 6, '527': 6, '137': 6, '363': 6, '314': 6, '186': 6, '130': 6, '883': 6, '167': 6, '843': 6, '82': 6, '391': 6, '286': 6, '126': 6, '431': 5, '80': 5, '31': 5, '21': 5, '268': 5, '211': 5, '365': 5, '298': 5, '873': 5, '297': 5, '207': 5, '30': 5, '1060': 5, '115': 5, '284': 4, '327': 4, '88': 4, '132': 4, '855': 4, '20': 4, '214': 4, '176': 4, '92': 4, '34': 4, '260': 4, '166': 4, '290': 4, '177': 4, '291': 4, '6': 4, '278': 4, '129': 4, '233': 4, '787': 4, '103': 4, '367': 4, '198': 3, '36': 3, '727': 3, '53': 3, '100': 3, '7': 3, '1017': 3, '39': 3, '140': 3, '203': 3, '250': 3, '329': 3, '372': 3, '376': 3, '3': 3, '190': 3, '123': 3, '160': 3, '94': 3, '71': 2, '187': 2, '312': 2, '108': 2, '201': 2, '271': 2, '240': 2, '987': 2, '904': 2, '142': 2, '389': 2, '396': 1, '262': 1, '328': 1, '230': 1, '104': 1, '281': 1, '805': 1, '199': 1, '146': 1, '112': 1, '221': 1, '293': 1, '11': 1, '81': 1, '366': 1}
        # self.freq_dict = {813: 19970, 1092: 14281, 147: 13522, 189: 10375, 13: 9151, 671: 8419, 51: 7615, 194: 7394, 1059: 6564, 121: 6542, 896: 5955, 1046: 5591, 79: 5382, 780: 5259, 156: 5163, 369: 4416, 744: 3890, 477: 3692, 738: 3665, 1034: 3570, 188: 3500, 835: 3005, 903: 2552, 420: 2548, 1099: 2327, 552: 2180, 485: 2097, 776: 2075, 161: 2050, 489: 2045, 1039: 2001, 733: 1895, 304: 1881, 612: 1789, 111: 1762, 962: 1744, 487: 1685, 501: 1667, 1062: 1540, 961: 1526, 541: 1492, 734: 1480, 483: 1472, 405: 1457, 737: 1446, 597: 1428, 480: 1414, 335: 1403, 718: 1397, 554: 1390, 99: 1327, 259: 1302, 663: 1286, 584: 1283, 616: 1278, 418: 1260, 434: 1213, 768: 1200, 833: 1138, 724: 1083, 993: 1073, 949: 1053, 626: 1050, 784: 1016, 1020: 977, 698: 965, 1084: 961, 492: 959, 830: 957, 872: 956, 754: 921, 800: 902, 538: 901, 433: 888, 542: 866, 464: 859, 212: 838, 70: 831, 1072: 816, 283: 810, 615: 810, 131: 787, 1035: 775, 650: 773, 498: 766, 639: 750, 579: 745, 1061: 744, 955: 682, 25: 676, 404: 676, 559: 672, 796: 637, 430: 636, 655: 630, 975: 627, 1093: 622, 519: 622, 624: 613, 670: 600, 231: 595, 858: 593, 180: 593, 716: 583, 758: 582, 436: 582, 182: 580, 668: 570, 950: 568, 764: 566, 415: 565, 494: 564, 746: 560, 580: 555, 634: 551, 184: 545, 886: 543, 532: 540, 29: 530, 378: 523, 923: 518, 742: 515, 45: 515, 844: 504, 127: 499, 125: 494, 922: 493, 1098: 487, 543: 481, 598: 480, 931: 477, 313: 467, 704: 462, 753: 461, 1023: 454, 747: 453, 586: 446, 603: 440, 720: 433, 583: 427, 952: 422, 774: 422, 953: 413, 14: 408, 227: 403, 682: 399, 821: 393, 339: 388, 582: 387, 974: 382, 428: 373, 1022: 371, 703: 369, 954: 369, 308: 368, 239: 362, 684: 358, 465: 357, 809: 355, 520: 354, 462: 338, 573: 332, 581: 332, 940: 329, 558: 329, 348: 328, 592: 327, 545: 327, 438: 323, 725: 322, 322: 317, 925: 312, 1019: 303, 633: 303, 497: 302, 493: 298, 665: 297, 595: 296, 506: 295, 331: 293, 33: 292, 916: 290, 587: 289, 745: 286, 503: 284, 236: 281, 1032: 273, 515: 273, 568: 270, 803: 269, 202: 269, 647: 266, 766: 265, 517: 263, 991: 262, 783: 262, 978: 258, 707: 257, 691: 257, 553: 256, 726: 256, 490: 254, 432: 254, 133: 254, 994: 253, 488: 253, 1089: 251, 1030: 251, 23: 244, 534: 244, 524: 243, 607: 240, 58: 240, 840: 239, 1064: 236, 637: 236, 786: 234, 282: 233, 819: 233, 690: 230, 728: 229, 1085: 228, 884: 227, 450: 226, 834: 225, 822: 224, 908: 223, 482: 221, 713: 221, 1045: 220, 332: 219, 445: 219, 695: 219, 756: 218, 971: 216, 563: 216, 864: 215, 354: 210, 721: 208, 252: 207, 1012: 207, 890: 206, 570: 205, 739: 205, 518: 205, 889: 204, 701: 204, 672: 203, 705: 202, 929: 200, 678: 196, 1056: 195, 832: 194, 1009: 193, 983: 188, 495: 186, 628: 186, 548: 185, 927: 184, 811: 182, 963: 181, 555: 181, 1038: 180, 910: 179, 870: 179, 26: 177, 631: 177, 675: 176, 988: 174, 992: 173, 924: 170, 620: 170, 426: 169, 688: 167, 871: 167, 621: 165, 1040: 162, 731: 160, 1001: 160, 638: 160, 973: 159, 1097: 158, 306: 157, 989: 156, 692: 156, 849: 155, 826: 153, 522: 153, 280: 152, 551: 150, 676: 150, 627: 150, 996: 148, 972: 148, 383: 148, 791: 147, 912: 147, 932: 147, 591: 146, 78: 146, 863: 144, 1081: 143, 1069: 143, 425: 142, 41: 141, 888: 141, 448: 140, 887: 139, 1096: 137, 985: 137, 508: 137, 1033: 137, 576: 135, 76: 133, 680: 133, 496: 131, 825: 130, 829: 130, 915: 129, 741: 128, 116: 127, 511: 127, 1058: 126, 732: 124, 629: 124, 600: 124, 1037: 124, 507: 123, 666: 123, 18: 121, 535: 121, 669: 120, 1068: 120, 708: 120, 999: 120, 749: 119, 948: 119, 35: 119, 622: 119, 359: 118, 393: 118, 179: 118, 619: 118, 510: 117, 204: 116, 413: 115, 654: 115, 566: 114, 709: 114, 217: 114, 1083: 114, 454: 114, 1071: 113, 947: 113, 794: 113, 642: 112, 750: 112, 347: 112, 681: 111, 273: 111, 468: 111, 808: 111, 907: 110, 1065: 110, 516: 110, 649: 108, 192: 108, 882: 107, 55: 106, 479: 106, 422: 106, 861: 106, 459: 104, 1018: 104, 685: 104, 755: 103, 771: 103, 662: 103, 502: 102, 1066: 100, 1: 100, 546: 99, 674: 99, 540: 98, 1024: 98, 529: 97, 966: 97, 967: 97, 234: 97, 960: 97, 823: 97, 995: 96, 939: 96, 959: 96, 287: 96, 400: 95, 505: 95, 185: 95, 1025: 95, 1051: 95, 866: 94, 921: 94, 860: 94, 135: 94, 693: 93, 645: 93, 253: 93, 138: 93, 285: 93, 536: 92, 898: 92, 1080: 92, 585: 92, 763: 91, 409: 91, 608: 90, 664: 90, 316: 90, 406: 89, 175: 88, 850: 88, 679: 88, 761: 88, 661: 87, 653: 87, 941: 87, 838: 86, 449: 86, 537: 86, 1055: 85, 567: 85, 1101: 85, 714: 85, 556: 84, 1015: 83, 258: 82, 43: 82, 334: 81, 513: 81, 1010: 81, 307: 81, 936: 81, 457: 81, 874: 80, 885: 80, 105: 80, 512: 80, 478: 79, 1036: 79, 942: 79, 836: 79, 804: 79, 867: 79, 466: 79, 846: 78, 596: 78, 1077: 78, 368: 77, 875: 77, 453: 77, 920: 77, 815: 76, 461: 76, 226: 76, 458: 75, 588: 75, 824: 74, 60: 74, 702: 73, 1091: 72, 32: 72, 486: 72, 65: 72, 102: 72, 410: 71, 613: 71, 275: 71, 232: 71, 61: 71, 528: 70, 521: 70, 816: 70, 155: 69, 191: 69, 842: 69, 735: 69, 618: 69, 1013: 68, 1007: 68, 463: 68, 700: 68, 390: 68, 879: 67, 965: 67, 565: 67, 795: 67, 648: 67, 1003: 67, 906: 66, 1053: 66, 710: 66, 905: 66, 1041: 66, 881: 66, 437: 66, 444: 66, 373: 66, 964: 65, 968: 65, 895: 64, 914: 64, 1027: 64, 769: 64, 345: 64, 154: 64, 210: 63, 439: 63, 338: 63, 696: 63, 248: 63, 611: 63, 827: 62, 564: 62, 1088: 62, 218: 62, 1095: 62, 557: 61, 294: 61, 96: 61, 956: 61, 610: 61, 986: 60, 351: 60, 767: 60, 113: 60, 289: 60, 384: 59, 699: 59, 862: 59, 91: 59, 1002: 59, 891: 59, 1047: 59, 593: 58, 412: 58, 901: 58, 606: 58, 447: 58, 657: 58, 114: 58, 899: 58, 1050: 57, 1044: 57, 667: 57, 723: 57, 793: 57, 577: 57, 807: 56, 590: 56, 276: 56, 817: 56, 269: 56, 549: 56, 913: 56, 162: 55, 1054: 55, 706: 55, 1005: 55, 937: 54, 578: 54, 677: 53, 1073: 53, 423: 53, 806: 53, 163: 53, 646: 52, 145: 52, 173: 52, 1004: 52, 323: 52, 911: 52, 604: 52, 1078: 52, 514: 52, 660: 51, 828: 51, 934: 51, 878: 51, 605: 51, 976: 51, 623: 51, 977: 51, 641: 50, 748: 50, 349: 50, 42: 50, 1006: 50, 569: 50, 868: 49, 1057: 49, 772: 49, 712: 49, 673: 48, 399: 48, 40: 48, 533: 48, 1063: 48, 424: 48, 1043: 48, 797: 48, 256: 48, 951: 48, 1086: 48, 711: 47, 760: 47, 574: 47, 446: 47, 467: 47, 820: 47, 318: 46, 782: 46, 876: 46, 778: 46, 571: 46, 998: 45, 831: 45, 865: 45, 110: 45, 562: 45, 321: 45, 1049: 45, 683: 44, 757: 44, 86: 44, 196: 44, 375: 44, 263: 43, 770: 43, 274: 43, 69: 43, 473: 43, 95: 43, 1026: 42, 902: 42, 401: 41, 938: 41, 451: 41, 427: 41, 969: 41, 24: 41, 470: 41, 107: 41, 360: 40, 309: 40, 124: 40, 818: 40, 484: 40, 773: 40, 1074: 40, 272: 39, 572: 39, 324: 39, 560: 39, 945: 39, 636: 39, 178: 38, 279: 38, 1000: 38, 237: 38, 946: 38, 144: 38, 686: 38, 775: 38, 315: 37, 740: 37, 848: 37, 526: 37, 149: 37, 926: 37, 729: 36, 195: 36, 411: 36, 980: 36, 421: 36, 792: 36, 158: 36, 984: 36, 935: 36, 614: 36, 979: 36, 530: 35, 441: 35, 251: 35, 656: 35, 429: 35, 382: 35, 44: 35, 722: 35, 759: 34, 48: 34, 1075: 34, 442: 34, 1070: 34, 715: 34, 15: 34, 777: 34, 659: 34, 475: 34, 1082: 33, 292: 33, 57: 33, 249: 33, 789: 32, 385: 32, 257: 32, 342: 32, 788: 32, 847: 32, 134: 32, 122: 32, 893: 31, 408: 31, 651: 31, 1102: 31, 539: 31, 90: 30, 299: 30, 531: 30, 265: 30, 1100: 30, 730: 30, 205: 30, 990: 30, 509: 30, 958: 30, 894: 30, 141: 29, 997: 29, 358: 29, 397: 29, 900: 29, 74: 29, 1014: 29, 500: 29, 50: 29, 630: 28, 270: 28, 617: 28, 419: 28, 83: 28, 1042: 28, 499: 28, 957: 28, 9: 28, 877: 27, 837: 27, 717: 27, 609: 27, 839: 27, 320: 27, 547: 27, 169: 27, 440: 26, 151: 26, 933: 26, 109: 26, 1016: 26, 326: 26, 471: 26, 472: 26, 165: 26, 640: 26, 751: 26, 658: 26, 762: 25, 387: 25, 1079: 25, 469: 25, 106: 25, 504: 25, 859: 25, 801: 24, 550: 24, 602: 24, 625: 24, 810: 24, 455: 24, 869: 24, 75: 24, 802: 24, 689: 23, 120: 23, 1008: 23, 814: 23, 325: 23, 171: 23, 150: 22, 67: 22, 398: 22, 402: 22, 59: 22, 743: 22, 785: 22, 909: 22, 341: 22, 841: 22, 880: 22, 245: 22, 93: 22, 22: 22, 222: 22, 694: 22, 1067: 22, 374: 22, 981: 22, 857: 21, 1029: 21, 170: 21, 247: 21, 601: 21, 417: 21, 229: 21, 632: 21, 216: 21, 350: 21, 1011: 21, 456: 21, 380: 21, 781: 20, 1028: 20, 379: 20, 943: 20, 736: 20, 851: 20, 928: 20, 765: 20, 17: 20, 403: 19, 362: 19, 852: 19, 414: 19, 255: 19, 136: 19, 98: 19, 589: 19, 377: 19, 435: 19, 416: 19, 89: 19, 407: 18, 223: 18, 153: 18, 0: 18, 944: 18, 970: 18, 371: 18, 687: 18, 337: 18, 481: 18, 1090: 17, 266: 17, 52: 17, 1021: 17, 525: 17, 261: 17, 336: 17, 117: 17, 790: 17, 228: 17, 5: 17, 344: 17, 982: 17, 644: 17, 317: 16, 357: 16, 1087: 16, 353: 16, 575: 16, 491: 16, 856: 16, 301: 16, 392: 16, 172: 16, 85: 16, 799: 16, 361: 15, 300: 15, 208: 15, 930: 15, 319: 15, 443: 14, 139: 14, 1076: 14, 152: 14, 2: 14, 594: 14, 853: 14, 157: 14, 8: 14, 77: 14, 12: 14, 267: 14, 238: 14, 63: 13, 64: 13, 27: 13, 1052: 13, 1048: 13, 330: 13, 97: 13, 213: 13, 352: 13, 4: 13, 49: 12, 244: 12, 72: 12, 148: 12, 1094: 12, 697: 12, 918: 12, 28: 12, 16: 12, 356: 12, 246: 12, 779: 12, 295: 11, 386: 11, 62: 11, 1031: 11, 206: 11, 897: 11, 474: 11, 66: 11, 46: 11, 243: 10, 225: 10, 719: 10, 209: 10, 343: 10, 54: 10, 87: 10, 220: 10, 10: 10, 56: 10, 310: 10, 164: 10, 752: 10, 302: 10, 38: 10, 277: 10, 919: 10, 143: 10, 296: 9, 854: 9, 452: 9, 303: 9, 523: 9, 643: 9, 476: 9, 193: 9, 200: 9, 47: 9, 305: 9, 183: 9, 798: 9, 288: 9, 395: 8, 599: 8, 174: 8, 812: 8, 168: 8, 84: 8, 652: 8, 381: 8, 460: 8, 254: 8, 394: 8, 561: 8, 215: 8, 68: 8, 364: 7, 37: 7, 346: 7, 101: 7, 333: 7, 224: 7, 370: 7, 845: 7, 355: 7, 635: 7, 917: 7, 544: 7, 892: 7, 119: 7, 340: 7, 159: 7, 73: 7, 128: 7, 235: 7, 241: 7, 19: 7, 242: 7, 197: 7, 181: 6, 264: 6, 219: 6, 118: 6, 311: 6, 388: 6, 527: 6, 137: 6, 363: 6, 314: 6, 186: 6, 130: 6, 883: 6, 167: 6, 843: 6, 82: 6, 391: 6, 286: 6, 126: 6, 431: 5, 80: 5, 31: 5, 21: 5, 268: 5, 211: 5, 365: 5, 298: 5, 873: 5, 297: 5, 207: 5, 30: 5, 1060: 5, 115: 5, 284: 4, 327: 4, 88: 4, 132: 4, 855: 4, 20: 4, 214: 4, 176: 4, 92: 4, 34: 4, 260: 4, 166: 4, 290: 4, 177: 4, 291: 4, 6: 4, 278: 4, 129: 4, 233: 4, 787: 4, 103: 4, 367: 4, 198: 3, 36: 3, 727: 3, 53: 3, 100: 3, 7: 3, 1017: 3, 39: 3, 140: 3, 203: 3, 250: 3, 329: 3, 372: 3, 376: 3, 3: 3, 190: 3, 123: 3, 160: 3, 94: 3, 71: 2, 187: 2, 312: 2, 108: 2, 201: 2, 271: 2, 240: 2, 987: 2, 904: 2, 142: 2, 389: 2, 396: 1, 262: 1, 328: 1, 230: 1, 104: 1, 281: 1, 805: 1, 199: 1, 146: 1, 112: 1, 221: 1, 293: 1, 11: 1, 81: 1, 366: 1}

        self.multilabel_binarizer = MultiLabelBinarizer().fit([list(range(config.TRAIN_NUM_CLASS)),])
        self.labelframe = None

        if self.load_strategy == "train":
            print("Training Dataframe: {}".format(self.train_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.train_dataframe[target_col]]).tolist()
            id = self.train_dataframe.index.tolist()

            # """Presudo Labeling"""
            # self.presudo_dataframe = pd.read_csv(config.DIRECTORY_PRESUDO_CSV, delimiter=',', encoding="utf-8-sig", engine='python').set_index(id_col)
            # for index in self.presudo_dataframe.index.tolist():
            #     probability = self.presudo_dataframe.Label[index]
            #     id.append('data/HisCancer_dataset/test/'+index+'.npy')
            #     self.labelframe.append([1-probability, probability])

        elif self.load_strategy == "test" or self.load_strategy == "predict":
            print("Predicting Dataframe: {}".format(self.test_dataframe.head()))
            self.labelframe = self.multilabel_binarizer.transform([(int(i) for i in str(s).split()) for s in self.test_dataframe[target_col]])
            id = self.test_dataframe.index.tolist()
        else:
            raise ValueError("the argument [load_strategy] recieved and undefined value: [{}], which is not one of 'train', 'test', 'predict'".format(load_strategy))
        id = list(id)
        self.id_len = int(len(id) * config.TRAIN_DATA_PERCENT)
        self.id = id[:self.id_len]

        self.indices = np.array(list(range(self.id_len)))
        self.indices_to_id = dict(zip(self.indices, self.id))
        self.id_to_indices = {v: k for k, v in self.indices_to_id.items()}

        print("""
            Load Dir:       {}, {}
            ID Size:      {}/{}
            Data Percent:   {}
            Label Size:     {}
            Frame Size:      {}/{}
        """.format(train_csv_dir, test_csv_dir, self.id_len, "?", config.TRAIN_DATA_PERCENT, len(self.labelframe), len(id), "?"))

    def __len__(self):
        return self.id_len


    def get_stratified_samplers(self, fold=-1):
        """
        :param fold: fold number
        :return: dictionary[fold]["train" or "val"]
        """
        X = self.indices
        y = np.array(list(self.get_load_label_by_indice(x) for x in X))

        # print("Indice:{}, Id:{}, Label:{}".format(X[0], self.id[0], y[0]))

        mskf = MultilabelStratifiedKFold(n_splits=fold, random_state=None)
        folded_samplers = dict()


        if config.DEBUG_WRITE_SPLIT_CSV or not os.path.exists(config.DIRECTORY_SPLIT):
            if os.path.exists(config.DIRECTORY_SPLIT):
                os.remove(config.DIRECTORY_SPLIT)
                print("WARNING: the split file '{}' already exist, remove file".format(config.DIRECTORY_SPLIT))

            fold_dict = []
            for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
                print("#{} TRAIN:{} TEST:{}".format(fold, train_index, test_index))
                x_t = train_index
                # y_t = np.array([y[j] for j in train_index])
                x_e = test_index
                # y_e = np.array([y[j] for j in test_index])

                fold_dict.append([x_t, x_e])

                folded_samplers[fold] = dict()
                folded_samplers[fold]["train"] = SubsetRandomSampler(x_t)
                folded_samplers[fold]["val"] = SubsetRandomSampler(x_e)

                def write_cv_distribution(writer, y_t, y_e):
                    y_t_dict = np.bincount((y_t.astype(np.int8) * np.array(list(range(config.TRAIN_NUM_CLASS)))).flatten())
                    y_e_dict = np.bincount((y_e.astype(np.int8) * np.array(list(range(config.TRAIN_NUM_CLASS)))).flatten())
                    # F, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), sharey='none')
                    # ax1.bar(list(range(len(y_t_dict))), y_t_dict)
                    # ax2.bar(list(range(len(y_e_dict))), y_e_dict)
                    F = plt.figure()
                    ax = F.add_subplot(111)
                    tr = ax.bar(np.arange(len(y_t_dict)) -0.2, y_t_dict, width=0.4, color='tab:red', log=True)
                    ev = ax.bar(np.arange(len(y_e_dict)) +0.2, y_e_dict, width=0.4, color='tab:blue', log=True)
                    ax.legend((tr[0], ev[0]), ('trian', 'eval'))
                    ax.set_ylabel('exp', color='tab:blue')
                    for i, v in enumerate(y_t_dict): ax.text(i - 0.2, v + 3, str(v), color='red', fontweight='bold')
                    for i, v in enumerate(y_e_dict): ax.text(i + 0.2, v + 3, str(v), color='blue', fontweight='bold')
                    tensorboardwriter.write_data_distribution(self.writer, F, fold)

                # write_cv_distribution(self.writer, y_t, y_e)
            np.save(config.DIRECTORY_SPLIT, fold_dict)
        else:
            fold_dict = np.load(config.DIRECTORY_SPLIT)
            pbar = tqdm(fold_dict)
            for fold, items in enumerate(pbar):
                pbar.set_description_str("Creating Folds from Dictionary")
                x_t = items[0]
                x_e = items[1]

                folded_samplers[fold] = dict()
                folded_samplers[fold]["train"] = SubsetRandomSampler(x_t)
                folded_samplers[fold]["val"] = SubsetRandomSampler(x_e)

            # gc.collect()
        return folded_samplers

    def get_fold_samplers(self, fold=-1):

        data = self.indices[:-(self.id_len % fold)]
        left_over = self.indices[-(self.id_len % fold):]
        cv_size = (len(self.indices) - len(left_over)) / fold

        print("     CV_size: {}".format(cv_size))
        print("     Fold: {}".format(fold))

        folded_train_indice = dict()
        folded_val_indice = dict()
        folded_samplers = dict()
        for i in range(fold):
            folded_val_indice[i] = list(set(data[i * cv_size:(i + 1) * cv_size]))
            folded_train_indice[i] = list(set(data[:]) - set(folded_val_indice[i]))
            print("     Fold#{}_train_size: {}".format(i, len(folded_train_indice[i])))
            print("     Fold#{}_val_size: {} + {}".format(i, len(folded_val_indice[i]), len(left_over)))
            folded_samplers[i] = {}
            folded_samplers[i]["train"] = SubsetRandomSampler(folded_train_indice[i])
            folded_samplers[i]["val"] = SubsetRandomSampler(folded_val_indice[i] + left_over)

        return folded_samplers

    def __getitem__(self, indice):
        """

        :param indice:
        :return: id, one hot encoded label, nparray image of (r, g, b, y) from 0~255 (['red', 'green', 'blue', 'yellow']) (4, W, H)
        """
        return (self.indices_to_id[indice], self.get_load_image_by_indice(indice), self.get_load_label_by_indice(indice))

    def get_load_image_by_indice(self, indice):
        id = self.indices_to_id[indice]
        return self.get_load_image_by_id(id)

    def get_load_image_by_id(self, id):
        return np.load(id)

    def get_load_label_by_indice(self, indice):
        """

        :param indice: id
        :return: one hot encoded label
        """
        if len(self.labelframe) - 1 < indice: return None
        return np.float32(self.labelframe[indice])

    def get_load_label_by_id(self, id):
        """

        :param indice: id
        :return: one hot encoded label
        """
        return np.float32(self.labelframe[self.id_to_indices[id]])

def train_aug(term):
    if config.epoch <= config.MODEL_FREEZE_EPOCH +2:
        return Compose([
        HorizontalFlip(p=term % 2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.9, 0.3), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),

        # OneOf([CLAHE(clip_limit=2),
        #        IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
        #        IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
        #        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        #        JpegCompression(quality_lower=99, quality_upper=100),
        #        Blur(blur_limit=2),
        #        GaussNoise()], p=0.5),
        RandomGamma(gamma_limit=(90, 110), p=0.8),
        # AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            RandomCrop(config.AUGMENTATION_RESIZE+80, config.AUGMENTATION_RESIZE+80),
            AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
            # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
            DoNothing(p=1),
        ], p=1),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),
    ])
    elif config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.8, 0.5), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            OneOf([CLAHE(clip_limit=2),
                   IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
                   IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
                   RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                   JpegCompression(quality_lower=99, quality_upper=100),
                   Blur(blur_limit=2),
                   GaussNoise()], p=0.8),
            RandomGamma(gamma_limit=(90, 110), p=0.8),
            OneOf([
                RandomCrop(config.AUGMENTATION_RESIZE+80, config.AUGMENTATION_RESIZE+80),
                AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
                # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else:
        return Compose([
            HorizontalFlip(p=term % 2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.8, 0.5), rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            OneOf([CLAHE(clip_limit=2),
                   IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.)),
                   IAAEmboss(alpha=(0.1, 0.2), strength=(0.2, 0.7)),
                   RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                   JpegCompression(quality_lower=99, quality_upper=100),
                   Blur(blur_limit=2),
                   GaussNoise()], p=0.8),
            RandomGamma(gamma_limit=(90, 110), p=0.8),
            OneOf([
                RandomCrop(config.AUGMENTATION_RESIZE+80, config.AUGMENTATION_RESIZE+80),
                AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
                # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
def eval_aug(term):
    if config.epoch > config.AUGMENTATION_RESIZE_CHANGE_EPOCH:
        return Compose([
            HorizontalFlip(p=term % 2),
            OneOf([
                RandomCrop(config.AUGMENTATION_RESIZE+80, config.AUGMENTATION_RESIZE+80),
                AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
                # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
                DoNothing(p=1),
            ], p=1),
            Resize(config.AUGMENTATION_RESIZE_CHANGE, config.AUGMENTATION_RESIZE_CHANGE, interpolation=cv2.INTER_CUBIC),  # 1344
        ])
    else: return Compose([
        HorizontalFlip(p=term % 2),
        OneOf([
            RandomCrop(config.AUGMENTATION_RESIZE+80, config.AUGMENTATION_RESIZE+80),
            AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),
            # Compose([AdaptivePadIfNeeded(border_mode=cv2.BORDER_CONSTANT),], p=1),
            DoNothing(p=1),
        ], p=1),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),  # 1344
    ])
def test_aug(term):
    return Compose([
        HorizontalFlip(p=term % 2),
        PadIfNeeded(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, border_mode=cv2.BORDER_CONSTANT),
        Resize(config.AUGMENTATION_RESIZE, config.AUGMENTATION_RESIZE, interpolation=cv2.INTER_CUBIC),
    ])
def tta_aug(term):
    return train_aug(term)
def train_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        if config.global_steps[config.fold] ==1: print(id, image_0.shape, labels_0.shape)
        new_batch.append(transform(id, image_0, labels_0, train=True, val=False))
    batch = new_batch
    return collate(batch)


def val_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=True))
    batch = new_batch
    return collate(batch)

def test_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=False, val=False))
    batch = new_batch
    return collate(batch)

def tta_collate(batch):
    """TRASNFORM"""
    new_batch = []
    for id, image_0, labels_0 in batch:
        new_batch.append(transform(id, image_0, labels_0, train=True, val=True))
    batch = new_batch
    return collate(batch)

def collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def transform(ids, image_0, labels_0, train, val):
    """

    :param ids:
    :param image_0:
    :param labels_0:
    :param train:
    :param val:
    :return:
    """

    REGULARIZATION_TRAINSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: cv2.resize(x,(config.AUGMENTATION_RESIZE,config.AUGMENTATION_RESIZE), interpolation=cv2.INTER_CUBIC),
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
        ])

    if not train and not val:
        term = config.eval_index % 8
        TEST_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: test_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TEST_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)

    """ https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
    Cubic interpolation is computationally more complex, and hence slower than linear interpolation. However, the quality of the resulting image will be higher.
    """
    if not val and train:
        term = config.epoch % 8
        TRAIN_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: train_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TRAIN_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
    elif not train and val:
        term = config.eval_index % 8
        PREDICT_TRANSFORM_IMG = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
            lambda x: eval_aug(term)(image=x),
            lambda x: x['image'],
            lambda x: np.clip(x, a_min=0, a_max=255),
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD),
        ])
        image = PREDICT_TRANSFORM_IMG(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)
    elif train and val:
        term = config.eval_index % 8
        TTA_TRANSFORM = transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), # and don't put them in strong_aug()
            lambda x: tta_aug(term)(image=x), # Yes, you have to use image=xxx
            lambda x: x['image'], # abstract the actual image after the augmentation
            lambda x: np.clip(x, a_min=0, a_max=255), # make the image within the range
            transforms.ToTensor(),
            # Normalize(mean=config.AUGMENTATION_MEAN, std=config.AUGMENTATION_STD), # this needs to be set accordingly
        ])
        image = TTA_TRANSFORM(image_0)
        image_0 = REGULARIZATION_TRAINSFORM(image_0)
        if config.global_steps[config.fold] == 1: print(ids.shape, image.shape, labels_0.shape, image_0.shape)
        return (ids, image, labels_0, image_0)

# class StratifiedRandomSampler(object):
#     r"""Samples elements randomly from a given list of indices, without replacement.
#
#     Arguments:
#         indices (sequence): a sequence of indices
#     """
#
#     def __init__(self, indices, labels):
#         self.indices = indices
#         self.labels = labels
#
#         mskf = MultilabelStratifiedKFold(n_splits=go    , random_state=None)
#         for fold, (train_index, test_index) in enumerate(mskf.split(X, y)):
#             print("#{} TRAIN:{} TEST:{}".format(fold, train_index, test_index))
#             x_t = train_index
#             # y_t = np.array([y[j] for j in train_index])
#             x_e = test_index
#             # y_e = np.array([y[j] for j in test_index])
#
#     def __iter__(self):
#         return (self.indices[i] for i in torch.randperm(len(self.indices)))
#
#     def __len__(self):
#         return len(self.indices)