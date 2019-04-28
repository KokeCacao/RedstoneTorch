import os
import numpy as np

import config
from utils.memory import write_memory
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('WARNING: No display found. Using non-interactive Agg backend for loading matplotlib.')
    mpl.use('Agg')
from matplotlib import pyplot as plt

write_memory = write_memory


def write_text(writer, text, step):
    writer.add_text('text', text, global_step=step, walltime=None)


def write_loss(writer, loss_dict, global_step):
    writer.add_scalars('train/loss_scalar', loss_dict, global_step)


def write_threshold(writer, classes, score, threshold, fold):
    writer.add_scalars('threshold/threshold_distribution', {"Class/{}".format(classes): score}, threshold)

def write_threshold_class(writer, best_threshold_dict, best_val_dict, class_list):
    dic = {813: 19970, 1092: 14281, 147: 13522, 189: 10375, 13: 9151, 671: 8419, 51: 7615, 194: 7394, 1059: 6564, 121: 6542, 896: 5955, 1046: 5591, 79: 5382, 780: 5259, 156: 5163, 369: 4416, 744: 3890, 477: 3692, 738: 3665, 1034: 3570, 188: 3500, 835: 3005, 903: 2552, 420: 2548, 1099: 2327, 552: 2180, 485: 2097, 776: 2075, 161: 2050, 489: 2045, 1039: 2001, 733: 1895, 304: 1881, 612: 1789, 111: 1762, 962: 1744, 487: 1685, 501: 1667, 1062: 1540, 961: 1526, 541: 1492, 734: 1480, 483: 1472, 405: 1457, 737: 1446, 597: 1428, 480: 1414, 335: 1403, 718: 1397, 554: 1390, 99: 1327, 259: 1302, 663: 1286, 584: 1283, 616: 1278, 418: 1260, 434: 1213, 768: 1200, 833: 1138, 724: 1083, 993: 1073, 949: 1053, 626: 1050, 784: 1016, 1020: 977, 698: 965, 1084: 961, 492: 959, 830: 957, 872: 956, 754: 921, 800: 902, 538: 901, 433: 888, 542: 866, 464: 859, 212: 838, 70: 831, 1072: 816, 283: 810, 615: 810, 131: 787, 1035: 775, 650: 773, 498: 766, 639: 750, 579: 745, 1061: 744, 955: 682, 25: 676, 404: 676, 559: 672, 796: 637, 430: 636, 655: 630, 975: 627, 1093: 622, 519: 622, 624: 613, 670: 600, 231: 595, 858: 593, 180: 593, 716: 583, 758: 582, 436: 582, 182: 580, 668: 570, 950: 568, 764: 566, 415: 565, 494: 564, 746: 560, 580: 555, 634: 551, 184: 545, 886: 543, 532: 540, 29: 530, 378: 523, 923: 518, 742: 515, 45: 515, 844: 504, 127: 499, 125: 494, 922: 493, 1098: 487, 543: 481, 598: 480, 931: 477, 313: 467, 704: 462, 753: 461, 1023: 454, 747: 453, 586: 446, 603: 440, 720: 433, 583: 427, 952: 422, 774: 422, 953: 413, 14: 408, 227: 403, 682: 399, 821: 393, 339: 388, 582: 387, 974: 382, 428: 373, 1022: 371, 703: 369, 954: 369, 308: 368, 239: 362, 684: 358, 465: 357, 809: 355, 520: 354, 462: 338, 573: 332, 581: 332, 940: 329, 558: 329, 348: 328, 592: 327, 545: 327, 438: 323, 725: 322, 322: 317, 925: 312, 1019: 303, 633: 303, 497: 302, 493: 298, 665: 297, 595: 296, 506: 295, 331: 293, 33: 292, 916: 290, 587: 289, 745: 286, 503: 284, 236: 281, 1032: 273, 515: 273, 568: 270, 803: 269, 202: 269, 647: 266, 766: 265, 517: 263, 991: 262, 783: 262, 978: 258, 707: 257, 691: 257, 553: 256, 726: 256, 490: 254, 432: 254, 133: 254, 994: 253, 488: 253, 1089: 251, 1030: 251, 23: 244, 534: 244, 524: 243, 607: 240, 58: 240, 840: 239, 1064: 236, 637: 236, 786: 234, 282: 233, 819: 233, 690: 230, 728: 229, 1085: 228, 884: 227, 450: 226, 834: 225, 822: 224, 908: 223, 482: 221, 713: 221, 1045: 220, 332: 219, 445: 219, 695: 219, 756: 218, 971: 216, 563: 216, 864: 215, 354: 210, 721: 208, 252: 207, 1012: 207, 890: 206, 570: 205, 739: 205, 518: 205, 889: 204, 701: 204, 672: 203, 705: 202, 929: 200, 678: 196, 1056: 195, 832: 194, 1009: 193, 983: 188, 495: 186, 628: 186, 548: 185, 927: 184, 811: 182, 963: 181, 555: 181, 1038: 180, 910: 179, 870: 179, 26: 177, 631: 177, 675: 176, 988: 174, 992: 173, 924: 170, 620: 170, 426: 169, 688: 167, 871: 167, 621: 165, 1040: 162, 731: 160, 1001: 160, 638: 160, 973: 159, 1097: 158, 306: 157, 989: 156, 692: 156, 849: 155, 826: 153, 522: 153, 280: 152, 551: 150, 676: 150, 627: 150, 996: 148, 972: 148, 383: 148, 791: 147, 912: 147, 932: 147, 591: 146, 78: 146, 863: 144, 1081: 143, 1069: 143, 425: 142, 41: 141, 888: 141, 448: 140, 887: 139, 1096: 137, 985: 137, 508: 137, 1033: 137, 576: 135, 76: 133, 680: 133, 496: 131, 825: 130, 829: 130, 915: 129, 741: 128, 116: 127, 511: 127, 1058: 126, 732: 124, 629: 124, 600: 124, 1037: 124, 507: 123, 666: 123, 18: 121, 535: 121, 669: 120, 1068: 120, 708: 120, 999: 120, 749: 119, 948: 119, 35: 119, 622: 119, 359: 118, 393: 118, 179: 118, 619: 118, 510: 117, 204: 116, 413: 115, 654: 115, 566: 114, 709: 114, 217: 114, 1083: 114, 454: 114, 1071: 113, 947: 113, 794: 113, 642: 112, 750: 112, 347: 112, 681: 111, 273: 111, 468: 111, 808: 111, 907: 110, 1065: 110, 516: 110, 649: 108, 192: 108, 882: 107, 55: 106, 479: 106, 422: 106, 861: 106, 459: 104, 1018: 104, 685: 104, 755: 103, 771: 103, 662: 103, 502: 102, 1066: 100, 1: 100, 546: 99, 674: 99, 540: 98, 1024: 98, 529: 97, 966: 97, 967: 97, 234: 97, 960: 97, 823: 97, 995: 96, 939: 96, 959: 96, 287: 96, 400: 95, 505: 95, 185: 95, 1025: 95, 1051: 95, 866: 94, 921: 94, 860: 94, 135: 94, 693: 93, 645: 93, 253: 93, 138: 93, 285: 93, 536: 92, 898: 92, 1080: 92, 585: 92, 763: 91, 409: 91, 608: 90, 664: 90, 316: 90, 406: 89, 175: 88, 850: 88, 679: 88, 761: 88, 661: 87, 653: 87, 941: 87, 838: 86, 449: 86, 537: 86, 1055: 85, 567: 85, 1101: 85, 714: 85, 556: 84, 1015: 83, 258: 82, 43: 82, 334: 81, 513: 81, 1010: 81, 307: 81, 936: 81, 457: 81, 874: 80, 885: 80, 105: 80, 512: 80, 478: 79, 1036: 79, 942: 79, 836: 79, 804: 79, 867: 79, 466: 79, 846: 78, 596: 78, 1077: 78, 368: 77, 875: 77, 453: 77, 920: 77, 815: 76, 461: 76, 226: 76, 458: 75, 588: 75, 824: 74, 60: 74, 702: 73, 1091: 72, 32: 72, 486: 72, 65: 72, 102: 72, 410: 71, 613: 71, 275: 71, 232: 71, 61: 71, 528: 70, 521: 70, 816: 70, 155: 69, 191: 69, 842: 69, 735: 69, 618: 69, 1013: 68, 1007: 68, 463: 68, 700: 68, 390: 68, 879: 67, 965: 67, 565: 67, 795: 67, 648: 67, 1003: 67, 906: 66, 1053: 66, 710: 66, 905: 66, 1041: 66, 881: 66, 437: 66, 444: 66, 373: 66, 964: 65, 968: 65, 895: 64, 914: 64, 1027: 64, 769: 64, 345: 64, 154: 64, 210: 63, 439: 63, 338: 63, 696: 63, 248: 63, 611: 63, 827: 62, 564: 62, 1088: 62, 218: 62, 1095: 62, 557: 61, 294: 61, 96: 61, 956: 61, 610: 61, 986: 60, 351: 60, 767: 60, 113: 60, 289: 60, 384: 59, 699: 59, 862: 59, 91: 59, 1002: 59, 891: 59, 1047: 59, 593: 58, 412: 58, 901: 58, 606: 58, 447: 58, 657: 58, 114: 58, 899: 58, 1050: 57, 1044: 57, 667: 57, 723: 57, 793: 57, 577: 57, 807: 56, 590: 56, 276: 56, 817: 56, 269: 56, 549: 56, 913: 56, 162: 55, 1054: 55, 706: 55, 1005: 55, 937: 54, 578: 54, 677: 53, 1073: 53, 423: 53, 806: 53, 163: 53, 646: 52, 145: 52, 173: 52, 1004: 52, 323: 52, 911: 52, 604: 52, 1078: 52, 514: 52, 660: 51, 828: 51, 934: 51, 878: 51, 605: 51, 976: 51, 623: 51, 977: 51, 641: 50, 748: 50, 349: 50, 42: 50, 1006: 50, 569: 50, 868: 49, 1057: 49, 772: 49, 712: 49, 673: 48, 399: 48, 40: 48, 533: 48, 1063: 48, 424: 48, 1043: 48, 797: 48, 256: 48, 951: 48, 1086: 48, 711: 47, 760: 47, 574: 47, 446: 47, 467: 47, 820: 47, 318: 46, 782: 46, 876: 46, 778: 46, 571: 46, 998: 45, 831: 45, 865: 45, 110: 45, 562: 45, 321: 45, 1049: 45, 683: 44, 757: 44, 86: 44, 196: 44, 375: 44, 263: 43, 770: 43, 274: 43, 69: 43, 473: 43, 95: 43, 1026: 42, 902: 42, 401: 41, 938: 41, 451: 41, 427: 41, 969: 41, 24: 41, 470: 41, 107: 41, 360: 40, 309: 40, 124: 40, 818: 40, 484: 40, 773: 40, 1074: 40, 272: 39, 572: 39, 324: 39, 560: 39, 945: 39, 636: 39, 178: 38, 279: 38, 1000: 38, 237: 38, 946: 38, 144: 38, 686: 38, 775: 38, 315: 37, 740: 37, 848: 37, 526: 37, 149: 37, 926: 37, 729: 36, 195: 36, 411: 36, 980: 36, 421: 36, 792: 36, 158: 36, 984: 36, 935: 36, 614: 36, 979: 36, 530: 35, 441: 35, 251: 35, 656: 35, 429: 35, 382: 35, 44: 35, 722: 35, 759: 34, 48: 34, 1075: 34, 442: 34, 1070: 34, 715: 34, 15: 34, 777: 34, 659: 34, 475: 34, 1082: 33, 292: 33, 57: 33, 249: 33, 789: 32, 385: 32, 257: 32, 342: 32, 788: 32, 847: 32, 134: 32, 122: 32, 893: 31, 408: 31, 651: 31, 1102: 31, 539: 31, 90: 30, 299: 30, 531: 30, 265: 30, 1100: 30, 730: 30, 205: 30, 990: 30, 509: 30, 958: 30, 894: 30, 141: 29, 997: 29, 358: 29, 397: 29, 900: 29, 74: 29, 1014: 29, 500: 29, 50: 29, 630: 28, 270: 28, 617: 28, 419: 28, 83: 28, 1042: 28, 499: 28, 957: 28, 9: 28, 877: 27, 837: 27, 717: 27, 609: 27, 839: 27, 320: 27, 547: 27, 169: 27, 440: 26, 151: 26, 933: 26, 109: 26, 1016: 26, 326: 26, 471: 26, 472: 26, 165: 26, 640: 26, 751: 26, 658: 26, 762: 25, 387: 25, 1079: 25, 469: 25, 106: 25, 504: 25, 859: 25, 801: 24, 550: 24, 602: 24, 625: 24, 810: 24, 455: 24, 869: 24, 75: 24, 802: 24, 689: 23, 120: 23, 1008: 23, 814: 23, 325: 23, 171: 23, 150: 22, 67: 22, 398: 22, 402: 22, 59: 22, 743: 22, 785: 22, 909: 22, 341: 22, 841: 22, 880: 22, 245: 22, 93: 22, 22: 22, 222: 22, 694: 22, 1067: 22, 374: 22, 981: 22, 857: 21, 1029: 21, 170: 21, 247: 21, 601: 21, 417: 21, 229: 21, 632: 21, 216: 21, 350: 21, 1011: 21, 456: 21, 380: 21, 781: 20, 1028: 20, 379: 20, 943: 20, 736: 20, 851: 20, 928: 20, 765: 20, 17: 20, 403: 19, 362: 19, 852: 19, 414: 19, 255: 19, 136: 19, 98: 19, 589: 19, 377: 19, 435: 19, 416: 19, 89: 19, 407: 18, 223: 18, 153: 18, 0: 18, 944: 18, 970: 18, 371: 18, 687: 18, 337: 18, 481: 18, 1090: 17, 266: 17, 52: 17, 1021: 17, 525: 17, 261: 17, 336: 17, 117: 17, 790: 17, 228: 17, 5: 17, 344: 17, 982: 17, 644: 17, 317: 16, 357: 16, 1087: 16, 353: 16, 575: 16, 491: 16, 856: 16, 301: 16, 392: 16, 172: 16, 85: 16, 799: 16, 361: 15, 300: 15, 208: 15, 930: 15, 319: 15, 443: 14, 139: 14, 1076: 14, 152: 14, 2: 14, 594: 14, 853: 14, 157: 14, 8: 14, 77: 14, 12: 14, 267: 14, 238: 14, 63: 13, 64: 13, 27: 13, 1052: 13, 1048: 13, 330: 13, 97: 13, 213: 13, 352: 13, 4: 13, 49: 12, 244: 12, 72: 12, 148: 12, 1094: 12, 697: 12, 918: 12, 28: 12, 16: 12, 356: 12, 246: 12, 779: 12, 295: 11, 386: 11, 62: 11, 1031: 11, 206: 11, 897: 11, 474: 11, 66: 11, 46: 11, 243: 10, 225: 10, 719: 10, 209: 10, 343: 10, 54: 10, 87: 10, 220: 10, 10: 10, 56: 10, 310: 10, 164: 10, 752: 10, 302: 10, 38: 10, 277: 10, 919: 10, 143: 10, 296: 9, 854: 9, 452: 9, 303: 9, 523: 9, 643: 9, 476: 9, 193: 9, 200: 9, 47: 9, 305: 9, 183: 9, 798: 9, 288: 9, 395: 8, 599: 8, 174: 8, 812: 8, 168: 8, 84: 8, 652: 8, 381: 8, 460: 8, 254: 8, 394: 8, 561: 8, 215: 8, 68: 8, 364: 7, 37: 7, 346: 7, 101: 7, 333: 7, 224: 7, 370: 7, 845: 7, 355: 7, 635: 7, 917: 7, 544: 7, 892: 7, 119: 7, 340: 7, 159: 7, 73: 7, 128: 7, 235: 7, 241: 7, 19: 7, 242: 7, 197: 7, 181: 6, 264: 6, 219: 6, 118: 6, 311: 6, 388: 6, 527: 6, 137: 6, 363: 6, 314: 6, 186: 6, 130: 6, 883: 6, 167: 6, 843: 6, 82: 6, 391: 6, 286: 6, 126: 6, 431: 5, 80: 5, 31: 5, 21: 5, 268: 5, 211: 5, 365: 5, 298: 5, 873: 5, 297: 5, 207: 5, 30: 5, 1060: 5, 115: 5, 284: 4, 327: 4, 88: 4, 132: 4, 855: 4, 20: 4, 214: 4, 176: 4, 92: 4, 34: 4, 260: 4, 166: 4, 290: 4, 177: 4, 291: 4, 6: 4, 278: 4, 129: 4, 233: 4, 787: 4, 103: 4, 367: 4, 198: 3, 36: 3, 727: 3, 53: 3, 100: 3, 7: 3, 1017: 3, 39: 3, 140: 3, 203: 3, 250: 3, 329: 3, 372: 3, 376: 3, 3: 3, 190: 3, 123: 3, 160: 3, 94: 3, 71: 2, 187: 2, 312: 2, 108: 2, 201: 2, 271: 2, 240: 2, 987: 2, 904: 2, 142: 2, 389: 2, 396: 1, 262: 1, 328: 1, 230: 1, 104: 1, 281: 1, 805: 1, 199: 1, 146: 1, 112: 1, 221: 1, 293: 1, 11: 1, 81: 1, 366: 1}
    current_freq = -1
    current_thres = 0
    current_val = 0
    current_count = 0
    freq = -1

    for c in class_list:
        freq = dic[c]
        thres = best_threshold_dict[c]
        val = best_val_dict[c]
        if current_freq == freq:
            current_count = current_count + 1
            current_thres = current_thres + thres
            current_val = current_val + val
        else:
            if current_count != 0: writer.add_scalars('threshold/threshold_frequency', {"Threshold": current_thres/current_count, "Validation": current_val/current_count}, freq)

            current_freq = freq
            current_thres = thres
            current_val = val
            current_count = 1
    writer.add_scalars('threshold/threshold_frequency', {"Threshold": current_thres/current_count, "Validation": current_val/current_count}, freq)


def write_find_lr(writer, loss, lr):
    writer.add_scalars('threshold/find_lr/', {"LRx100000": loss}, int(lr*100000))


def write_best_threshold(writer, classes, score, threshold, area_under, epoch, fold):
    if classes == -1:
        writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Score/{}".format(classes): score,
                                                                        "Threshold/{}".format(classes): threshold,
                                                                        "AreaUnder/{}".format(classes): area_under}, epoch)
    else:
        writer.add_scalars('threshold/best_threshold/{}'.format(fold), {"Threshold/{}".format(classes): threshold}, epoch)


def write_data_distribution(writer, F, fold, unique=False):
    if unique:
        writer.add_figure("data/fold_distribution", F, 0)
        return
    writer.add_figure("data/fold_distribution/{}".format(fold), F, 0)

def write_shakeup(writer, dictionary, sorted_keys, std, epoch):
    for i, key in enumerate(sorted_keys):
        public_lb, private_lb = dictionary[key]
        writer.add_scalars('threshold/Shakeup/', {"Public LB": public_lb}, i)
        writer.add_scalars('threshold/Shakeup/', {"Private LB": private_lb}, i)
    writer.add_scalars('threshold/shake_up_change/', {"Standard Deviation": std}, epoch)
    try:
        writer.add_histogram("eval/loss_distribution", np.array(sorted_keys), epoch)
    except Exception as e:
        print("Having some trouble writing histogram: `writer.add_histogram(\"eval/loss_distribution\", sorted_keys, epoch)`")


def write_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/skfacc_distribution", loss_list, epoch)

def write_classwise_loss_distribution(writer, loss_list, epoch):
    writer.add_histogram("eval/classwise_skfacc_distribution", loss_list, epoch)


def write_pred_distribution(writer, pred_list, epoch):
    writer.add_histogram("eval/pred_distribution", pred_list, epoch)


def write_pr_curve(writer, label, predicted, epoch, fold):
    writer.add_pr_curve("eval/pr_curve/{}".format(fold), label, predicted, epoch)


def write_image(writer, msg, F, epoch):
    writer.add_figure("eval/image/{}".format(msg), F, 0)


def write_predict_image(writer, msg, F, epoch):
    writer.add_figure("predict/image/{}".format(msg), F, epoch)


def write_eval_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_scalar', loss_dict, epoch)


def write_epoch_loss(writer, loss_dict, epoch):
    writer.add_scalars('eval/loss_epoch', loss_dict, epoch)


def write_best_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id.split("/")[-1], label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)

def write_focus(writer, id, cam, img, label, pred, n, fold):
    F = plt.figure()
    plt.imshow(cam)
    plt.title("Id:{} label:{} pred:{}".format(id, label, pred))
    plt.grid(False)
    writer.add_figure("focus/#{}-{}-cam".format(n, fold), F, 0)

    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} pred:{}".format(id, label, pred))
    plt.grid(False)
    writer.add_figure("focus/#{}-{}-img".format(n, fold), F, 0)

def write_plot(writer, F, message):
    writer.add_figure("figure/{}".format(message), F, 0)

def write_worst_img(writer, img, label, id, loss, fold):
    F = plt.figure()
    plt.imshow(img)
    plt.title("Id:{} label:{} loss:{}".format(id.split("/")[-1], label, loss))
    plt.grid(False)
    writer.add_figure("worst/{}".format(fold), F, 0)

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
     return True
    elif data.ndim == 3:
     if 3 <= data.shape[2] <= 4:
      return True
     else:
      print('The "data" has 3 dimensions but the last dimension ' 
        'must have a length of 3 (RGB) or 4 (RGBA), not "{}".' 
        ''.format(data.shape[2]))
      return False
    else:
     print('To visualize an image the data must be 2 dimensional or ' 
       '3 dimensional, not "{}".' 
       ''.format(data.ndim))
     return False
