import subprocess
import os
import numpy as np
import math
from openpyxl import load_workbook
from csv import reader

EDF_PREFIX = "cfs/polysomnography/edfs/cfs-visit5-"
EDF_SUFFIX = ".edf"

XML_PREFIX = "cfs/polysomnography/annotations-events-nsrr/cfs-visit5-"
XML_SUFFIX = "-nsrr.xml"

OUT_PREFIX = "debug/"
OUT_SUFFIX = ".out"

FEATURES_PREFIX = "data/mnc/features/"
LABELS_PREFIX = "data/mnc/labels/"

PSD = ["SLOW", "DELTA", "THETA", "ALPHA", "SIGMA", "BETA", "GAMMA"]

EPOCH_LENGTH = 30

ERROR = False

# ids_file = test file list of ids
# token = string token from nsrr
def run(ids_file, token):
    ids_file = open(ids_file)
    ids = []
    # for line in ids_file:
    #     ids.append(line.replace('\n', ''))

    file = reader(ids_file)
    for row in file:
        if row[0] == 'ID':
            continue
        elif row[1] == 'CNC':
            ids.append('mnc/cnc/' + row[0].lower() + '-nsrr')
        elif row[1] == 'SSC':
            ids.append('mnc/ssc/' + row[0].lower() + '-nsrr')
        elif row[3] == 1:
            ids.append('mnc/dhc/training/' + row[0].lower() + '-nsrr')
        elif row[4] == 1:
            ids.append('mnc/dhc/test/control' + row[0].lower() + '-nsrr')
        elif row[5] == 1:
            ids.append('mnc/dhc/test/nc-lh' + row[0].lower() + '-nsrr')
        elif row[6] == 1:
            ids.append('mnc/dhc/test/nc-nh' + row[0].lower() + '-nsrr')



    for id in ids:
        global ERROR
        ERROR = False
        edf = id + EDF_SUFFIX
        xml = id + XML_SUFFIX
        out = OUT_PREFIX + id + OUT_SUFFIX
        out = open(out, 'w')
        result = subprocess.run('nsrr ' + 'download ' + edf + " " + token + ' --fast', shell = True, stdout = out)
        if result.returncode != 0:
            out.write("Error downloading " + edf)
            ERROR = True
            continue
        result = subprocess.run('nsrr ' + 'download ' + xml + " " + token + ' --fast', shell = True, stdout = out)
        if result.returncode != 0:
            out.write("Error downloading " + xml)
            ERROR = True
            continue

        process(id, edf, xml, out)
        cleanup(id, edf, xml, out)



def cleanup(id, edf, xml, out):
    global ERROR
    try:
        os.remove(edf)
    except FileNotFoundError:
        out.write("No edf to delete " + edf)
    try:
        os.remove(xml)
    except FileNotFoundError:
        out.write("No xml to delete " + xml)

    if ERROR:
        out.close()
        os.rename(OUT_PREFIX + id + OUT_SUFFIX, OUT_PREFIX + id + "ERROR" + OUT_SUFFIX)

    print("--- " + id + " DONE --- ERROR=" + str(ERROR))

# lstm input: [batch, timesteps, feature]

# features: [SNORE-PSD, THOR-PSD, SpO2-STATS, PULSE-STATS, Light-STATS]
# PSD: [SLOW, DELTA, THETA, ALPHA, SIGMA, BETA, GAMMA]
# STATS: [MAX, MEAN, MEDIAN, MIN, RMS]
def process(id, edf, xml, out):
    global ERROR
    arr = []
    signals = "sig=SNORE,THOR_EFFORT,SpO2,SaO2,PULSE"
    data = "psg.db"
    result = subprocess.run('luna ' + edf + " " + signals + ' -o ' + data + ' < ' + 'commands.txt', shell = True, stdout = None)
    if result.returncode != 0:
        out.write("Error Luna " + edf)
        ERROR = True
        return

    if not ERROR:
        epochs = retrieve_epochs(data, out)
        print("---------------------------epochs")
        print(epochs)
    for i in range(epochs):
        arr.append([])

    if not ERROR:
        retrieve_psd('SNORE', arr, data, out)
    if not ERROR:
        retrieve_psd('THOR_EFFORT', arr, data, out)
    if not ERROR:
        retrieve_stats('SpO2', arr, data, out)
    if not ERROR:
        retrieve_stats('SaO2', arr, data, out)
    if not ERROR:
        retrieve_stats('PULSE', arr, data, out)
    if not ERROR:
        sudden_changes(arr)


    arr = np.array(arr)
    np.save(FEATURES_PREFIX + id + "features", arr)

    labels = np.zeros(epochs)
    if not ERROR:
        retrieve_labels(labels, xml, out)
    np.save(LABELS_PREFIX + id + "labels", labels)


def retrieve_labels(labels, xml, out):
    global ERROR
    temp = open("TEMP", "w+")
    result = subprocess.run('luna --xml ' + xml + ' | find "Wake"', shell = True, stdout = temp)
    temp.seek(0)
    if result.returncode != 0:
        out.write("Error Reading XML")
        ERROR = True
        return
    for line in temp:
        list = line.split()
        start = int(list[0]) // EPOCH_LENGTH
        end = int(list[2]) // EPOCH_LENGTH
        for i in range(start, end):
            labels[i] = 1


def retrieve_epochs(data, out):
    global ERROR
    temp = open("TEMP", "w+")
    result = subprocess.run(['destrat', data, '+EPOCH'], stdout = temp)
    temp.seek(0)
    if result.returncode != 0:
        out.write("Error Destrat Epoch")
        ERROR = True
        return
    first = True
    for line in temp:
        if (first):
            first = False
        else:
            return int(line.split()[3])

def retrieve_psd(signal, arr, data, out):
    global ERROR
    temp = open("TEMP", "w+")
    for band in PSD:
        temp.seek(0)
        result = subprocess.run(['destrat', data, '+PSD', '-r', 'CH/' + signal, 'B/' + band, 'E'], stdout = temp)
        if result.returncode != 0:
            out.write("Error Destrat " + signal + band)
            ERROR = True
            return
        temp.seek(0)
        for line in temp:
            break
        for line, a in zip(temp, arr):
            a.append(np.float32(float(line.split()[4])))

def retrieve_stats(signal, arr, data, out):
    global ERROR
    temp = open("TEMP", "w+")
    temp.seek(0)
    result = subprocess.run(['destrat', data, '+STATS', '-r', 'CH/' + signal, 'E'], stdout = temp)
    if result.returncode != 0:
        out.write("Error Destrat " + signal)
        ERROR = True
        return
    temp.seek(0)
    for line in temp:
        print(line)
        break
    for line, a in zip(temp, arr):
        list = line.split()
        for i in range(3, 8):
            a.append(np.float32(float(list[i])))

F1_RANGE = 59
F2_RANGE = 4
F3_RANGE = 4
F3N = 9

def sudden_changes(arr):
    np_arr = np.array(arr)
    length = np_arr.shape[0]
    for i, a in enumerate(arr):
        f1_min = max(0, i - F1_RANGE)
        f1_max = min(length, i + F1_RANGE + 1)
        f1_arr = np_arr[f1_min : f1_max]
        f1_arr = np.average(f1_arr, 0)

        f2_min = max(0, i - F2_RANGE)
        f2_max = min(length, i + F2_RANGE + 1)
        f2_arr = np_arr[f2_min : f2_max]
        f2_arr = np.median(f2_arr, 0)

        f3_min = max(0, i - F3_RANGE)
        f3_max = min(length, i + F3_RANGE + 1)
        f3_arr = np_arr[f3_min : f3_max]
        f3_vals = np.average(f3_arr, 0)

        for j in range(len(a)):
            e = a[j]
            f1 = e - f1_arr[j]
            f2 = e - f2_arr[j]
            f3 = 0
            for a2 in f3_arr:
                val = a2[j] - f3_vals[j]
                val = val ** 2
                f3 += val
            f3 /= f3_arr.shape[0]
            f3 = math.sqrt(f3)

            a.append(np.float32(f1))
            a.append(np.float32(f2))
            a.append(np.float32(f3))
