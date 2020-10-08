import subprocess
import os
import numpy as np

EDF_PREFIX = "cfs/polysomnography/edfs/cfs-visit5-"
EDF_SUFFIX = ".edf"

XML_PREFIX = "cfs/polysomnography/annotations-events-nsrr/cfs-visit5-"
XML_SUFFIX = "-nsrr.xml"

OUT_PREFIX = "debug/"
OUT_SUFFIX = ".out"

FEATURES_PREFIX = "data/features/"
LABELS_PREFIX = "data/labels/"

PSD = ["SLOW", "DELTA", "THETA", "ALPHA", "SIGMA", "BETA", "GAMMA"]

EPOCH_LENGTH = 5

ERROR = False

# ids_file = test file list of ids
# token = string token from nsrr
def run(ids_file, token):
    ids_file = open(ids_file)
    ids = []
    for line in ids_file:
        ids.append(line.replace('\n', ''))

    for id in ids:
        global ERROR
        ERROR = False
        edf = EDF_PREFIX + id + EDF_SUFFIX
        xml = XML_PREFIX + id + XML_SUFFIX
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
            a.append(float32(line.split()[4]))

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
            a.append(float32(list[i]))
