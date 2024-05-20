import csv
import os

# 處理espnet從downloads 到data下dev, train, test的wav.scp, text, utt2spk, spk2utt

"""
設計重點：
    * 訓練語料、音檔和文字的對應，主要是透過basename來達成
    * 從音檔檔案，可以推出：基檔名（對應文字）、語者（位置推導）
    * 建立音檔id 
basename: '1', '2', ...
utt_id: TR12345678, training utterance
        TS12345678, testing utterance 
spk_id: S001
主要資料：
    train-toneless.csv 
    text_ref: basename, text
    *.wav
    train_wav: utt_id, location, text, basename
    # version 2.0 train_wav: utt_id, file_path, text, basename, has_data
    test_wav: utt_id, location, text
    text_ref: 0,1 : 0, N train_wav_inf

    validation_num #用來作驗証數量：1/10
    create_files_dev
        從train_wav 取出對應區間
        create_wav_scp
            utt_id file_path
        create_text
            utt_id text
        create_utt2spk
            utt_id speaker_id
        create_spk2utt
            speaker_id utt_id
    create_files_train
        從train_wav 取出對應區間
    create_files_train
    create_files_test
處理訓練音檔
讀入音檔編號及對應文字
    檢查編號是否重覆
切換到train/train目錄
    針對每個檔案
        檔案編號是否存在
        產生utt-id
        產生檔案位置
        對應文字


處理測試音檔
"""
# 文字對應基檔名
text_ref = {}

#validation_num
validation_num = 312 # 3119 * 1/10

# wav_inf utt-id, {basename, speaker_id, text}
train_wav_inf = {}
test_wav_inf = {}


def create_text_ref(filename):
    """
    讀進train-toneless.csv
    basename, 基檔名
    ref_text 對照文字
    """
    # clear train_data
    text_ref.clear()
    with open(filename, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header
        next(csv_reader, None)
        for row in csv_reader:
            # 檢查紀錄是否存在
            basename = row[0]
            text = row[1]
            if basename in text_ref:
                print("duplicate key:" + basename)
                break
            text_ref[basename] = {}
            text_ref[basename]["text"] = text




def loop_thru_train_data():
    for key, value in text_ref.items():
        print("Key:", key, "Value:", value)

def get_speaker_id(file):
    #未來可能需要從檔案位置取得語者編號，目前只有一個語者
    return "S0001"

def get_ts_utt_id(id):
    return 'TS{:08d}'.format(id)

def get_tr_utt_id(id):
    return 'TR{:08d}'.format(id)

def get_train_wav_inf(directory):
# wav_inf utt-id, {basename, speaker_id, text}
# wav_inf = {}
    wav_inf = {}
    id = 1 # utt_id 編號
    for file in os.listdir(directory):
        #file 會是單獨的檔名，無目錄資訊
        if file.endswith(".wav"):
            basename, ext = os.path.splitext(file)
            text = text_ref[basename].get("text")
            if text is None: 
                print("training wav file has no text: filename: "+file)
                break # 訓練語料沒對應文字就叫停
            utt_id = get_tr_utt_id(id)
            id+=1
            wav_inf[utt_id] = {}
            wav_inf[utt_id]["text"] = text 
            wav_inf[utt_id]["basename"] = basename 
            wav_inf[utt_id]["location"] = os.path.join(directory, file)
            wav_inf[utt_id]["speaker_id"] = get_speaker_id(file)
    return wav_inf





def proc_train_wav(wav_dir):
    # 未來可能會有多個目錄
    return get_train_wav_inf(wav_dir)


def proc_test_wav(wav_dir):
    # 未來可能會有多個目錄
    return get_test_wav_inf(wav_dir)

def get_test_wav_inf(directory):
# wav_inf utt-id, {basename, speaker_id, text}
    wav_inf = {}

    id = 1 # utt_id 編號
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            utt_id = get_ts_utt_id(id)
            # utt_id, ext = os.path.splitext(file)
            id+=1
            wav_inf[utt_id] = {}
            wav_inf[utt_id]["location"] = os.path.join(directory, file)
            wav_inf[utt_id]["speaker_id"] = get_speaker_id(file)

    return wav_inf

def del_data_files(dir):
    for file in ["wav.scp","text","utt2spk","spk2utt"]:
        file = dir+"/"+file
        if os.path.exists(file):
            os.remove(file)

def create_data_files_train(wav_inf):
    sub_wav_inf = {key: wav_inf[key] for key in sorted(wav_inf)[validation_num:]}
    train_data_dir = "/home/ganqunx/github/espnet/egs2/aishell/asr1/data/train"
    del_data_files(train_data_dir)
    write_train_files(train_data_dir, sub_wav_inf)

def write_test_files(data_dir, wav_inf):
    file_wav_scp = open(os.path.join(data_dir, "wav.scp"), "a")
    file_text = open(os.path.join(data_dir, "text"), "a")
    file_utt2spk = open(os.path.join(data_dir, "utt2spk"), "a")
    file_spk2utt = open(os.path.join(data_dir, "spk2utt"), "a")
    for key, value in wav_inf.items():
        utt_id = key
        location = value["location"]
        text = "a i u e o"
        speaker_id = value["speaker_id"]
        file_wav_scp.write(key+" "+location+"\n")
        file_text.write(key+" "+text+"\n")
        file_utt2spk.write(key+" "+speaker_id+"\n")
        file_spk2utt.write(speaker_id+" "+key+"\n")
    file_wav_scp.close()
    file_text.close()
    file_utt2spk.close()
    file_spk2utt.close()

def write_train_files(data_dir, wav_inf):
    file_wav_scp = open(os.path.join(data_dir, "wav.scp"), "a")
    file_text = open(os.path.join(data_dir, "text"), "a")
    file_utt2spk = open(os.path.join(data_dir, "utt2spk"), "a")
    file_spk2utt = open(os.path.join(data_dir, "spk2utt"), "a")
    for key, value in wav_inf.items():
        utt_id = key
        location = value["location"]
        text = value["text"]
        speaker_id = value["speaker_id"]
        file_wav_scp.write(key+" "+location+"\n")
        file_text.write(key+" "+text+"\n")
        file_utt2spk.write(key+" "+speaker_id+"\n")
        file_spk2utt.write(speaker_id+" "+key+"\n")

    file_wav_scp.close()
    file_text.close()
    file_utt2spk.close()
    file_spk2utt.close()



def create_data_files_dev(wav_inf):
    sub_wav_inf = {key: wav_inf[key] for key in sorted(wav_inf)[:validation_num]}
    dev_data_dir = "/home/ganqunx/github/espnet/egs2/aishell/asr1/data/dev"
    del_data_files(dev_data_dir)
    write_train_files(dev_data_dir, sub_wav_inf)


def create_data_files_test(wav_inf):
    test_data_dir = "/home/ganqunx/github/espnet/egs2/aishell/asr1/data/test"
    del_data_files(test_data_dir)
    write_test_files(test_data_dir, wav_inf)

text_ref_file = "/home/ganqunx/github/espnet/egs2/aishell/asr1/downloads/taiwanese_asr/train/train-toneless.csv" 
train_wav_dir = "/home/ganqunx/github/espnet/egs2/aishell/asr1/downloads/taiwanese_asr/train/train/16k"
test_wav_dir = "/home/ganqunx/github/espnet/egs2/aishell/asr1/downloads/taiwanese_asr/test/test/16k"
create_text_ref(text_ref_file)
train_wav_inf = proc_train_wav(train_wav_dir)
test_wav_inf = proc_test_wav(test_wav_dir)
create_data_files_dev(train_wav_inf)
create_data_files_train(train_wav_inf)
create_data_files_test(test_wav_inf)