import csv
import os

# 處理espnet執行完畢的結果

"""
設計重點：
結果：
/home/ganqunx/github/espnet/egs2/aishell/asr1/exp/asr_train_asr_branchformer_raw_en_bpe952_sp/decode_asr_branchformer_asr_model_valid.acc.ave/test/text
utt_id text 
目標：
產出：result.csv
id,sentence
782321,a
786664,hit eh hoo i e su e lang eh eh eh eh hoo i eh hoo i e si a eh hoo i eh e soo i eh e

測試過，前面編號沒照順序沒關係

流程：
    讀入result
    寫入dic_result key:utt-id, value:text
    讀入data/test/wav.scp
        utt_id location
    寫入dic_wav
        key:utt-id, value:location
    解譯
        open result.csv for write, with 'w' write mode
        write the header
        loop thru dic_result
            for each record
                find value in dic_wav
                if not found, then alarm and stop
                location => filename => basename
                write a record
                    basename, text
        close file



"""
#        basename = get_basename(location)
def get_basename(location):
    filename = os.path.basename(location)
    basename, ext = os.path.splitext(filename)
    return basename

    # 讀入result
result_file = "/home/ganqunx/github/espnet/egs2/aishell/asr1/exp/asr_train_asr_branchformer_raw_en_bpe952_sp/decode_asr_branchformer_asr_model_valid.acc.ave/test/text"
dic_result = {}
with open(result_file, "r") as file:
    for line in file:
        cols = line.split(" ", 1)
        utt_id = cols[0]
        text = cols[1].strip()  # Remove any leading/trailing whitespaces including newline
        dic_result[utt_id] = text

    # 讀入data/test/wav.scp
file_wav_scp = "/home/ganqunx/github/espnet/egs2/aishell/asr1/data/test/wav.scp"
dic_wav = {}
with open(file_wav_scp, "r") as file:
    for line in file:
        cols = line.split(" ", 1)
        utt_id = cols[0]
        location = cols[1].strip()  # Remove any leading/trailing whitespaces including newline
        basename = get_basename(location)
    #     utt_id basename
    # 寫入dic_wav
    #     key:utt-id, value:location
        dic_wav[utt_id] = basename
    # 解譯
    #     open result.csv for write, with 'w' write mode

file_result_csv = "result.csv"
with open(file_result_csv, "w") as file:
    #     write the header
    file.write("id,sentence\n")
    #     loop thru dic_result
    for key, value in dic_result.items():
    #         for each record
    #             find value in dic_wav
        basename = dic_wav[key]
    #             if not found, then alarm and stop
        if basename is None:
            print("Can't find basename, UTT-ID:"+key)
            break
    #             write a record
    #                 basename, text
        line = basename+","+value+"\n" # value = text
        file.write(line)
    #     close file
