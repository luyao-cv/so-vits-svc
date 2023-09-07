import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

debug = False
local_model_root = '/paddle/luyao15/project/so-vits-svc/weights'
Author = glob.glob("/paddle/luyao15/project/so-vits-svc/weights/*")
Author = list(x.split("/paddle/luyao15/project/so-vits-svc/weights/")[1] for x in Author)
Author = dict(zip(Author, Author))
# Author = {"孙燕姿":"sun","Zecora":"Zecora"}
output_format = "wav"
vc_transform = 0
cluster_ratio = 0.5
slice_db = -40
noise_scale = 0.4
pad_seconds = 0.5
cl_num = 0
lg_num = 0
lgr_num = 0.75
f0_predictor = "pm"
enhancer_adaptive_key = 0
cr_threshold = 0.05
k_step = 100
use_spk_mix = False
second_encoding = False
loudness_envelope_adjustment = 0

cuda = {}
spks = list(Author.keys())
model = None

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
        
    
def modelAnalysis(device, sid, msg):
    global model, spks, Author

    device = cuda[device] if "CUDA" in device else device
    print(device)
    model_path = glob.glob(os.path.join(local_model_root, Author[sid], 'G*.pth'))[0]
    config_path = glob.glob(os.path.join(local_model_root, Author[sid], '*.json'))[0]
    cluster_model_path = glob.glob(os.path.join(local_model_root, Author[sid], '*.pt'))[0]
    diff_config_path = glob.glob(os.path.join(local_model_root, Author[sid], '*.json'))[0]
    diff_model_path = glob.glob(os.path.join(local_model_root, Author[sid], '*.pth'))[0]
    print(config_path)
    print(cluster_model_path)
    print(diff_config_path)
    print(diff_model_path)
    model = Svc(model_path,
            config_path,
            device=device if device != "Auto" else None,
            nsf_hifigan_enhance=False,
            cluster_model_path=cluster_model_path,
            diffusion_model_path=diff_model_path,
            diffusion_config_path=diff_config_path,
            shallow_diffusion = False,
            only_diffusion = False,
            spk_mix_enable = False,
            feature_retrieval = False,
            # model_path,
            # config_path,
            # device=device if device != "Auto" else None,
            # cluster_model_path = cluster_model_path,
            # nsf_hifigan_enhance=False,
            # diffusion_model_path = diff_model_path,
            # diffusion_config_path = diff_config_path,
            # shallow_diffusion = True if diff_model_path is not None else False,
            # only_diffusion = False,
            # spk_mix_enable = False,
            # feature_retrieval = False
            )
    #   Auto False    False False False
    spks = list(model.spk2id.keys())
    device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
    _name = "/paddle/luyao15/project/so-vits-svc/weights/"
    msg = f"成功加载模型到设备{device_name}上\n"
    if cluster_model_path is None:
        msg += "未加载聚类模型或特征检索模型\n"
    if diff_model_path is None:
        msg += "未加载扩散模型\n"
    else:
        msg += f"扩散模型{diff_model_path.split(_name)[1]}加载成功\n"
    msg += "当前模型的可用音色：\n"
    for i in spks:
        msg += i + " "
    # import pdb
    # pdb.set_trace()
    Author[spks[0]] = Author[sid]
    sid = spks[0]
    return sid, msg

def vc_fn(sid, input_audio, auto_f0):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 未知原因Gradio上传的filepath会有一个奇怪的固定后缀，这里去掉
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, auto_f0):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    print(audio_path,sid,vc_transform,slice_db,cluster_ratio,auto_f0,noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment)
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Column():
        model_load_button = gr.Button(value="加载模型", variant="primary")
        sid = gr.Dropdown(label="音色（说话人）",choices = spks,value=spks[0])
        sid_output = gr.Textbox(label="Output Message")
        auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False)
        

    with gr.Tabs():
        with gr.TabItem("音频转音频"):
            vc_input3 = gr.Audio(label="选择音频", type="filepath", value="/paddle/luyao15/project/so-vits-svc/2_ly.mp3")
            vc_submit = gr.Button("音频转换", variant="primary")
        with gr.TabItem("文字转音频"):
            text2tts=gr.Textbox(label="重新加载模型，在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪", value="大家好，我是孙燕姿")
            with gr.Row():
                tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
            vc_submit2 = gr.Button("文字转换", variant="primary")
    with gr.Row():
        with gr.Column():
            vc_output1 = gr.Textbox(label="Output Message")
        with gr.Column():
            vc_output2 = gr.Audio(label="Output Audio", interactive=False)
        
    device = gr.Dropdown(label="推理设备, 默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
    model_load_button.click(modelAnalysis,[device, sid, sid_output],[sid,sid_output])
    vc_submit.click(vc_fn, [sid, vc_input3, auto_f0], [vc_output1, vc_output2])
    vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, auto_f0], [vc_output1, vc_output2])
      

    os.system("start http://10.21.226.179:8908/")
    app.launch(share=True, server_name="0.0.0.0", server_port=8908)