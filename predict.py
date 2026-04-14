import os
import sys
import urllib.parse
from argparse import Namespace
from cog import BasePredictor, Input, Path as CogPath
import hashlib
import requests

sys.path.insert(0, os.path.abspath("src"))

import main as m

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        input_audio: CogPath = Input(
            description="Upload your audio file here.",
            default=None,
        ),
        rvc_model: str = Input(
            description="RVC model for a specific voice. If using a custom model, this should match the name of the downloaded model. If a 'custom_rvc_model_download_url' is provided, this will be automatically set to the name of the downloaded model.",
            default="Obama",
            choices=[
                "Obama",
                "Trump",
                "Sandy",
                "Rogan",
                "Obama",
                "CUSTOM",
            ],
        ),
        custom_rvc_model_download_url: str = Input(
            description="URL to download a custom RVC model. If provided, the model will be downloaded (if it doesn't already exist) and used for prediction, regardless of the 'rvc_model' value.",
            default=None,
        ),
        # 新增 1：直接传 pth
        pth_model_download_url: str = Input(
            description="Direct URL to a .pth file. If provided, it takes priority over zip URL.",
            default=None,
        ),
        # 新增 2：可选 index
        index_file_download_url: str = Input(
            description="Optional direct URL to a .index file paired with pth.",
            default=None,
        ),
        pitch_change: float = Input(
            description="Adjust pitch of AI vocals in semitones. Use positive values to increase pitch, negative to decrease.",
            default=0,
        ),
        index_rate: float = Input(
            description="Control how much of the AI's accent to leave in the vocals.",
            default=0.5,
            ge=0,
            le=1,
        ),
        filter_radius: int = Input(
            description="If >=3: apply median filtering to the harvested pitch results.",
            default=3,
            ge=0,
            le=7,
        ),
        rms_mix_rate: float = Input(
            description="Control how much to use the original vocal's loudness (0) or a fixed loudness (1).",
            default=0.25,
            ge=0,
            le=1,
        ),
        f0_method: str = Input(
            description="Pitch detection algorithm. 'rmvpe' for clarity in vocals, 'mangio-crepe' for smoother vocals.",
            default="rmvpe",
            choices=["rmvpe", "mangio-crepe"],
        ),
        crepe_hop_length: int = Input(
            description="When `f0_method` is set to `mangio-crepe`, this controls how often it checks for pitch changes in milliseconds.",
            default=128,
        ),
        protect: float = Input(
            description="Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable.",
            default=0.33,
            ge=0,
            le=0.5,
        ),
        output_format: str = Input(
            description="wav for best quality and large file size, mp3 for decent quality and small file size.",
            default="mp3",
            choices=["mp3", "wav"],
        ),
    ) -> CogPath:
        """
        Runs a single prediction on the model.
        """
        if pth_model_download_url:
            def _download_to(url: str, dst: str):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
            # 用 URL 做稳定目录名，避免重复下载
            key_src = pth_model_download_url + "|" + (index_file_download_url or "")
            model_key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
            rvc_dirname = f"custom_{model_key}"
            model_dir = os.path.join(m.rvc_models_dir, rvc_dirname)
            os.makedirs(model_dir, exist_ok=True)
            # pth 文件名
            pth_name = urllib.parse.unquote(pth_model_download_url.split("/")[-1]).split("?")[0]
            if not pth_name.endswith(".pth"):
                pth_name = "model.pth"
            pth_path = os.path.join(model_dir, pth_name)
            if not os.path.exists(pth_path):
                print(f"[+] Downloading pth -> {pth_path}")
                _download_to(pth_model_download_url, pth_path)
            # 可选 index
            if index_file_download_url:
                index_name = urllib.parse.unquote(index_file_download_url.split("/")[-1]).split("?")[0]
                if not index_name.endswith(".index"):
                    index_name = "model.index"
                index_path = os.path.join(model_dir, index_name)
                if not os.path.exists(index_path):
                    print(f"[+] Downloading index -> {index_path}")
                    _download_to(index_file_download_url, index_path)
            rvc_model = rvc_dirname
        elif custom_rvc_model_download_url:
            custom_rvc_model_download_name = urllib.parse.unquote(
                custom_rvc_model_download_url.split("/")[-1]
            )
            custom_rvc_model_download_name = os.path.splitext(
                custom_rvc_model_download_name
            )[0]
            print(
                f"[!] The model will be downloaded as '{custom_rvc_model_download_name}'."
            )
            m.download_online_model(
                url=custom_rvc_model_download_url,
                dir_name=custom_rvc_model_download_name,
                overwrite=True
            )
            rvc_model = custom_rvc_model_download_name
        else:
            print(
                "[!] Since no URL was provided, we will use the selected RVC model."
            )

        rvc_dirname = rvc_model
        if not os.path.exists(os.path.join(m.rvc_models_dir, rvc_dirname)):
            raise Exception(
                f"The folder {os.path.join(m.rvc_models_dir, rvc_dirname)} does not exist."
            )

        output_path = m.voice_conversion(
            str(input_audio),
            rvc_dirname,
            pitch_change,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect
        )
        print(f"[+] Converted audio generated at {output_path}")

        # Return the output path
        return CogPath(output_path)
