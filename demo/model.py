from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class miniCPM(LLM):

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = """You are an AI assistant whose name is miniCPM (ArcherBot).
        - miniCPM (ArcherBot) is a conversational language model that is developed by AISI (北京科学智能研究院). It is designed to be helpful, honest, and harmless.
        - miniCPM (ArcherBot) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, temperature=0.8, top_p=0.8)
        return response

    @property
    def _llm_type(self) -> str:
        return "miniCPM"