import numpy as np
from funasr.register import tables
import logging
import random
import re


@tables.register("prompt_classes", "MultiContextPrompt")
class MultiContextPrompt:
    CONTEXT_TEMPLATES = {
        'en': {
            'header': "Please combine the context information provided below to complete the speech transcription task more accurately. If there is no relevant information, we will leave it blank.\n",
            'fields': {
                'hist_context': "Historical transcription: {hist_context}\n",
                'one_pass_result': "One-pass result: {one_pass_result}\n",
                'hotwords': "Hotword list: {hotwords}\n"
            }
        },
        'zh': {
            'header': "请结合下面提供的上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n",
            'fields': {
                'hist_context': "历史转写结果：{hist_context}\n",
                'one_pass_result': "一遍解码结果：{one_pass_result}\n",
                'hotwords': "热词列表：{hotwords}\n"
            }
        }
    }

    def __init__(self,
                 use_hist=True,
                 use_one_pass_result=True,
                 use_hotwords=True,
                 use_asr_hotwords=True,
                 use_multi_lingual_prompt=True,
                 **kwargs):
        self.use_hist = use_hist
        self.use_one_pass_result = use_one_pass_result
        self.use_hotwords = use_hotwords
        self.use_asr_hotwords = use_asr_hotwords
        self.use_multi_lingual_prompt = use_multi_lingual_prompt
        self.kwargs = kwargs

        chinese_hotwords_list = kwargs.get("chinese_hotwords_list", "")
        english_hotwords_list = kwargs.get("english_hotwords_list", "")
        if chinese_hotwords_list:
            self.chinese_hotwords_list, self.chinese_hotwords_num = self.get_hotwords_list(chinese_hotwords_list)
        else:
            self.chinese_hotwords_list = None
            self.chinese_hotwords_num = 0
        logging.info(f"chinese_hotwords_num: {self.chinese_hotwords_num}")

        if english_hotwords_list:
            self.english_hotwords_list, self.english_hotwords_num = self.get_hotwords_list(english_hotwords_list)
        else:
            self.english_hotwords_list = None
            self.english_hotwords_num = 0
        logging.info(f"english_hotwords_num: {self.english_hotwords_num}")

        self.max_neg_hotwords_num = kwargs.get("max_neg_hotwords_num", 900)
        self.min_neg_hotwords_num = kwargs.get("min_neg_hotwords_num", 0)

    def get_hotwords_list(self, hotwords_file):
        with open(hotwords_file, "r") as f:
            hotwords_list = f.read().strip().split("\n")
        return hotwords_list, len(hotwords_list)

    def detect_language(self, text):
        if isinstance(text, list):
            text = " ".join(text)

        chinese_pattern = re.compile(
            "["
            "\u4e00-\u9fff"  # CJK Unified Ideographs
            "]+"
        )

        english_pattern = re.compile(r'[A-Za-z]+')

        chinese_matches = chinese_pattern.findall(text)
        english_matches = english_pattern.findall(text)

        chinese_length = sum(len(match) for match in chinese_matches)
        english_length = sum(len(match) for match in english_matches)

        total_length = len(text)

        if total_length == 0:
            return 'zh'

        if (chinese_length > english_length) and (chinese_length / total_length > 0.3):
            return 'zh'
        else:
            return 'en'

    def hotwords_sampling(self, hotwords):

        # hotwords_list = hotwords.split(", ")
        hotwords_list = hotwords
        selected_hotwords = []
        if self.max_neg_hotwords_num > -1:
            max_neg_hotwords_num = min(self.max_neg_hotwords_num, len(hotwords_list))
        else:
            max_neg_hotwords_num = len(hotwords_list)

        if self.min_neg_hotwords_num < max_neg_hotwords_num:
            selected_hotwords_num = np.random.randint(self.min_neg_hotwords_num, max_neg_hotwords_num + 1)
        else:
            selected_hotwords_num = max_neg_hotwords_num
        if selected_hotwords_num > 0:
            selected_hotwords = np.random.choice(hotwords_list, selected_hotwords_num, replace=False).tolist()

        return selected_hotwords, selected_hotwords_num

    def get_prompt(self, item, language):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        hotwords = None
        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
        if self.use_asr_hotwords and item.get("asr_hotwords"):
            hotwords = item["asr_hotwords"]
        if hotwords is not None and hotwords != "":
            language = self.detect_language(hotwords)
            if language == 'en':
                neg_hotwords = self.english_hotwords_list
            else:
                neg_hotwords = self.chinese_hotwords_list
            if neg_hotwords is not None:
                selected_neg_hotwords, selected_neg_hotwords_num = self.hotwords_sampling(neg_hotwords)
            else:
                selected_neg_hotwords = []

            if not isinstance(hotwords, list):
                pos_hotwords = hotwords.split(", ")
            else:
                pos_hotwords = hotwords
            hotwords = pos_hotwords + selected_neg_hotwords
            random.shuffle(hotwords)
            hotwords = ", ".join(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += ''.join(context_lines)
        else:
            prompt += "\n\n\n"

        return prompt

    def get_inference_prompt(self, item, language="zh"):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        hotwords = None
        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
        if self.use_asr_hotwords and item.get("asr_hotwords"):
            hotwords = item["asr_hotwords"]
        if hotwords is not None and hotwords != "":
            print(f"hotwords: {hotwords}")
            language = self.detect_language(hotwords)
            if language == 'en':
                neg_hotwords = self.english_hotwords_list
            else:
                neg_hotwords = self.chinese_hotwords_list
            if neg_hotwords is not None:
                selected_neg_hotwords, selected_neg_hotwords_num = self.hotwords_sampling(neg_hotwords)
            else:
                selected_neg_hotwords = []

            if not isinstance(hotwords, list):
                pos_hotwords = hotwords.split(", ")
            else:
                pos_hotwords = hotwords
            hotwords = pos_hotwords + selected_neg_hotwords
            print(f"selected_neg_hotwords_num: {selected_neg_hotwords_num}")
            random.shuffle(hotwords)
            hotwords = ", ".join(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += ''.join(context_lines)
        else:
            prompt += "\n\n\n"

        return prompt


@tables.register("prompt_classes", "MultiContextPromptNew")
class MultiContextPromptNew:
    CONTEXT_TEMPLATES = {
        'en': {
            'header': "Please combine the context information to complete the speech transcription task more accurately. If there is no relevant information, we will leave it blank.\n\n",
            'context_header': "**Context:**\n",
            'fields': {
                'hist_context': "Historical transcription: {hist_context}\n",
                'one_pass_result': "One-pass result: {one_pass_result}\n",
                'hotwords': "Hotword list: {hotwords}\n"
            }
        },
        'zh': {
            'header': "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n",
            'context_header': "**上下文信息：**\n",
            'fields': {
                'hist_context': "历史转写结果：{hist_context}\n",
                'one_pass_result': "一遍解码结果：{one_pass_result}\n",
                'hotwords': "热词列表：{hotwords}\n"
            }
        }
    }

    def __init__(self,
                 use_hist=True,
                 use_one_pass_result=True,
                 use_hotwords=True,
                 use_multi_lingual_prompt=True,
                 **kwargs):
        self.use_hist = use_hist
        self.use_one_pass_result = use_one_pass_result
        self.use_hotwords = use_hotwords
        self.use_multi_lingual_prompt = use_multi_lingual_prompt

        self.use_full_hotwords_ratio = kwargs.get("use_full_hotwords_ratio", 0.2)
        self.max_hotwords_num = kwargs.get("max_hotwords_num", -1)
        self.min_hotwords_num = kwargs.get("min_hotwords_num", 15)

    def hotwords_sampling(self, hotwords):

        hotwords_list = hotwords.split(", ")
        if self.max_hotwords_num > 0:
            max_hotwords_num = min(self.max_hotwords_num, len(hotwords_list))
        else:
            max_hotwords_num = len(hotwords_list)

        if self.min_hotwords_num < max_hotwords_num:
            selected_hotwords_num = np.random.randint(self.min_hotwords_num, max_hotwords_num + 1)
        else:
            selected_hotwords_num = max_hotwords_num

        selected_hotwords = np.random.choice(hotwords_list, selected_hotwords_num, replace=False)
        hotwords_list = ", ".join(selected_hotwords)

        return hotwords_list, selected_hotwords_num

    def get_prompt(self, item, language):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
            if np.random.rand() < self.use_full_hotwords_ratio:
                hotwords = hotwords
            else:
                hotwords, selected_hotwords_num = self.hotwords_sampling(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += template['context_header'] + ''.join(context_lines)

        return prompt

    def get_inference_prompt(self, hist_context="", one_pass_result="", hotwords=""):
        language = 'zh' if self.use_multi_lingual_prompt and np.random.rand() < 0.5 else 'en'
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if hist_context:
            context_lines.append(template['fields']['hist_context'].format(hist_context=hist_context))
        if one_pass_result:
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=one_pass_result))
        if hotwords:
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += template['context_header'] + ''.join(context_lines)

        return prompt
