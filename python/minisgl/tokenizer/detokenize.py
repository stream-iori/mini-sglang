from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from transformers import LlamaTokenizer

from minisgl.message import DetokenizeMsg

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # 检查是否为 CJK 字符 (中日韩统一表意文字)
    # 这对于流式解码很重要，因为 CJK 字符不需要空格分隔
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """
    返回包含完整单词的最长可打印子串。
    用于处理 utf-8 多字节字符被截断或单词未生成完整的情况。
    """
    # 如果以换行符结尾，通常意味着一个完整的句子或段落结束，直接输出
    if text.endswith("\n"):
        return text
    # 如果最后一个是 CJK 字符，直接输出 (CJK 不需要空格)
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # 如果倒数第二个是 CJK，输出除了最后一个字符以外的部分
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # 否则 (英文等)，输出到最后一个空格为止，保留未完成的单词在缓冲区
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int] # 已生成的所有 Token IDs
    decoded_str: str       # 已解码的完整字符串
    read_offset: int  # length of read ids (已处理的 ID 长度)
    surr_offset: int  # length of surr ids (乱码/截断保护偏移量)
    sent_offset: int  # length of sent out string (已发送给前端的字符串长度)


class DetokenizeManager:
    """
    Detokenize 管理器。
    运行在 Detokenizer 进程中。
    负责将后端生成的 Token ID 流式解码为文本，并处理 unicode 截断和乱码问题。
    """
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        
        # 1. 更新状态并准备解码
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not msg.finished:
                s.decoded_ids.append(msg.next_token)
            
            # read_ids: 需要本次解码的部分
            read_ids.append(s.decoded_ids[s.surr_offset :])
            # surr_ids: 之前保留的可能乱码的部分 (surrogate)
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # 2. 批量解码
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            # 获取新增的文本 (去除之前的 surr 部分)
            new_text = read_str[len(surr_str) :]
            
            # 3. 处理解码结果
            # 如果不含  (REPLACEMENT CHARACTER)，说明解码成功
            if len(new_text) > 0 and not new_text.endswith("\ufffd"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset # 推进 surr_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # 否则可能遇到了被截断的多字节字符，使用启发式方法只输出“安全”的部分
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 计算本次增量输出
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            
            # 4. 清理完成的请求
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs