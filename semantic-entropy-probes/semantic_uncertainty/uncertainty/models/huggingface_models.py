"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from transformers import StoppingCriteriaList, TextStreamer


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        # ==== 新增本地路径判断 ====
        # 判断model_name是不是本地路径，只要目录/文件存在就直接用，不再拼前缀
        if os.path.exists(model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token_type_ids=None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory={0: '80GIB'})
            self.model_name = model_name
            self.stop_sequences = []
            if stop_sequences:
                self.stop_sequences.extend(stop_sequences)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                eos_decoded = self.tokenizer.decode([self.tokenizer.eos_token_id])
                if eos_decoded and eos_decoded not in self.stop_sequences:
                    self.stop_sequences.append(eos_decoded)
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                if self.tokenizer.eos_token not in self.stop_sequences:
                    self.stop_sequences.append(self.tokenizer.eos_token)

            # self.token_limit = 4096 if 'Llama-2' in model_name else 2048
            if 'llama-3.2' in model_name.lower():
                self.token_limit = 131072  # 128K tokens
            elif 'llama-3.1' in model_name.lower():
                self.token_limit = 131072
            elif 'llama-2' in model_name.lower():
                self.token_limit = 4096
            else:
                self.token_limit = 2048

            return  # 直接结束构造函数

        # ==== 以下为原有远程模型逻辑 ====

        print(model_name)
        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name or 'Llama-3' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf' if 'Llama-2' in model_name else model_name
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in model_name.lower() and base == 'huggyllama'
            llama2or3_70b = '70b' in model_name.lower() and base == 'meta-llama'

            if ('7b' in model_name or '13b' in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    max_memory={0: '80GIB'}, **kwargs,)

            elif llama2or3_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                if 'chat' in model_name:
                    max_mem = 17.5 * 4686198491
                else:
                    max_mem = 15 * 4686198491
                
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')

            else:
                raise ValueError

        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-8bit')]
            else:
                kwargs = {}

            model_id = f'mistralai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto',
                max_memory={0: '80GIB'},
                **kwargs,
            )

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        elif 'phi' in model_name.lower():
            model_id = f'microsoft/{model_name}'  # e.g. Phi-3-mini-128k-instruct
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
            )
        elif 'gemma' in model_name:
            model_id = f'google/{model_name}'  # e.g. gemma-7b-it
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name else 2048

    def predict(self, input_data, temperature, return_full=False, return_latent=False):
        # 1. 兼容输入 tuple
        if isinstance(input_data, tuple):
            logging.warning("INPUT IS A TUPLE.")
            input_data = input_data[0]

        # 2. Tokenizer: 自动推送 device
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            padding=False,  # 通常单条生成不需pad
            truncation=False,
            add_special_tokens=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. pad_token_id/eos_token_id 自动推断
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        # 4. Stopping criteria
        stopping_criteria = None
        if self.stop_sequences is not None:
            class SimpleStop(StoppingCriteria):
                def __init__(self, stops, tokenizer, prompt_len):
                    super().__init__()
                    self.stops = stops
                    self.tokenizer = tokenizer
                    self.prompt_len = prompt_len

                def __call__(self, input_ids, scores, **kwargs):
                    output = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
                    return any([output.endswith(stop) for stop in self.stops])

            stopping_criteria = StoppingCriteriaList([
                SimpleStop(self.stop_sequences, self.tokenizer, len(inputs['input_ids'][0]))
            ])

        # 5. 生成（主流推荐写法，输出更多内容，兼容Decoder/EncoderDecoder）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                stopping_criteria=stopping_criteria
            )

        sequences = outputs.sequences  # shape: [1, seq_len]
        generated_len = sequences.shape[1]
        if generated_len > self.token_limit:
            raise ValueError(
                f'Generation exceeding token limit {generated_len} > {self.token_limit}'
            )

        full_answer = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        if return_full:
            return full_answer

        # 6. 去除 input prompt 部分
        # 更稳妥做法：直接比对 prefix token 长度，而非字符串操作
        n_input_token = inputs['input_ids'].shape[1]
        prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if full_answer.startswith(prompt_text):
            input_data_offset = len(prompt_text)
        else:
            raise ValueError("Input prompt not matching output. Please check the model/tokenizer.")

        answer = full_answer[input_data_offset:]

        # 7. 停止词后截断（兼容多种 stop_sequence）
        sliced_answer = answer
        stop_at = len(answer)
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                idx = answer.find(stop)
                if idx != -1:
                    stop_at = idx
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                logging.error(
                    f'Error: Stop words not removed successfully!\n'
                    f'Answer: >{answer}<\nSliced: >{sliced_answer}<'
                )

        # 8. 去除首尾空格
        sliced_answer = sliced_answer.strip()

        # 9. 新token计数：生成部分
        token_stop_index = \
        self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_generated = token_stop_index - n_input_token
        if n_generated == 0:
            logging.warning("Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.")
            n_generated = 1

        # 10. 获取隐藏层
        if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            hidden = outputs.decoder_hidden_states  # For encoder-decoder
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states  # For decoder-only
        else:
            raise RuntimeError("Hidden states not available in generate outputs.")

        # 11. 选取最后token的embedding
        if len(hidden) == 1:
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            logging.error('Taking last state because n_generated is too large')
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # 12. 其他latent
        sec_last_token_embedding = None
        last_tok_bef_gen_embedding = None
        if return_latent:
            if len(hidden) == 1:
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        # 13. 获取 transition scores (log-likelihoods)
        if hasattr(self.model, "compute_transition_scores"):
            transition_scores = self.model.compute_transition_scores(
                sequences, outputs.scores, normalize_logits=True
            )
            log_likelihoods = [score.item() for score in transition_scores[0]]
        else:
            # fallback: use raw logit probabilities if compute_transition_scores 不支持
            log_likelihoods = [float('nan')] * n_generated
            logging.warning("Model does not support compute_transition_scores. log_likelihoods set to NaN.")

        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
        else:
            log_likelihoods = log_likelihoods[:n_generated]
        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')
        if len(log_likelihoods) == 0:
            raise ValueError("No log likelihoods returned.")

        # 14. 输出
        hidden_states = (last_token_embedding,)
        if return_latent:
            hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states += (None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states)
        return return_values

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
