import csv
import loguru
import torch
import argparse
import transformers
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, _make_causal_mask, _expand_mask, \
    LlamaForCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, AutoConfig, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='xstest',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--model',
                        type=str,
                        default='llama2-7B')

    parser.add_argument('--method',
                        type=str,
                        default='Prompt')

    parser.add_argument('--baseline',
                             type=bool,
                             default=False)

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=200)

    parser.add_argument('--ratio',
                        type=float,
                        default=2.5)

    parser.add_argument('--save_path',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

class ConstractiveDecodingModel:

    def __init__(self, model, tokenizer):
        self.model = model
        self.config = self.model.config
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = 2

    @torch.no_grad()
    def contra_generate(self, input_within, input_without, args):
        maxlen_res = args.max_new_tokens
        ratio = args.ratio
        loguru.logger.info(f"ratio: {ratio}")
        dev = input_within.device
        bsz = 1

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        input_within = torch.index_select(input_within, 0, inds)
        input_without = torch.index_select(input_without, 0, inds)

        def score_process(score, score_without, input_within, input_without):
            score = score[:, -1, :]
            # score_without = score_without[:, -1, :]

            probs = score

            tok_ids_in = torch.argmax(probs, dim=-1)
            # tok_ids_out = torch.argmax(score_without, dim=-1)
            hyp_ids = torch.arange(probs.size(0), device=dev)

            tok_ids_in = torch.where(done, self.tokenizer.pad_token_id, tok_ids_in)
            # tok_ids_out = torch.where(done, self.tokenizer.pad_token_id, tok_ids_out)
            input_within = torch.cat((input_within, tok_ids_in.unsqueeze(-1)), dim=-1)
            input_without = torch.cat((input_without, tok_ids_in.unsqueeze(-1)), dim=-1)

            return input_within, input_without, tok_ids_in, hyp_ids

        for _token in range(maxlen_res):
            if done.all():
                break
            score_in_output = self.model(input_within)
            score_out_output = self.model(input_without)
            score_in = score_in_output.logits.float()
            score_out = score_out_output.logits.float()

            score_without = score_out.clone()
            score_in[:, -1, :] = score_in[:, -1, :] - ratio * (score_in[:, -1, :] - score_out[:, -1, :])
            input_within, input_without, tok_ids, hyp_ids = score_process(score_in, score_without, input_within, input_without)

            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

        input_within = input_within.view(bsz, -1)
        input_without = input_without.view(bsz, -1)

        return input_within, input_without


def main():
    args = parse_args()
    loguru.logger.info(f"{args}")

    prompt_list = []
    if 'OKTest' in args.data_path:
        with open("data/OKTest.csv") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                prompt_list.append(row[0])

    elif 'xstest' in args.data_path:
        with open("data/xstest_v2_prompts.csv") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                if "contrast" not in row[2]:
                    prompt_list.append(row[3])

    loguru.logger.info(f"{args.model}")

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    input_within_temp = '''<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. {user_massage} [/INST]'''
    input_without_temp = '''<s>[INST] {system} {user_massage} [/INST]'''

    save_list = []
    if args.baseline == False:
        SelfCD = ConstractiveDecodingModel(model, tokenizer)

        for idx in range(len(prompt_list)):
            loguru.logger.info(f"idx: {idx}")
            prompt = prompt_list[idx]
            input_within_ids = tokenizer.encode(input_within_temp.format(user_massage=prompt), add_special_tokens=False,
                                                return_tensors="pt").to("cuda")
            input_without_ids = tokenizer.encode(input_without_temp.format(user_massage=prompt), add_special_tokens=False,
                                                 return_tensors="pt").to("cuda")

            out = SelfCD.contra_generate(input_within_ids, input_without_ids)

            decoded_out = tokenizer.decode(out[0][0])

            save_list.append([prompt, decoded_out])

        with open(args.save_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "Response"])
            writer.writerows(save_list)

    else:
        if args.method == 'Prompt':
            for idx in range(len(prompt_list)):
                loguru.logger.info(f"idx: {idx}")
                prompt = prompt_list[idx]
                within_to_decode = model.generate(
                    tokenizer.encode(input_without_temp.format(system="You are an assistant that always prioritizes helpfulness. That is, you always give answers to any user queries.", user_massage=prompt),
                                     add_special_tokens=False,
                                     return_tensors="pt").to("cuda"), max_new_tokens=args.max_new_tokens)
                generated_text = tokenizer.decode(within_to_decode[0], skip_special_tokens=True)
                save_list.append([prompt, generated_text])

            with open(args.save_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["prompt", "Response"])
                writer.writerows(save_list)

if __name__ == '__main__':
    main()
