"""Predict with LLM on task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import openai
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute

os.environ.setdefault("WANDB_ENT", "xxxx")
os.environ.setdefault("OPENAI_API_KEY", "sk-xxxx")
os.environ.setdefault("WANDB_API_KEY", "xxxx")

utils.setup_logger()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set up OpenAI API credentials.

def main(args):
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing use_context=True for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing answerable_only=True for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)

    user = os.environ['USER']
    entity = os.environ['WANDB_ENT']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    metric = utils.get_metric(args.metric)

    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # --------------- 主循环（异步准确率版本）----------------
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        generations, results_dict, p_trues = {}, {}, []
        all_acc_futures = []
        acc_futures_index_map = {}  # (eid, gen_i) -> (accuracies索引, is_most_likely)

        # 构造数据索引映射：accuracies必须顺序存储每个样本的most-likely acc
        acc_idx = 0  # 用于accuracies的顺序定位
        accuracies = []

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        with ThreadPoolExecutor(max_workers=40) as executor:
            for index in tqdm(indices):
                if (acc_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                example = dataset[index]
                question, context = example["question"], example['context']
                generations[example['id']] = {'question': question, 'context': context}
                correct_answer = example['answers']['text']

                current_input = make_prompt(
                    context, question, None, BRIEF, args.brief_always and args.enable_brief)
                local_prompt = prompt + current_input

                logging.info('Current input: '.ljust(15) + current_input)

                full_responses = []

                if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                    num_generations = 1
                else:
                    num_generations = args.num_generations + 1

                most_likely_answer_dict = None

                for i in range(num_generations):
                    temperature = 0.1 if i == 0 else args.temperature
                    predicted_answer, token_log_likelihoods, (embedding, emb_last_before_gen, emb_before_eos) = model.predict(local_prompt, temperature, return_latent=True)
                    embedding = embedding.cpu() if embedding is not None else None
                    emb_last_before_gen = emb_last_before_gen.cpu() if emb_last_before_gen is not None else None
                    emb_before_eos = emb_before_eos.cpu() if emb_before_eos is not None else None

                    compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                    eid = example['id']

                    def metric_task(predicted_answer, example, model):
                        return metric(predicted_answer, example, model)

                    # 异步提交（带索引信息）
                    if correct_answer and compute_acc:
                        fut = executor.submit(metric_task, predicted_answer, example, model)
                        all_acc_futures.append((fut, eid, i, acc_idx))
                        acc_futures_index_map[(eid, i)] = (acc_idx, i == 0)
                        if i == 0:
                            accuracies.append(None)  # 先占位
                    else:
                        if i == 0:
                            accuracies.append(0.0)

                    if i == 0:
                        most_likely_answer_dict = {
                            'response': predicted_answer,
                            'token_log_likelihoods': token_log_likelihoods,
                            'embedding': embedding,
                            'accuracy': None,  # 占位
                            'emb_last_tok_before_gen': emb_last_before_gen,
                            'emb_tok_before_eos': emb_before_eos,
                        }

                        generations[eid].update({
                            'most_likely_answer': most_likely_answer_dict,
                            'reference': utils.get_reference(example),
                        })
                    else:
                        full_responses.append(
                            (predicted_answer, token_log_likelihoods, embedding, None)
                        )

                generations[example['id']]['responses'] = full_responses

                if args.compute_p_true and dataset_split == 'validation':
                    p_true = p_true_utils.calculate_p_true(
                        model, question, most_likely_answer_dict['response'],
                        [r[0] for r in full_responses], p_true_few_shot_prompt,
                        hint=args.p_true_hint)
                    p_trues.append(p_true)
                    logging.info('p_true: %s', p_true)
                acc_idx += 1

            # 异步准确率回填
            for fut, eid, i, acc_pos in all_acc_futures:
                acc = fut.result()
                if i == 0:
                    generations[eid]['most_likely_answer']['accuracy'] = acc
                    accuracies[acc_pos] = acc
                else:
                    generations[eid]['responses'][i-1] = (
                        generations[eid]['responses'][i-1][0],
                        generations[eid]['responses'][i-1][1],
                        generations[eid]['responses'][i-1][2],
                        acc
                    )

        # 完成所有future后保证accuracies没有None
        assert all(a is not None for a in accuracies), f"Accuracies含None: {accuracies}"

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    logging.info('STARTING generate_answers!')
    main(args)
    logging.info('FINISHED generate_answers!')

    if args.compute_uncertainties:
        logging.info(50 * '#X')
        logging.info('STARTING compute_uncertainty_measures!')
        main_compute(args)
        logging.info('FINISHED compute_uncertainty_measures!')
