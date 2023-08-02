# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
from types import MethodType
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import json
import safetensors
import torch
from datasets import Dataset as HfDataset
from peft import PeftModel
from requests.exceptions import HTTPError
from torch.nn import Module
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, HubStrategy
from transformers.training_args import TrainingArguments

from swift.hub import HubApi, ModelScopeConfig, Repository
from swift.hub.check_model import check_local_model_is_latest
from swift.hub.constants import ModelVisibility
from swift.tuners import SwiftModel
from swift.utils.constants import Invoke
from swift.utils.logger import get_logger
from .utils import can_return_loss, find_labels, get_function

logger = get_logger()


def _push_to_hub(self: Repository,
                 commit_message: str = 'Commit files to Modelscope Hub',
                 **kwargs):
    blocking = kwargs.get('blocking', True)
    self.push(commit_message)
    if not blocking:
        # Compatible with transformers
        return None, None
    else:
        return None


class PushToMsHubMixin:
    repo: Repository

    def _add_patterns_to_gitignores(
            self,
            patterns: List[str],
            commit_message: Optional[str] = None) -> None:
        # Make sure we only do this on the main process
        if not self.is_world_process_zero():
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to .gitignore'

        # Get current .gitignore content
        repo_dir = self.repo.model_dir
        gitignore_path = os.path.join(repo_dir, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                current_content = f.read()
        else:
            current_content = ''

        # Add the patterns to .gitignore
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if content == '' or content.endswith('\n'):
                    content += f'{pattern}\n'
                else:
                    content += f'\n{pattern}\n'

        # Write the .gitignore file if it has changed
        if content != current_content:
            with open(gitignore_path, 'w') as f:
                logger.debug(f'Writing .gitignore file. Content: {content}')
                f.write(content)
        self.repo.push(commit_message)

    def init_git_repo(self, at_init: bool = False) -> None:
        if not self.is_world_process_zero():
            return
        # Make sure the repo exists.
        api = HubApi()
        hub_token = self.args.hub_token
        if hub_token is None:
            hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if hub_token is not None:
            api.login(hub_token)

        hub_model_id = self.args.hub_model_id
        assert hub_model_id is not None, 'Please enter a valid hub_model_id'
        if '/' not in self.args.hub_model_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
            hub_model_id = f'{user_name}/{hub_model_id}'
            logger.info(
                f"'/' not in hub_model_id, setting hub_model_id: {hub_model_id}"
            )
            self.args.hub_model_id = hub_model_id

        visibility = ModelVisibility.PRIVATE if self.args.hub_private_repo else ModelVisibility.PUBLIC
        try:
            api.create_model(hub_model_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass

        if (os.path.exists(self.args.output_dir)
                and os.listdir(self.args.output_dir)
                and self.args.overwrite_output_dir and at_init):
            # directory not empty.
            shutil.rmtree(self.args.output_dir)
        self.repo = Repository(self.args.output_dir, hub_model_id)
        self.repo.push_to_hub = MethodType(_push_to_hub, self.repo)
        self.repo.local_dir = self.repo.model_dir  # hf compatibility

        # By default, ignore the checkpoint folders
        _commit_message = 'Add `{}` patterns to .gitignore'
        if not os.path.exists(
                os.path.join(self.args.output_dir, '.gitignore')
        ) and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS:
            self._add_patterns_to_gitignores(
                ['checkpoint-*/'], _commit_message.format('checkpoint-*/'))

        # Add 'runs/' to .gitignore, ignore tensorboard files
        self._add_patterns_to_gitignores(['runs/'],
                                         _commit_message.format('runs/'))

        # Add '*.sagemaker' to .gitignore if using SageMaker
        if os.environ.get('SM_TRAINING_ENV'):
            self._add_patterns_to_gitignores(
                ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                _commit_message.format('*.sagemaker'))

        self.push_in_progress = None

    def push_to_hub(self,
                    commit_message: str = 'End of training',
                    **kwargs) -> None:
        # user calls manually `push_to_hub` with `self.args.push_to_hub = False`
        create_model_card = kwargs.pop('create_model_card', None)
        if not hasattr(self, 'repo'):
            self.init_git_repo()
        self.save_model(_internal_call=True)

        if not self.is_world_process_zero():
            return

        self.repo.push_to_hub(commit_message, **kwargs)
        # push separately the model card to be independant from the rest of the model
        readme_path = os.path.join(self.args.output_dir, 'README.md')
        if create_model_card is None:
            create_model_card = not os.path.exists(readme_path)
        if create_model_card and self.args.should_save:
            model_name = kwargs.pop('model_name', None)
            if model_name is None and self.args.should_save:
                if self.args.hub_model_id is not None:
                    model_name = self.args.hub_model_id.split('/')[-1]
                else:
                    model_name = os.path.basename(self.args.output_dir)
            self.create_model_card(model_name=model_name, **kwargs)
            self.repo.push_to_hub('update model card README.md', **kwargs)


class SwiftMixin:

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset,
                                              Dict[str, HfDataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer,
                                   torch.optim.lr_scheduler.LambdaLR] = (None,
                                                                         None),
                 preprocess_logits_for_metrics: Optional[Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        check_model = kwargs.get('check_model', True)
        if check_model and hasattr(model, 'model_dir'):
            check_local_model_is_latest(
                model.model_dir,
                user_agent={
                    Invoke.KEY: Invoke.LOCAL_TRAINER,
                    Invoke.THIRD_PARTY: Invoke.SWIFT
                })
        # mro
        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics,
                         callbacks, optimizers, preprocess_logits_for_metrics)

        if get_function(model.__class__.forward) is not get_function(
                model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)

    @staticmethod
    def _create_configuration_file(model: Module, output_dir: str) -> None:
        cfg = getattr(model, 'cfg', {})
        configuration_path = os.path.join(output_dir, 'configuration.json')
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r') as f:
                res = json.load(f)
        else:
            res = {}
        if 'framework' not in res:
            res['framework'] = cfg.get('framework', 'pytorch')
        if 'task' not in res:
            res['task'] = cfg.get('task', 'text-generation')
        with open(configuration_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')
        self._create_configuration_file(self.model, output_dir)

        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        # save model, tokenizer, args
        save_safetensors = getattr(self.args, 'save_safetensors', False)
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                _unwrap_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=save_safetensors)
            else:
                logger.info(
                    'Trainer.model is not a `PreTrainedModel`, only saving its state dict.'
                )
                if save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, 'model.safetensors'))
                else:
                    torch.save(state_dict,
                               os.path.join(output_dir, 'pytorch_model.bin'))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))