# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from huggingface_hub import HfApi
from transformers import TrainingArguments
from transformers.testing_utils import is_staging_test

from optimum.neuron.trainers import NeuronCacheCallaback, TrainiumTrainer
from optimum.neuron.utils.cache_utils import (
    NEURON_COMPILE_CACHE_NAME,
    NeuronHash,
    get_neuron_cache_path,
    list_files_in_neuron_cache,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .utils import StagingTestMixin, create_dummy_dataset, create_tiny_pretrained_model


@is_trainium_test
@is_staging_test
class NeuronCacheCallabackTestCase(StagingTestMixin, TestCase):
    def test_neuron_hash_for_model(self):
        with TemporaryDirectory() as tmpdirname:
            args = TrainingArguments(tmpdirname)
        model = self.create_tiny_pretrained_model(random_num_linears=True)
        inputs = {
            "x": torch.rand((1,)),
        }

        callback = NeuronCacheCallaback()

        # We first check that no hashes is in the hash cache already.
        self.assertFalse(callback.neuron_hashes)

        callback.neuron_hash_for_model(args, model, inputs)
        neuron_hash = callback.neuron_hashes[(model, (tuple(inputs["x"].shape),), torch.float32)]

        same_neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

        self.assertEqual(neuron_hash, same_neuron_hash, "Neuron hashes should be equal")
        self.assertEqual(len(callback.neuron_hashes.keys()), 1, "There should be only one entry in neuron_hashes.")

    def test_try_to_fetch_cached_model(self):
        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO
        model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            args = TrainingArguments(tmpdirname)
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            model(**inputs)
            neuron_hash = NeuronHash(model, ((8, 1),), torch.float32)
            push_to_cache_on_hub(neuron_hash, Path(tmpdirname) / NEURON_COMPILE_CACHE_NAME)

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            callback = NeuronCacheCallaback()
            args = TrainingArguments(tmpdirname)
            inputs = {"x": torch.rand((24, 1))}
            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

            found_in_cache = callback.try_to_fetch_cached_model(neuron_hash)
            self.assertFalse(found_in_cache, "No model should have been fetched.")

            inputs = {"x": torch.rand((8, 1))}
            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)

            files_before_fetching = list_files_in_neuron_cache(
                callback.tmp_neuron_cache_path, only_relevant_files=True
            )
            tmp_neuron_cache_state = list(callback.tmp_neuron_cache_state)
            neuron_cache_state = list_files_in_neuron_cache(Path(tmpdirname), only_relevant_files=True)

            found_in_cache = callback.try_to_fetch_cached_model(neuron_hash)
            self.assertTrue(found_in_cache, "A model should have been fetched.")

            files_after_fetching = list_files_in_neuron_cache(callback.tmp_neuron_cache_path, only_relevant_files=True)
            new_tmp_neuron_cache_state = list(callback.tmp_neuron_cache_state)
            new_neuron_cache_state = list_files_in_neuron_cache(Path(tmpdirname), only_relevant_files=True)

            files_diff = [f for f in files_after_fetching if f not in files_before_fetching]
            state_diff = [f for f in new_tmp_neuron_cache_state if f not in tmp_neuron_cache_state]
            neuron_cache_files_diff = [f for f in new_neuron_cache_state if f not in neuron_cache_state]

            self.assertNotEqual(files_diff, [])
            self.assertListEqual(files_diff, state_diff)
            self.assertEqual(len(files_diff), len(neuron_cache_files_diff))

    def test_synchronize_temporary_neuron_cache_state(self):
        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            callback = NeuronCacheCallaback()

            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertListEqual(diff, [], "The diff should be empty.")

            model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            # No compilation happens if not printing for some reason...
            print(model(**inputs))
            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertNotEqual(diff, [], "The diff should not be empty.")

            diff = callback.synchronize_temporary_neuron_cache_state()
            self.assertListEqual(
                diff, [], "The diff should be empty because nothing happened since last synchronization"
            )

    def test_synchronize_temporary_neuron_cache(self):
        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO
        model = self.create_tiny_pretrained_model(random_num_linears=True).to("xla")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)
            args = TrainingArguments(tmpdirname)
            callback = NeuronCacheCallaback()

            callback.synchronize_temporary_neuron_cache()
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertListEqual(files_in_repo, [], "Repo should be empty.")
            self.assertListEqual(files_in_cache, [], "Cache should be empty.")

            # Running some compilation.
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            print(model(**inputs))

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertNotEqual(files_in_repo, [], "Repo should not be empty.")
            self.assertNotEqual(files_in_cache, [], "Cache should not be empty.")

            # Using the same inputs, nothing should be uploaded.
            inputs = {"x": torch.rand((8, 1)).to("xla")}
            print(model(**inputs))

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()
            new_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            new_files_in_repo = [f for f in new_files_in_repo if not f.startswith(".")]
            new_files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertListEqual(files_in_repo, new_files_in_repo, "No new file should be in the Hub.")
            self.assertListEqual(files_in_cache, new_files_in_cache, "No new file should be in the cache.")

            # New shahpe, should upload.
            inputs = {"x": torch.rand((24, 1)).to("xla")}
            print(model(**inputs))

            neuron_hash = callback.neuron_hash_for_model(args, model, inputs)
            diff = callback.synchronize_temporary_neuron_cache_state()
            callback.neuron_hash_to_files[neuron_hash].extend(diff)

            callback.synchronize_temporary_neuron_cache()
            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(callback.neuron_cache_path, only_relevant_files=True)
            self.assertNotEqual(files_in_repo, new_files_in_repo, "New files should be in the Hub.")
            self.assertNotEqual(files_in_cache, new_files_in_cache, "New files should be in the cache.")

    def test_train_and_eval(self):
        os.environ["CUSTOM_CACHE_REPO"] = self.CUSTOM_PRIVATE_CACHE_REPO

        # We take a batch size that does not divide the total number of samples.
        num_train_samples = 1000
        per_device_train_batch_size = 32
        dummy_train_dataset = create_dummy_dataset({"x": (1,), "labels": (1,)}, num_train_samples)

        # We take a batch size that does not divide the total number of samples.
        num_eval_samples = 100
        per_device_eval_batch_size = 16
        dummy_eval_dataset = create_dummy_dataset({"x": (1,), "labels": (1,)}, num_eval_samples)

        model = create_tiny_pretrained_model(random_num_linears=True)

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertListEqual(files_in_repo, [], "Repo should be empty.")
            self.assertListEqual(files_in_cache, [], "Cache should be empty.")

            args = TrainingArguments(
                tmpdirname,
                do_train=True,
                do_eval=True,
                bf16=True,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                save_steps=10,
                num_train_epochs=2,
            )
            trainer = TrainiumTrainer(
                model,
                args,
                train_dataset=dummy_train_dataset,
                eval_dataset=dummy_eval_dataset,
            )
            start = time.time()
            trainer.train()
            end = time.time()
            first_training_duration = end - start

            files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            files_in_repo = [f for f in files_in_repo if not f.startswith(".")]
            files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertNotEqual(files_in_repo, [], "Repo should not be empty after first training.")
            self.assertNotEqual(files_in_cache, [], "Cache should not be empty after first training.")

        with TemporaryDirectory() as tmpdirname:
            set_neuron_cache_path(tmpdirname)

            new_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            new_files_in_repo = [f for f in new_files_in_repo if not f.startswith(".")]
            new_files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertNotEqual(new_files_in_repo, [], "Repo should not be empty.")
            self.assertListEqual(new_files_in_cache, [], "Cache should be empty.")

            args = TrainingArguments(
                tmpdirname,
                do_train=True,
                do_eval=True,
                bf16=True,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                save_steps=10,
                num_train_epochs=2,
            )
            trainer = TrainiumTrainer(
                model,
                args,
                train_dataset=dummy_train_dataset,
                eval_dataset=dummy_eval_dataset,
            )
            start = time.time()
            trainer.train()
            end = time.time()
            second_training_duration = end - start

            last_files_in_repo = HfApi().list_repo_files(repo_id=self.CUSTOM_PRIVATE_CACHE_REPO)
            last_files_in_repo = [f for f in last_files_in_repo if not f.startswith(".")]
            last_files_in_cache = list_files_in_neuron_cache(get_neuron_cache_path(), only_relevant_files=True)
            self.assertListEqual(
                files_in_repo, last_files_in_repo, "No file should have been added to the Hub after first training."
            )
            self.assertListEqual(
                files_in_cache,
                last_files_in_cache,
                "No file should have been added to the cache after first training.",
            )

            self.assertTrue(
                second_training_duration < first_training_duration,
                "Second training should be faster because cached graphs can be used.",
            )
