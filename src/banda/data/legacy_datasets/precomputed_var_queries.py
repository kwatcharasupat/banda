import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from bandx.data.moisesdb.legacy_dataset import (
    FINE_LEVEL_INSTRUMENTS,
    MoisesDBBaseDataset,
    MoisesDBFullTrackDataset,
)
import sys

_PENDING_TRANSFER = not True


class MoisesDBPrecomputedVarQueryDataset(MoisesDBBaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        npy_memmap: bool=True,
        target_length=None,
        data_type: str="npz",
        # variant="var-radius-passt-pca-unconst-test-all",
        variant: str="var-radius-passt-pca-unconst",
        flatten_cov: bool=not False,
        deterministic_unif: float=0.5,
        fixed_radii=None,
        ignore_ids: str="/home/kwatchar3/projects/coda2/visualize/ijcai/ignore-test.csv",
    ) -> None:
        super().__init__(
            split=split,
            data_path=data_root,
            npy_memmap=npy_memmap,
        )

        self.data_type = data_type
        self.variant = variant

        if split in ["test", "val"]:
            # df = {
            #     "val": pd.read_csv(
            #         os.path.join(
            #             data_root, "var-radius-passt", "indices", "sampled_val.csv"
            #         )
            #     )
            # }

            print("Loading val dataset")
            paths = sorted(
                glob.glob(
                    os.path.join(data_root, self.variant, "indices", split, "*.csv")
                )
            )

            df = {
                os.path.basename(path).split(".")[0]: pd.read_csv(path)
                .sample(frac=1)
                .reset_index(drop=True)
                for path in tqdm(paths)
            }

        elif split == "train":
            print("Loading train dataset")
            paths = sorted(
                glob.glob(
                    os.path.join(data_root, self.variant, "indices", "train", "*.csv")
                )
            )

            print(paths)

            df = {
                os.path.basename(path).split(".")[0]: pd.read_csv(path)
                .sample(frac=1)
                .reset_index(drop=True)
                for path in tqdm(paths)
            }
        else:
            raise ValueError(f"Invalid split {split}")

        print(df)

        self.df = pd.concat(df.values(), ignore_index=True).reset_index(drop=True)

        if ignore_ids is not None:
            df_ignore = pd.read_csv(ignore_ids)

            print("Ignoring ", len(df_ignore), " samples")

            comb_ids = df_ignore["comb_id"].values
            self.df = self.df[~self.df["combination_uuid_str"].isin(comb_ids)]

            # exit()

        self.true_length = len(self.df)
        print(f"Loaded {split} dataset with {self.true_length} samples")

        self.flatten_cov = flatten_cov

        # self.n_keys = len(df)
        # self.df_keys = list(df.keys())
        # self.offsets = np.concatenate(
        #     [[0], np.cumsum([len(sub_df) for sub_df in df.values()])]
        # )
        # self.true_length = sum([len(sub_df) for sub_df in df.values()])

        # print(self.offsets)

        print(f"Loaded {split} dataset with {self.true_length} samples")

        self.target_length = (
            target_length if target_length is not None else self.true_length
        )

        if self.target_length < self.true_length:
            if self.split == "val":
                dfg = self.df.groupby(["song_id", "n_in_mix", "n_in_target"])

                group_sizes = [len(group) for _, group in dfg]
                group_keys = list(dfg.groups.keys())

                group_sorted = np.argsort(group_sizes)

                n_allocated = 0
                n_remaining = self.target_length
                dfs = []

                n_groups = len(group_sizes)

                for group_idx in group_sorted:
                    if n_remaining <= 0:
                        raise ValueError("n_remaining <= 0")

                    n_per_group = n_remaining // (n_groups - len(dfs))

                    group = dfg.get_group(group_keys[group_idx])
                    group_size = len(group)

                    if group_size <= n_per_group:
                        dfs.append(group)
                        n_allocated += group_size
                        n_remaining -= group_size
                    else:
                        every_n = group_size // n_per_group
                        dfs.append(group.iloc[::every_n])
                        n_allocated += n_per_group
                        n_remaining -= n_per_group

                # print([len(dfss) for dfss in dfs])

                df = pd.concat(dfs, ignore_index=True)
                self.df = df

                self.true_length = len(self.df)

                # print(f"Downsampled {split} dataset to {self.true_length} samples")

                # raise NotImplementedError

        self.deterministic_unif = deterministic_unif
        self.fixed_radii = fixed_radii

        print(f"Variant: {self.variant}")
        print(
            f"Loaded {split} dataset with {self.true_length} samples and target length {self.target_length}"
        )
        # exit()

    def __len__(self) -> int:
        return self.target_length

    def get_stem(self, *, stem: str, identifier: pd.Series):
        start_sample = identifier.start
        end_sample = identifier.end

        song_id = identifier.song_id
        path = os.path.join(self.data_path, "npy2", song_id)

        assert self.npy_memmap

        if os.path.exists(os.path.join(path, f"{stem}.npy")):
            audio = np.load(os.path.join(path, f"{stem}.npy"), mmap_mode="r")
        else:
            raise FileNotFoundError(f"{song_id, stem}")

        audio = audio[:, start_sample:end_sample]

        return audio

    def _get_query(self, identifier: pd.Series):
        path = os.path.join(
            self.data_path,
            self.variant,
            "passt",
            identifier.song_id,
            str(identifier.n_in_mix),
            str(identifier.n_in_target),
            f"{identifier.combination_uuid_str}.{self.data_type}",
        )

        if _PENDING_TRANSFER:
            if not os.path.exists(path):
                # warnings.warn(f"Path {path} does not exist")
                return None

        if self.npy_memmap:
            # print(self.npy_memmap)
            query = np.load(path, mmap_mode="r")
        else:
            raise NotImplementedError

        # if self.data_type == "npz":
        # query = query["passt"]
        # print(query.keys())
        # raise NotImplementedError

        return query

    def get_identifier(self, index: int):
        return self.df.iloc[index]

    def old_get_identifier(self, index: int):
        if self.n_keys == 1:
            sub_df_key = self.df_keys[0]
            sub_df = self.df[sub_df_key]
            return sub_df.iloc[index]

        # sub_df_idx = np.searchsorted(self.offsets, index, side="right") - 1
        # sub_df_key = self.df_keys[sub_df_idx]
        # sub_df = self.df[sub_df_key]
        # sub_idx = index - (self.offsets[sub_df_idx-1] if sub_df_idx > 0 else 0)

        sub_df_idx = np.searchsorted(self.offsets, index, side="right")
        sub_df_key = self.df_keys[sub_df_idx]
        sub_df = self.df[sub_df_key]
        sub_idx = index - self.offsets[sub_df_idx]

        return sub_df.iloc[sub_idx]

    def _get_audio(self, identifier: pd.Series):
        target_stems = identifier[identifier == "T"].index
        mixture_stems = identifier[identifier == "M"].index

        target_stem_audio = {
            stem: self.get_stem(stem=stem, identifier=identifier)
            for stem in target_stems
        }

        mixture_stem_audio = {
            stem: self.get_stem(stem=stem, identifier=identifier)
            for stem in mixture_stems
        }

        return target_stem_audio, mixture_stem_audio

    def __getitem__(self, index: int):
        try:
            if self.split == "train":
                index = np.random.randint(0, self.true_length)

            identifier = self.get_identifier(index)

            target_stem_audio, mixture_stem_audio = self._get_audio(
                identifier=identifier
            )
            passt = self._get_query(identifier=identifier)

            if _PENDING_TRANSFER and passt is None:
                return None

            if self.split == "train":
                unif = np.random.rand()
            else:
                unif = self.deterministic_unif

            if self.data_type == "npy":
                centroid = passt[0, :]
                inner_radii = passt[1, :]
                outer_radii = passt[2, :]
            elif self.data_type == "npz":
                centroid = passt["centroid"]
                inner_radii = passt["inner_radii"]
                outer_radii = passt["outer_radii"]
                eigvecs = passt["eigvecs"]

            # old way
            # new way

            if self.data_type == "npy":
                radii = inner_radii + unif * (outer_radii - inner_radii)
                eigvals = np.square(radii)
                query = np.concatenate([centroid, radii], axis=0)
            elif self.data_type == "npz":
                if self.fixed_radii is None:
                    inner_radii2 = np.square(inner_radii)
                    outer_radii2 = np.square(outer_radii)
                    eigvals = inner_radii2 + unif * (outer_radii2 - inner_radii2)
                else:
                    # eigvals = np.square(self.fixed_radii, dtype=inner_radii.dtype) * np.ones_like(inner_radii)
                    eigvals = np.minimum(
                        np.square(outer_radii),
                        np.square(self.fixed_radii, dtype=outer_radii.dtype),
                    )

                if self.flatten_cov:
                    # cov = np.dot(eigvecs, np.dot(np.diag(radii), eigvecs.T))
                    cov = np.einsum(
                        "ij,j,kj->ik",
                        eigvecs,
                        eigvals,
                        eigvecs,
                    )
                    triu_indices = np.triu_indices(cov.shape[0])
                    cov_triu = cov[triu_indices]
                    # print(cov_triu.shape, centroid.shape)
                    query = np.concatenate([centroid, cov_triu], axis=0)
                else:
                    max_eigval = np.max(eigvals)

                    eigval_filter = eigvals > max_eigval * 1e-8

                    safe_eigvals = np.where(eigval_filter, eigvals, 1.0)

                    inv_eigvals = 1.0 / safe_eigvals
                    inv_eigvals[~eigval_filter] = 0.0

                    # inv_cov = np.einsum(
                    #     "ij,j,kj->ik",
                    #     eigvecs,
                    #     inv_eigvals,
                    #     eigvecs
                    # )

                    # # print(inv_cov.shape, centroid.shape)

                    # query = {
                    #     "centroid": centroid,
                    #     "inv_cov": inv_cov,
                    # }

                    query = {
                        "centroid": centroid,
                        # "eigvals": eigvals,
                        "inv_eigvals": inv_eigvals,
                        "eigvecs": eigvecs,
                    }

            return {
                "sources": {
                    k: {
                        "audio": v.copy(),
                    }
                    for k, v in target_stem_audio.items()
                },
                "in_mixture": {
                    k: {
                        "audio": v.copy(),
                    }
                    for k, v in mixture_stem_audio.items()
                },
                "query": {
                    "passt": query,
                },
                "metadata": identifier.to_dict(),
            }
        except Exception:
            print(identifier)
            return None


class MoisesDBPrecomputedVarQueryFullTrackTestDataset(MoisesDBBaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        npy_memmap: bool=True,
    ) -> None:
        super().__init__(
            split=split,
            data_path=data_root,
            npy_memmap=npy_memmap,
        )
        self.full_track_dataset = MoisesDBFullTrackDataset(
            data_root=data_root, split=split, return_stems=True
        )

        self.full_track_dataset.song_to_stem = {
            k: [vv for vv in v if vv in FINE_LEVEL_INSTRUMENTS]
            for k, v in self.full_track_dataset.song_to_stem.items()
        }

        self.length = len(self.full_track_dataset)

        query_npz = glob.glob(
            os.path.join(data_root, "var-radius-passt", "specs", "*.npz")
        )

        queries = {
            os.path.basename(path).split(".")[0]: np.load(path) for path in query_npz
        }

        self.queries = {
            k: {kk: vv.astype(np.float32) for kk, vv in v.items()}
            for k, v in queries.items()
        }

    def __getitem__(self, index: int):
        full_track_item = self.full_track_dataset[index]
        stems = full_track_item["sources"].keys()

        full_track_item["queries"] = {
            stem: {"passt": self.queries[stem]}
            for stem in stems
            if stem in self.queries
        }

        return full_track_item

    def __len__(self) -> int:
        return self.length


if __name__ == "__main__":
    data_root = os.path.expandvars("$DATA_ROOT/moisesdb")

    dataset = MoisesDBPrecomputedVarQueryDataset(data_root=data_root, split="val")
    # dataset = MoisesDBPrecomputedVarQueryDataset(data_root=data_root, split="train", target_length=8192)

    print(len(dataset))

    ok = 0
    missing = 0

    for item in tqdm(dataset, total=len(dataset)):
        if item is None:
            missing += 1
        else:
            ok += 1

        if ok + missing == len(dataset):
            print(len(dataset), ok, missing)
            break

    print("Done with ", dataset.split)

    print(ok, missing)

    sys.exit()

    # dataset = MoisesDBPrecomputedVarQueryDataset(data_root=data_root, split="train")

    # print(len(dataset))

    # for item in dataset:
    #     if item is None:
    #         print("*")
    #     else:
    #         print(".")

    # print("Done with train")
