// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// launchVariantRe accepts the variant after the colon in "org/name:VARIANT",
// such as Q4_K_M or UD-Q4_K_M: the rule Splash applies to its own --model.
var launchVariantRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// splitGGUFRe matches one shard of a GGUF split into several files, such as
// "model-00001-of-00003.gguf".
var splitGGUFRe = regexp.MustCompile(`-\d{5}-of-\d{5}\.gguf$`)

// validLaunchModelID reports whether model may be written into the launch
// arguments by spec. A plain Hugging Face id is always accepted. With
// spec.Variant, "org/name:VARIANT" is accepted too; the repository part must
// then name an owner, and neither part may hold "--" or "..", which Splash
// refuses as well.
func validLaunchModelID(spec *ActionLaunchModel, model string) bool {
	if hfModelIDRe.MatchString(model) {
		return true
	}
	if !spec.Variant {
		return false
	}
	repo, variant, found := strings.Cut(model, ":")
	return found &&
		strings.Contains(repo, "/") &&
		hfModelIDRe.MatchString(repo) &&
		launchVariantRe.MatchString(variant) &&
		!strings.Contains(model, "--") &&
		!strings.Contains(model, "..")
}

// launchModelFile is the local GGUF file that model names in store, a model
// store in the Hugging Face cache layout:
// models--<org>--<name>/snapshots/<commit>/<file>.gguf, the commit read from
// refs/main.
//
// The variant after the colon picks the file whose name holds it, ignoring
// case. Without a variant the snapshot must hold exactly one GGUF. Vision
// projectors (mmproj*) are never the model. A model that matches no file, or
// more than one, is refused rather than guessed, and so is a GGUF split into
// shards, which the engines that use this take as a directory rather than a
// file.
func launchModelFile(store, model string) (string, error) {
	repo, variant, _ := strings.Cut(model, ":")
	snapshot, ok := hfSnapshotDir(store, repo, "main")
	if !ok {
		return "", fmt.Errorf("%s is not in the model store %s; copy it there first", repo, store)
	}
	entries, err := os.ReadDir(snapshot)
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", snapshot, err)
	}
	want := strings.ToLower(variant)
	var matches []string
	for _, entry := range entries {
		name := entry.Name()
		lower := strings.ToLower(name)
		if entry.IsDir() || filepath.Ext(lower) != ".gguf" || strings.HasPrefix(lower, "mmproj") {
			continue
		}
		if want != "" && !strings.Contains(lower, want) {
			continue
		}
		matches = append(matches, name)
	}
	slices.Sort(matches)
	switch {
	case len(matches) == 0 && variant != "":
		return "", fmt.Errorf("%s has no GGUF file for variant %s in %s", repo, variant, snapshot)
	case len(matches) == 0:
		return "", fmt.Errorf("%s has no GGUF file in %s", repo, snapshot)
	case len(matches) > 1:
		return "", fmt.Errorf("%s matches more than one GGUF file in %s (%s); name the variant exactly", model, snapshot, strings.Join(matches, ", "))
	case splitGGUFRe.MatchString(strings.ToLower(matches[0])):
		return "", fmt.Errorf("%s is a GGUF split into several files, which is not supported", model)
	}
	return filepath.Join(snapshot, matches[0]), nil
}

// ggufListSpec is the launch_model shape whose ids scanGGUFStore produces.
var ggufListSpec = &ActionLaunchModel{Variant: true}

// scanGGUFStore lists the GGUF models in store, a model store in the Hugging
// Face cache layout, for an engine that loads them with launch_model and
// file_flag. Each usable file is one entry, "org/name:VARIANT", the variant
// being the shortest run of the file name's trailing dash-separated parts that
// launchModelFile resolves back to that file and no other, such as Q4_K_M for
// Qwen3.8-27B-Q4_K_M.gguf, which is also how llama.cpp names it. A file no
// variant singles out is left out rather than listed under a name that would
// load something else. A missing store is an empty one.
func scanGGUFStore(store string) ([]hfCacheModel, error) {
	repos, err := os.ReadDir(store)
	if errors.Is(err, fs.ErrNotExist) {
		return []hfCacheModel{}, nil
	}
	if err != nil {
		return nil, err
	}
	models := []hfCacheModel{}
	for _, entry := range repos {
		org, name, ok := strings.Cut(strings.TrimPrefix(entry.Name(), "models--"), "--")
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "models--") || !ok {
			continue
		}
		repo := org + "/" + name
		snapshot, ok := hfSnapshotDir(store, repo, "main")
		if !ok {
			continue
		}
		files, err := os.ReadDir(snapshot)
		if err != nil {
			continue
		}
		for _, file := range files {
			lower := strings.ToLower(file.Name())
			if filepath.Ext(lower) != ".gguf" || strings.HasPrefix(lower, "mmproj") || splitGGUFRe.MatchString(lower) {
				continue
			}
			path := filepath.Join(snapshot, file.Name())
			id, ok := ggufModelID(store, repo, path)
			if !ok {
				continue
			}
			info, err := os.Stat(path)
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			models = append(models, hfCacheModel{ID: id, Size: info.Size()})
		}
	}
	slices.SortFunc(models, func(a, b hfCacheModel) int { return strings.Compare(a.ID, b.ID) })
	return models, nil
}

// ggufModelID is the id that names path, a GGUF file of repo in store.
func ggufModelID(store, repo, path string) (string, bool) {
	stem := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	parts := strings.Split(stem, "-")
	for k := 1; k <= len(parts); k++ {
		id := repo + ":" + strings.Join(parts[len(parts)-k:], "-")
		if !validLaunchModelID(ggufListSpec, id) {
			continue
		}
		if resolved, err := launchModelFile(store, id); err == nil && resolved == path {
			return id, true
		}
	}
	return "", false
}
