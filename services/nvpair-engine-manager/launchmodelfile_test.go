// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ggufStore lays out a model store in the Hugging Face cache layout, the way
// GGUFs copied in by hand sit in ~/.llamacpp, and returns the store and the
// snapshot directory that holds files.
func ggufStore(t *testing.T, repo string, files ...string) (store, snapshot string) {
	t.Helper()
	store = t.TempDir()
	const commit = "0123456789abcdef0123456789abcdef01234567"
	repoDir := filepath.Join(store, "models--"+strings.ReplaceAll(repo, "/", "--"))
	require.NoError(t, os.MkdirAll(filepath.Join(repoDir, "refs"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "refs", "main"), []byte(commit+"\n"), 0o644))
	snapshot = filepath.Join(repoDir, "snapshots", commit)
	require.NoError(t, os.MkdirAll(snapshot, 0o755))
	for _, name := range files {
		require.NoError(t, os.WriteFile(filepath.Join(snapshot, name), []byte("GGUF"), 0o644))
	}
	return store, snapshot
}

func TestValidLaunchModelID_AcceptsAVariantOnlyWhenTheManifestAllowsIt(t *testing.T) {
	plain := &ActionLaunchModel{Flag: "--model"}
	variant := &ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir"}
	cases := []struct {
		name  string
		spec  *ActionLaunchModel
		model string
		want  bool
	}{
		{"a plain id without variant support", plain, "Qwen/Qwen3.5-4B", true},
		{"a variant id without variant support", plain, "local/Qwen3.8-27B-GGUF:Q4_K_M", false},
		{"a plain id with variant support", variant, "Qwen/Qwen3.5-4B", true},
		{"a variant id with variant support", variant, "local/Qwen3.8-27B-GGUF:Q4_K_M", true},
		{"a variant with a dash", variant, "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", true},
		{"a variant with no owner", variant, "Qwen3.8-27B-GGUF:Q4_K_M", false},
		{"an empty variant", variant, "local/Qwen3.8-27B-GGUF:", false},
		{"two variants", variant, "local/Qwen3.8-27B-GGUF:Q4:K", false},
		{"a path in the variant", variant, "local/Qwen3.8-27B-GGUF:../x", false},
		{"a double dash in the repository", variant, "local/Qwen--x:Q4_K_M", false},
		{"an option in the variant", variant, "local/name:-x", false},
		{"a space in the variant", variant, "local/name:Q4 K", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, validLaunchModelID(tc.spec, tc.model), "%q", tc.model)
		})
	}
}

func TestLaunchModelFile_FindsTheGGUFTheVariantNames(t *testing.T) {
	store, snapshot := ggufStore(t, "local/Qwen3.8-27B-GGUF",
		"Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B-Q8_0.gguf", "mmproj-Q4_K_M.gguf", "README.md")

	file, err := launchModelFile(store, "local/Qwen3.8-27B-GGUF:q4_k_m")

	require.NoError(t, err)
	assert.Equal(t, filepath.Join(snapshot, "Qwen3.8-27B-Q4_K_M.gguf"), file, "the variant is matched ignoring case, and the vision projector is never the model")
}

func TestLaunchModelFile_UsesTheOnlyGGUFWhenNoVariantIsNamed(t *testing.T) {
	store, snapshot := ggufStore(t, "local/Small-GGUF", "small.gguf", "mmproj-small.gguf")

	file, err := launchModelFile(store, "local/Small-GGUF")

	require.NoError(t, err)
	assert.Equal(t, filepath.Join(snapshot, "small.gguf"), file)
}

func TestLaunchModelFile_RefusesAModelItCannotPickExactly(t *testing.T) {
	cases := []struct {
		name    string
		files   []string
		model   string
		message string
	}{
		{"a repository not in the store", nil, "local/Missing-GGUF:Q4_K_M", "is not in the model store"},
		{"a variant with no file", []string{"m-Q8_0.gguf"}, "local/Model-GGUF:Q4_K_M", "no GGUF file for variant Q4_K_M"},
		{"a variant that matches two files", []string{"m-Q4_K_M.gguf", "m-UD-Q4_K_M.gguf"}, "local/Model-GGUF:Q4_K_M", "more than one GGUF file"},
		{"no variant and two files", []string{"m-Q4_K_M.gguf", "m-Q8_0.gguf"}, "local/Model-GGUF", "more than one GGUF file"},
		{"a split GGUF", []string{"m-Q4_K_M-00001-of-00002.gguf"}, "local/Model-GGUF:Q4_K_M", "split into several files"},
		{"no GGUF at all", []string{"config.json"}, "local/Model-GGUF", "has no GGUF file"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store, _ := ggufStore(t, "local/Model-GGUF", tc.files...)

			_, err := launchModelFile(store, tc.model)

			assert.ErrorContains(t, err, tc.message)
		})
	}
}

func TestLaunchModelAction_FileFlagNeedsVariantAndItsOwnFlag(t *testing.T) {
	cases := []struct {
		name    string
		spec    ActionLaunchModel
		message string
	}{
		{"without variant", ActionLaunchModel{Flag: "--model", FileFlag: "--model-dir"}, "requires launch_model.variant"},
		{"the same flag twice", ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model"}, "must be a flag other than"},
		{"not a flag", ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "model-dir"}, "must be a flag other than"},
		{"a value in the flag", ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir=x"}, "must be a flag other than"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := tc.spec
			assert.ErrorContains(t, (&Action{LaunchModel: &spec}).validate("load_model"), tc.message)
		})
	}
	valid := ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir"}
	assert.NoError(t, (&Action{LaunchModel: &valid}).validate("load_model"))
}

func TestLaunchModelAction_FileFlagRequiresAModelsDir(t *testing.T) {
	m := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.EditableLaunch = &EditableLaunch{FixedArgs: []string{"serve"}, Controls: []LaunchControl{{Value: "{server.host}:{server.port}", Env: []string{"OLLAMA_HOST"}}}}
	p.Runtime.Ready = &Probe{TCP: "127.0.0.1:{port}"}
	p.ModelsDir = ""
	m.Platforms[key] = p
	m.Actions["load_model"] = Action{LaunchModel: &ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir"}}

	assert.ErrorContains(t, m.Validate(), "launch_model.file_flag requires platform")
}

// The Splash shape end to end: a variant id is written as --model, and the
// file it names in the store as --model-dir, then the engine restarts on both.
func TestLaunchModel_WritesTheFileTheModelNamesBesideTheModel(t *testing.T) {
	store, snapshot := ggufStore(t, "local/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B-Q8_0.gguf")
	e := switchModelExecutor(t, "--model")
	m, ok := e.reg.Get("fake")
	require.True(t, ok)
	m.Actions["load_model"] = Action{LaunchModel: &ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir"}}
	st := settingsState(t, e)
	st.mu.Lock()
	st.modelsDir = store
	st.mu.Unlock()

	raw, err := loadModel(t, e, "local/Qwen3.8-27B-GGUF:Q4_K_M")

	require.NoError(t, err)
	assert.Equal(t, loadReply{Model: "local/Qwen3.8-27B-GGUF:Q4_K_M", Restarted: true}, decodeLoadReply(t, raw))
	text := launchTextOf(t, e)
	assert.Contains(t, text, "--model local/Qwen3.8-27B-GGUF:Q4_K_M")
	assert.Contains(t, text, "--model-dir "+filepath.Join(snapshot, "Qwen3.8-27B-Q4_K_M.gguf"))
	assert.True(t, mustStatus(t, e).Running, "the engine serves the new model")

	// Switching variants replaces both flags instead of adding a second pair.
	_, err = loadModel(t, e, "local/Qwen3.8-27B-GGUF:Q8_0")
	require.NoError(t, err)
	text = launchTextOf(t, e)
	assert.Contains(t, text, "--model-dir "+filepath.Join(snapshot, "Qwen3.8-27B-Q8_0.gguf"))
	assert.NotContains(t, text, "Q4_K_M")

	// A model the store does not hold is refused and changes nothing.
	gen := startGeneration(t, e)
	_, err = loadModel(t, e, "local/Missing-GGUF:Q4_K_M")
	assert.ErrorContains(t, err, "is not in the model store")
	assert.Equal(t, text, launchTextOf(t, e), "a refused switch changed the launch text")
	assert.Equal(t, gen, startGeneration(t, e), "a refused switch restarted the engine")
}

func TestScanGGUFStore_NamesEachFileSoLoadFindsItAgain(t *testing.T) {
	store, _ := ggufStore(t, "local/Qwen3.8-27B-GGUF",
		"Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf", "Qwen3.8-27B-Q8_0.gguf",
		"mmproj-F16.gguf", "Big-Q4_K_M-00001-of-00002.gguf", "README.md")
	other, _ := ggufStore(t, "local/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-MXFP4_MOE.gguf")
	require.NoError(t, os.Rename(filepath.Join(other, "models--local--Qwen3.6-35B-A3B-GGUF"), filepath.Join(store, "models--local--Qwen3.6-35B-A3B-GGUF")))
	require.NoError(t, os.MkdirAll(filepath.Join(store, "models--incoai--Qwen3.8-27B-DFlash2", "refs"), 0o755))

	models, err := scanGGUFStore(store)

	require.NoError(t, err)
	var ids []string
	for _, m := range models {
		ids = append(ids, m.ID)
		assert.Equal(t, int64(len("GGUF")), m.Size, "%s size", m.ID)
	}
	// Q4_K_M alone would also match the UD file, so that one keeps its
	// "UD-" prefix and the plain one needs the name's preceding part as well.
	assert.Equal(t, []string{
		"local/Qwen3.6-35B-A3B-GGUF:MXFP4_MOE",
		"local/Qwen3.8-27B-GGUF:27B-Q4_K_M",
		"local/Qwen3.8-27B-GGUF:Q8_0",
		"local/Qwen3.8-27B-GGUF:UD-Q4_K_M",
	}, ids, "projectors, shards, other files and repositories without GGUFs are left out")
	for _, id := range ids {
		_, err := launchModelFile(store, id)
		assert.NoError(t, err, "%s must load", id)
	}
}

func TestScanGGUFStore_TreatsAMissingStoreAsEmpty(t *testing.T) {
	models, err := scanGGUFStore(filepath.Join(t.TempDir(), "absent"))

	require.NoError(t, err)
	assert.Empty(t, models)
}

func TestHFCacheAction_GGUFBelongsToListModels(t *testing.T) {
	assert.ErrorContains(t, (&Action{HFCache: &ActionHFCache{Dir: "{models_dir}", GGUF: true}}).validate("loaded_models"), "belongs to list_models")
	assert.NoError(t, (&Action{HFCache: &ActionHFCache{Dir: "{models_dir}", GGUF: true}, Result: &ActionResult{Array: "data", Field: "id"}}).validate("list_models"))
}
