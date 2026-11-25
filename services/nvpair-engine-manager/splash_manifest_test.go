// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"nvpair-shared/engines"
)

func bundledSplash(t *testing.T) *Manifest {
	t.Helper()
	reg := loadWithOverrides(t, t.TempDir())
	manifest, ok := reg.Get(engines.NameSplash)
	require.True(t, ok, "splash manifest not loaded")
	return manifest
}

// Splash ships only for Apple Silicon on macOS 26, and the release it installs
// is pinned by checksum.
func TestBundledSplashManifest_PinsAnAppleSiliconRelease(t *testing.T) {
	m := bundledSplash(t)

	require.Len(t, m.Platforms, 1, "Splash runs only on darwin/arm64")
	p, ok := m.Platforms["darwin/arm64"]
	require.True(t, ok)
	require.NotNil(t, p.Install)
	require.NotNil(t, p.Install.Fetch)
	assert.True(t, strings.HasPrefix(p.Install.Fetch.URL, "https://github.com/incoai/splash/releases/download/"), p.Install.Fetch.URL)
	assert.True(t, strings.HasSuffix(p.Install.Fetch.URL, "-arm64-macos26.tar.gz"), p.Install.Fetch.URL)
	assert.Len(t, p.Install.Fetch.SHA256, 64, "the release must be pinned by sha256")
}

// Splash serves models already on the Mac. It runs offline on loopback, reads
// the store llama.cpp shares, and is placed on the ports the engine table
// gives it.
func TestBundledSplashManifest_ServesOfflineFromTheSharedStore(t *testing.T) {
	m := bundledSplash(t)
	p := m.Platforms["darwin/arm64"]
	identity, ok := engines.ByName(engines.NameSplash)
	require.True(t, ok)

	assert.Equal(t, "~/.llamacpp", p.ModelsDir, "Splash shares llama.cpp's store")
	assert.Equal(t, "{models_dir}", p.Runtime.Env["HF_HUB_CACHE"], "the draft is found in the same store")
	assert.Contains(t, p.Runtime.EditableLaunch.FixedArgs, "--offline", "Splash must never reach the Hub")
	assert.Equal(t, "127.0.0.1", p.Runtime.Bind)
	assert.Equal(t, identity.EnginePortBase, p.Runtime.Port)
	assert.NotEqual(t, identity.FacadePort, p.Runtime.Port)
	assert.True(t, p.Runtime.WaitForModel, "Splash has no default model to start on")
	assert.Equal(t, advertisedModelsLoaded, m.AdvertisedModels, "only the served model is advertised")
}

func TestBundledSplashManifest_DeclaresTheModelActionsTheStackReliesOn(t *testing.T) {
	m := bundledSplash(t)

	load := m.Actions["load_model"].LaunchModel
	require.NotNil(t, load, "load_model must switch the launch model")
	assert.Equal(t, ActionLaunchModel{Flag: "--model", Variant: true, FileFlag: "--model-dir"}, *load)
	assert.True(t, m.Actions["unload_model"].StopEngine, "unload stops the one-model process")
	loaded := m.Actions["loaded_models"]
	require.NotNil(t, loaded.HTTP)
	assert.Equal(t, "/v1/models", loaded.HTTP.Path)
	assert.Equal(t, &ActionResult{Array: "data", Field: "id"}, loaded.Result)
	list := m.Actions["list_models"].HFCache
	require.NotNil(t, list, "list_models must read the store without a running engine")
	assert.Equal(t, ActionHFCache{Dir: "{models_dir}", GGUF: true}, *list)
	for _, name := range []string{"pull_model", "delete_model"} {
		_, ok := m.Actions[name]
		assert.False(t, ok, "%s is out of v1 (models are copied into the store by hand)", name)
	}
}

func TestBundledSplashInstallScript_SurvivesArgumentExpansion(t *testing.T) {
	p := bundledSplash(t).Platforms["darwin/arm64"]
	argv, err := resolveArgs(p.Install.Run, map[string]string{
		"download":    "/tmp/splash.tar.gz",
		"install_dir": "/tmp/splash",
	})

	require.NoError(t, err)
	for i := range argv {
		assert.Equal(t, argv[i], expandPathForOS(argv[i], "darwin"), "the executor rewrites install argument %d, which breaks the script", i)
	}
}

// The install script unpacks the release and writes the wrapper the runtime
// launches, which runs Splash's own Python on its launcher with the launch
// arguments. A stand-in release keeps the test small; its python3 records the
// arguments it was given.
func TestBundledSplashInstallScript_WritesAWrapperThatRunsTheLauncher(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("the install script runs only on darwin/arm64")
	}
	if major, err := macOSMajor(); err != nil || major < 26 {
		t.Skipf("the install script needs macOS 26 or newer (found %d, %v)", major, err)
	}
	p := bundledSplash(t).Platforms["darwin/arm64"]
	work := t.TempDir()
	release := filepath.Join(work, "release.tar.gz")
	writeTarGz(t, release, map[string]string{
		"splash-9.9.9-arm64-macos26/python/bin/python3":  "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname -- \"$0\")/../../args.txt\"\n",
		"splash-9.9.9-arm64-macos26/install/launcher.py": "# stand-in\n",
	})
	installDir := filepath.Join(work, "install dir with spaces")
	require.NoError(t, os.MkdirAll(installDir, 0o755))
	argv, err := resolveArgs(p.Install.Run, map[string]string{"download": release, "install_dir": installDir})
	require.NoError(t, err)

	out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput()
	require.NoError(t, err, "install script: %s", out)
	out, err = exec.Command(filepath.Join(installDir, "splash"), "serve", "--port", "8096").CombinedOutput()
	require.NoError(t, err, "wrapper: %s", out)

	args, err := os.ReadFile(filepath.Join(installDir, "args.txt"))
	require.NoError(t, err)
	assert.Equal(t, "-u\n"+filepath.Join(installDir, "install", "launcher.py")+"\nserve\n--port\n8096\n", string(args))
}

func macOSMajor() (int, error) {
	out, err := exec.Command("sw_vers", "-productVersion").Output()
	if err != nil {
		return 0, err
	}
	major, _, _ := strings.Cut(strings.TrimSpace(string(out)), ".")
	return strconv.Atoi(major)
}

func writeTarGz(t *testing.T, path string, files map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, f.Close()) }()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		require.NoError(t, tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(body)), Typeflag: tar.TypeReg}))
		_, err := tw.Write([]byte(body))
		require.NoError(t, err)
	}
	require.NoError(t, tw.Close())
	require.NoError(t, gz.Close())
}
