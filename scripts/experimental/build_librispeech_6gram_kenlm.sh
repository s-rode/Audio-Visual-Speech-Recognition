#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

kenlm_archive="external/src/kenlm.tar.gz"
kenlm_src="external/src/kenlm-master"
lm_dir="external/assets/lm/librispeech-6gram"
text_gz="$lm_dir/librispeech-lm-norm.txt.gz"
vocab="external/assets/lm/librispeech-4gram/unigrams.txt"
limit_vocab="$lm_dir/vocab.200k.txt"
tmp_dir="$repo_root/.tmp/kenlm-6gram"
lm_order="${KENLM_ORDER:-6}"
lm_prune="${KENLM_PRUNE:-0 0 0 1 1 1}"

mkdir -p external/src "$lm_dir" "$tmp_dir" .tmp/kenlm-build-tmp

if [[ ! -f "$kenlm_archive" ]]; then
  wget --no-config -c -O "$kenlm_archive" \
    https://github.com/kpu/kenlm/archive/refs/heads/master.tar.gz
fi

if [[ ! -d "$kenlm_src" ]]; then
  tar -xzf "$kenlm_archive" -C external/src
fi

if [[ ! -f "$text_gz" ]]; then
  wget --no-config -c -O "$text_gz" \
    https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
fi

if [[ ! -f "$vocab" ]]; then
  echo "Missing vocabulary: $vocab" >&2
  echo "Download or generate the LibriSpeech unigram vocabulary first." >&2
  exit 1
fi

awk '$1!="<s>" && $1!="</s>" && $1!="<unk>" {print $1}' "$vocab" > "$limit_vocab"

TMPDIR="$repo_root/.tmp/kenlm-build-tmp" \
  cmake -S "$kenlm_src" -B "$kenlm_src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release

TMPDIR="$repo_root/.tmp/kenlm-build-tmp" \
  cmake --build "$kenlm_src/build" --target lmplz build_binary -j "$(nproc)"

read -r -a prune_args <<< "$lm_prune"

zcat "$text_gz" | "$kenlm_src/build/bin/lmplz" \
  -o "$lm_order" \
  --limit_vocab_file "$limit_vocab" \
  --prune "${prune_args[@]}" \
  --discount_fallback \
  --temp_prefix "$tmp_dir" \
  > "$lm_dir/${lm_order}-gram.pruned.arpa"

"$kenlm_src/build/bin/build_binary" \
  "$lm_dir/${lm_order}-gram.pruned.arpa" \
  "$lm_dir/${lm_order}-gram.pruned.bin"

ls -lh "$lm_dir/${lm_order}-gram.pruned.arpa" "$lm_dir/${lm_order}-gram.pruned.bin"
