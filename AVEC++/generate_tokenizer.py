import csv
import re
import sys
from pathlib import Path
import sentencepiece as spm

# The tokenizer's vocab is only valid if it's trained on text normalized
# *exactly* the way real labels get built -- reuse AVEC/tools/build_lrs2_from_manifest.py's
# clean_reference() directly instead of a hand-rolled duplicate, so this can't
# drift out of sync again. (Previously this file had its own clean_transcript()
# that uppercased the corpus while clean_reference() lowercases -- BPE pieces
# are case-sensitive, so the trained vocab had ~zero overlap with real labels
# and every token decoded as SentencePiece's <unk> placeholder, "⁇".)
sys.path.insert(0, str(Path(__file__).resolve().parent / "AVEC" / "tools"))
from build_lrs2_from_manifest import clean_reference as clean_transcript

# 1. Define paths
manifest_path = "/Users/alexiswisniewski/Desktop/AVSR/data/voxmm_and_lrs-voxmm/lrs-voxmm_avec_roi/manifest.csv"
combined_text_path = "cleaned_transcripts.txt"
output_tokenizer_dir = Path("tokenizer")

# 3. Extract and clean references from manifest.csv
with open(manifest_path, "r", encoding="utf-8") as csv_file, \
     open(combined_text_path, "w", encoding="utf-8") as txt_file:
    
    reader = csv.DictReader(csv_file)
    for row in reader:
        raw_reference = row["reference"]
        cleaned_reference = clean_transcript(raw_reference)
        
        if cleaned_reference:
            txt_file.write(cleaned_reference + "\n")

# 4. Ensure output directory exists
output_tokenizer_dir.mkdir(parents=True, exist_ok=True)

# 5. Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=combined_text_path,
    model_prefix=str(output_tokenizer_dir / "tokenizerbpe256"),
    vocab_size=256,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

print(f"Done! Tokenizer saved to {output_tokenizer_dir / 'tokenizerbpe256.model'}")