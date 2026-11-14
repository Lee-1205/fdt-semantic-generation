# Local assets and reproducibility

This snapshot deliberately omits model weights, populated optimizers, scaler/RNG
binary recovery state, raw tensor traces and third-party natural-language corpora.
All original local assets remain unchanged. Hash references in historical reports
identify those original files; missing-file links lead here rather than claiming
the tensors are included.

The bundled authored A–H relational TRAIN/DEV metadata supports focused CPU contract
tests. No E2 final holdout, sealed test shard or evaluation registry was opened or
exported. TinyStories/OpenWebText/SQuAD/FineWeb text was not redistributed here;
historical source/license audits remain documentation only.

Original and exported SHA256 values are in `provenance/export_manifest.json`.
Host path normalization changes some text files. Original historical plan hashes
must not be used to assert that rewritten copies are execution-ready. Parent
weights and caches must be supplied and a new bounded plan/preflight prepared
before full-size research can run elsewhere. Closed workers have not been reset.

No new license is granted to third-party assets. Repository visibility is private.
